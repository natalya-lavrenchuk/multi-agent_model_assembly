"""laws.py -- law metadata: description, law-specific assumptions, roles, possible misprint.

    python -m extractor.laws papers/kontoravdi_2010              # run (answers cached)
    python -m extractor.laws papers/kontoravdi_2010 --dry-run    # how many tokens, send nothing
    python -m extractor.laws papers/kontoravdi_2010 --refresh    # ignore the cache

Run after symbols.py (it reads symbols.json). Writes papers/<id>/laws.json, which graph.py reads.

Code first: form (ODE, PDE, algebraic), what the law computes, side condition, printed value, the
section it is in. No balance/constitutive/definition type (removed 26 Oct: an LLM judgement the system
does not need; code's form says what the Constructor needs).
The LLM, one call per section (at most GROUP laws), gets the whole paper, then that section's equations, the text
just before and after each one, the symbols' one-line descriptions from symbols.json, and the
model-wide assumptions already recorded for the Sup (so it does not repeat them). Per law:
    description   quoted if the paper describes the law, otherwise its own judgement (inferred)
    assumptions   specific to this law, each with the paper's words
    roles         which symbol fills which slot of the law ('substrate' -> [GLC])
Checks by code: quotes must be in the paper; an assumption is blocking only if it is a process condition and the quote checks
out, otherwise a warning; a role must name a symbol of
the law (and only algebraic laws that are not a printed value keep roles).
"""
from __future__ import annotations
import json
import os
import re
import sys

from extractor import llm, triage
from extractor.classify import _which, squash
from extractor.graph import _computes, corrected
from extractor.paper import PAPER_NOTE, TEXT_LABELS, clean, load_model, load_snippets, models, with_paper

GROUP = 8                 # laws per call
TEXT_CHARS = 600          # text before / after an equation

PROMPT = """You annotate the equations (laws) of a published mechanistic process model.

You get one section of the paper: its equations in reading order with the text printed around them,
what each symbol means, and the assumptions already recorded for the whole model. For every
equation listed under "Laws", give:

description  One sentence in your own words: what the law states or computes, in process terms
             ("glucose balance in the fed-batch reactor", "Monod-type specific growth rate limited by
             glucose and glutamine"). If the text describes the equation, also copy the describing
             words exactly into quote (text, never the equation itself) and set description_basis
             "stated". Otherwise description_basis "inferred", quote null.
assumptions  Only assumptions specific to this law that the paper states, about the conditions under which the
             law holds or a simplification made in it ("cell death depends only on ammonia", "dilution is
             neglected", "operated at constant pressure"). Do not list what the law itself says: its
             mechanism, kinetic form or the effects it includes or leaves out ("acetate inhibits glucose
             uptake", "Monod-type kinetics", "particles seal off pore entrances", "a cake layer increases
             resistance", "formate hydrogenation is rate-determining"). That is the law, not an assumption, and
             it is already in the description. Do not repeat the model-wide assumptions. quote: the paper's
             exact words, or null if the paper does not say it.
             process_condition: true only if it is about how the user's process is built, fed or run, something
             the user can check for their own process without knowing the model: feed composition (sterile
             feed, no acetate in the feed), operating mode (constant pressure, constant flow, batch, steady
             state), mixing, temperature or pressure regime (isothermal), the equipment or material used (a
             cooling coil, a magnetite catalyst), the phase of the process (exponential growth). false for
             modelling choices: a property value taken from water, a term or temperature dependence neglected,
             a correlation or simplified form chosen, partial pressures used instead of fugacities, a parameter
             fixed or bounded, a mechanism detail.
             An empty list is a fine answer.
roles        The slots a rival law would have to fill. Use these role names when they fit, written
             exactly: substrate, product, inhibitor, biomass, catalyst, saturation constant,
             inhibition constant, rate constant, maximum rate, yield. These names come from reaction and growth kinetics; when the law is of
             another kind (transport, filtration, heat transfer, equilibrium) and none fits, use a
             short plain name instead (driving force, flux, resistance, transported species).
             Only for algebraic rate or property laws (code: algebraic): the species and the constant
             tied to each. Empty for differential equations and for laws that only define a quantity
             from others or give a printed value. Use the symbol exactly as written in the symbol
             list. Not every symbol needs a role.
issue        A misprint you can point to: name the term and say why it is wrong given the text, the
             symbol meanings or the units (a wrong species in a term, a sign, a missing factor), or two
             laws the text calls different but that are printed identically. One sentence; null if
             nothing or if unsure. A standard form is not a misprint (Contois: K X + S in the
             denominator), and neither is notation you find unusual.

Answer every listed equation exactly once. eq is the equation number only ("7"), not the equation.""" + PAPER_NOTE

SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['laws'],
    'properties': {'laws': {'type': 'array', 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['eq', 'description', 'description_basis', 'quote',
                     'assumptions', 'roles', 'issue'],
        'properties': {
            'eq': {'type': 'string'},
            'description': {'type': 'string'},
            'description_basis': {'type': 'string', 'enum': ['stated', 'inferred']},
            'quote': {'type': ['string', 'null']},
            'assumptions': {'type': 'array', 'items': {
                'type': 'object', 'additionalProperties': False,
                'required': ['text', 'quote', 'process_condition'],
                'properties': {'text': {'type': 'string'}, 'quote': {'type': ['string', 'null']},
                               'process_condition': {'type': 'boolean'}}}},
            'roles': {'type': 'array', 'items': {
                'type': 'object', 'additionalProperties': False, 'required': ['role', 'symbol'],
                'properties': {'role': {'type': 'string'}, 'symbol': {'type': 'string'}}}},
            'issue': {'type': ['string', 'null']}}}}}}


def _schema(group) -> dict:
    """SCHEMA with eq limited to this call's equation numbers: the model cannot write the equation
    itself, or a number not asked."""
    s = json.loads(json.dumps(SCHEMA))
    s['properties']['laws']['items']['properties']['eq'] = {'type': 'string', 'enum': list(group)}
    return s


def annotate(paper_dir: str, refresh: bool = False) -> dict:
    formulas, reads, _ = load_model(paper_dir)
    eqs = [str(f.get('eq') or i + 1) for i, f in enumerate(formulas)]
    sym_file = f'{paper_dir}/symbols.json'
    if not os.path.exists(sym_file) and not llm.DRY_RUN:
        raise SystemExit(f'run symbols.py first: {sym_file} not found')
    syms = corrected(paper_dir)[0] if os.path.exists(sym_file) else {}   # with the reviewer's corrections
    cfg = json.load(open(f'{paper_dir}/paper.json'))
    blocks = json.load(open(f'{paper_dir}/paddle_blocks.json'))['blocks']
    snippets = load_snippets(paper_dir)
    all_assumptions = [a for m in models(cfg) for a in m.get('model_assumptions', [])]   # every model of the paper
    model_assumptions = list(dict.fromkeys(a['text'] for a in all_assumptions))

    computes = dict(zip(eqs, _computes(reads)))
    code = {n: {'form': r['form'], 'computes': computes[n], 'condition': r['condition'], 'value': r.get('value')} for n, r in zip(eqs, reads)}
    where = {n: _position(blocks, f) for n, f in zip(eqs, formulas)}

    answers = {}

    def run(title, group):
        msg = with_paper(paper_dir, _message(title, group, eqs, formulas, reads, code, where, blocks, syms, model_assumptions))
        try:
            out = llm.cached_ask(paper_dir, PROMPT, msg, _schema(group), 'laws', 'low',
                                 label=f'laws {", ".join(group)} ({title[:40]})', refresh=refresh)
        except llm.JSONFailed:
            if len(group) == 1:
                print(f'  eq {group[0]}: no valid answer, left for review')
                return
            print(f'  laws {", ".join(group)}: answer did not fit, splitting in two')
            h = len(group) // 2
            run(title, group[:h])
            run(title, group[h:])
            return
        got = (out or {}).get('laws', [])
        for i, a in enumerate(got):
            n = _eq_of(a['eq'], group, i if len(got) == len(group) else None)
            if n and n not in answers:
                answers[n] = a

    for title, group in _groups(eqs, where, blocks):
        run(title, group)

    laws = {}
    for n, r in zip(eqs, reads):
        sec = _section(blocks, where[n])
        entry = {**code[n], 'section': sec}
        a = answers.get(n)
        entry.update(_checked(a, code[n], r, snippets, all_assumptions) if a else
                     {'status': 'pending', 'review': ['no answer from the LLM']})
        laws[n] = entry
    return {'model': llm.MODEL, 'laws': laws}


# ----------------------------------------------------------------------------- code
def _eq_of(said: str, group, position) -> str | None:
    """'7', '(7) \\mu_d = ...', or (older answers) the equation itself: then its place in the answer."""
    m = re.match(r'\s*\(?\s*([A-Za-z]?-?\d+[a-z]?)\s*\)?(?:\s|$)', said)
    if m and m.group(1) in group:
        return m.group(1)
    return group[position] if position is not None else None


ROLE_ALIASES = {'balance quantity': 'balanced quantity', 'limiting substrate': 'substrate',
                'limiting nutrient': 'substrate', 'reactant': 'substrate', 'inlet concentration': 'feed concentration',
                'yield coefficient': 'yield', 'half saturation constant': 'saturation constant',
                'monod constant': 'saturation constant'}


def _role_key(role: str) -> str:
    """'balanced_quantity', 'Balanced Quantity' -> 'balanced quantity'."""
    return ' '.join(role.replace('_', ' ').replace('-', ' ').lower().split())


def has_roles(c: dict) -> bool:
    """Roles are kept for algebraic laws that are not a printed value (code's reading of the equation)."""
    return c.get('form') not in ('ODE', 'PDE') and not c.get('value')


def _position(blocks, f) -> int | None:
    return next((i for i, b in enumerate(blocks) if b['page'] == f['page'] and b['bbox'] == f.get('bbox')), None)


def _is_heading(b) -> bool:
    """A section heading: labelled as one, or a short text line 'Appendix A. ...' / '3.1 Model ...'."""
    t = (b['content'] or '').strip()
    return b['label'] == 'paragraph_title' or \
        b['label'] == 'text' and len(t) < 120 and bool(re.match(r'(Appendix\b|\d+(\.\d+)*\.?\s+[A-Z])', t))


def _section(blocks, i) -> str:
    """The last section heading printed before block i."""
    if i is None:
        return ''
    return next((blocks[k]['content'].strip() for k in range(i, -1, -1) if _is_heading(blocks[k])), '')


def _groups(eqs, where, blocks):
    """Laws by section in reading order, packed into calls of at most GROUP laws: short sections
    share a call (Birol has 6 sections of 1-3 laws), a long one is split."""
    by_sec = {}
    for n in eqs:
        by_sec.setdefault(_section(blocks, where[n]) or '(no heading)', []).append(n)
    pieces = [(t, ns[k:k + GROUP]) for t, ns in by_sec.items() for k in range(0, len(ns), GROUP)]
    titles, group = [], []
    for t, ns in pieces:
        if group and len(group) + len(ns) > GROUP:
            yield '; '.join(titles), group
            titles, group = [], []
        titles.append(t)
        group += ns
    if group:
        yield '; '.join(titles), group


def _text_near(blocks, i, step) -> int | None:
    """The nearest text block before (step -1) or after (step +1) block i. Before: across other equations
    ('The mass balances on glucose and glutamine are:' introduces eqs 8-12). After: only directly after
    (a 'where ...' line), not past the next equation."""
    if i is None:
        return None
    k = i + step
    while 0 <= k < len(blocks):
        b = blocks[k]
        if _is_heading(b):
            return None                                   # a new section
        if b['label'] in TEXT_LABELS and (b['content'] or '').strip():
            return k
        if step > 0 and 'formula' in b['label'] and 'number' not in b['label']:
            return None                                   # the next equation comes first
        k += step
    return None


def _message(title, group, eqs, formulas, reads, code, where, blocks, syms, model_assumptions) -> str:
    """The section's equations in reading order with the text around them, the symbols they use, the
    model-wide assumptions. Text shared by two equations is printed once."""
    items = {}                                            # block index -> line
    for n in group:
        i = where[n]
        r, c = reads[eqs.index(n)], code[n]
        tag = f"code: {c['form']}, computes {c['computes'] or '?'}" + (', printed value' if c.get('value') else '')
        # several equations can sit in one block (a table of ODEs, a braced system split into parts):
        # key on (block, equation) so none of them overwrites another
        items[(i if i is not None else 10 ** 6, 1, eqs.index(n))] = f'({n}) {r["latex"]}    [{tag}]'
        for step in (-1, 1):
            k = _text_near(blocks, i, step)
            if k is not None and (k, 0, 0) not in items:
                t = ' '.join(blocks[k]['content'].split())
                items[(k, 0, 0)] = '    ' + (('...' + t[-TEXT_CHARS:]) if step < 0 and len(t) > TEXT_CHARS else
                                     (t[:TEXT_CHARS] + '...') if len(t) > TEXT_CHARS else t)
    used = list(dict.fromkeys(s for n in group for s in reads[eqs.index(n)]['lhs'] + reads[eqs.index(n)]['rhs']))
    gloss = []
    for s in used:
        d = syms.get(s, {})
        unit = f" [{d['unit']}]" if d.get('unit') else ''
        cls = 'computed' if d.get('computed_by') else d.get('class', '?')
        gloss.append(f"  {s}: {d.get('description', '?')}{unit} ({cls})")
    return (f'Section(s): {title}\n\nText and equations in reading order:\n'
            + '\n'.join(items[k] for k in sorted(items))
            + '\n\nSymbols:\n' + '\n'.join(gloss)
            + '\n\nModel-wide assumptions (already recorded, do not repeat):\n'
            + ('\n'.join(f'  - {a}' for a in model_assumptions) or '  (none recorded)')
            + f'\n\nLaws: {", ".join(group)}')


def _checked(a, c, r, snippets, model_assumptions) -> dict:
    review = []
    texts = [squash(x['text']) for x in snippets]
    found = lambda q: q is not None and len(squash(q)) >= 6 and any(squash(q) in t for t in texts)
    if a['description_basis'] == 'stated' and not a['quote'] and found(a['description']):
        a = {**a, 'quote': a['description']}               # the paper's words written as the description
    d_basis = 'stated' if a['description_basis'] == 'stated' and found(a['quote']) else 'inferred'
    if a['description_basis'] == 'stated' and d_basis == 'inferred':
        review.append('description quote not found in the paper')

    model_q = {squash(m.get('quote') or '') for m in model_assumptions} - {''}
    assumptions = []
    for x in a['assumptions']:
        ok = found(x['quote'])
        if ok and squash(x['quote']) in model_q:
            continue                                      # a model-wide assumption repeated
        # blocking: the paper says it (quote found) AND it is a condition a user can check for their process
        # (sterile feed, no acetate in the feed), however it is phrased ('assuming', 'considering', 'with');
        # a modelling choice inside the law (water's boiling point) only warns. (26 Oct: 'stated as an
        # assumption' is no longer required: it depended on the paper's wording and changed between runs.)
        sev = 'blocking' if ok and x.get('process_condition') else 'warning'
        assumptions.append({'text': x['text'], 'quote': x['quote'] if ok else None, 'severity': sev,
                            'process_condition': bool(x.get('process_condition')),
                            'reasoned': not ok})
        if sev == 'blocking':
            review.append(f"blocking assumption: {x['text']}")
        if x['quote'] and not ok:
            review.append(f"assumption quote not found: {x['text']}")

    law_syms = r['lhs'] + r['rhs']
    roles, dropped = [], []
    # Roles are kept only for constitutive laws: the swappable ones (a rival law has to be bound to the
    # model's variables). Balances and definitions have nothing to swap. The prompt still asks for them so
    # cached answers stay valid; code drops them here (the LLM's raw answer stays under 'llm').
    # Roles only where a rival law could be swapped in: algebraic laws that are not a printed value
    # (decided by code from the equation, no LLM classification). The LLM's raw answer stays under 'llm'.
    for x in (a['roles'] if has_roles(c) else []):
        s = _which(x['symbol'], law_syms)
        role = ROLE_ALIASES.get(_role_key(x['role']), _role_key(x['role']))
        (roles.append({'role': role, 'symbol': s}) if s else dropped.append(x))
    roles = [dict(t) for t in dict.fromkeys(tuple(x.items()) for x in roles)]   # [GLC] listed twice in eq 5
    if dropped:
        review.append(f"roles naming symbols not in the law dropped: {[x['symbol'] for x in dropped]}")
    if a.get('issue'):
        review.append(f"possible misprint: {a['issue']}")
    return {'description': a['description'], 'description_basis': d_basis,
            'quote': a['quote'] if d_basis == 'stated' else None,
            'assumptions': assumptions, 'roles': roles, 'issue': a.get('issue'),
            'status': 'pending' if review else 'accepted', 'review': review, 'llm': a}


def main(paper_dir: str, refresh: bool):
    result = annotate(paper_dir, refresh)
    print(llm.usage_line())
    if llm.DRY_RUN:
        return
    json.dump(result, open(f'{paper_dir}/laws.json', 'w'), indent=1, ensure_ascii=False)
    L = result['laws'].values()
    n = lambda f: sum(1 for v in L if f(v))
    print(f"{len(L)} laws: descriptions {n(lambda v: v.get('description_basis') == 'stated')} stated, "
          f"{n(lambda v: v.get('description_basis') == 'inferred')} inferred | "
          f"{sum(len(v.get('assumptions', [])) for v in L)} assumptions "
          f"({sum(a['severity'] == 'blocking' for v in L for a in v.get('assumptions', []))} blocking) | "
          f"{n(lambda v: bool(triage.must_check(v)))} to check, {n(lambda v: bool(v.get('review')) and not triage.must_check(v))} for information")
    print(f'-> {paper_dir}/laws.json')


if __name__ == '__main__':
    llm.DRY_RUN = '--dry-run' in sys.argv
    main(sys.argv[1], '--refresh' in sys.argv)
