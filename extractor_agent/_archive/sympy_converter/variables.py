"""variables.py -- define every symbol the equations use: meaning, unit, role.

PIPELINE
    equations (paddle.json)  -> symbols + which law computes each   (structure_check, code)
    blocks (paddle_blocks.json) -> candidate snippets per symbol    (candidates.py, code)
    LLM: per symbol, PICK the candidate that defines it, or none     (the only judgement step)
    gates                                                            (code)
    -> variables.json + report

THE LLM PICKS, IT DOES NOT WRITE
    It sees a closed list of symbols and, per symbol, numbered snippets from the paper.
    It answers with a snippet number (or null), a short meaning, the unit copied from that
    snippet, and a role. The quote IS the snippet, so it cannot paraphrase or recall a
    textbook definition: an answer that is not one of the listed snippets is rejected.

GATES (code, all paper-independent)
    G1 pick is one of this symbol's candidates, or null
    G2 the picked snippet agrees with the PDF text layer (or the PDF has none: unverifiable)
    G3 unit string appears in the picked snippet and parses (count units: cell, chain, mRNA, gene)
    G4 units printed for the symbol elsewhere in the paper do not conflict with it
    G5 role agrees with structure (computed by a law -> not a constant) and with tables
       (a numeric value in a table row -> constant)
    G6 the match between symbol and snippet was exact (not split-name / OCR-confusable)
    G7 a definition was found at all ('not defined in this paper' is allowed, but flagged)
    Every failed gate becomes a flag for a human. Nothing is silently fixed.

KG MAPPING (claude/KG_structure_decisions.md)
    role constant                       -> Parameter   (fixed=true only for universal constants)
    role state_or_computed / operating_input -> Variable

Usage (from extractor_agent/):
    python -m stages.variables papers/kontoravdi_2010 --dry-run      # no LLM: candidates, prompts, code-only checks
    python -m stages.variables papers/kontoravdi_2010                # needs GROQ_API_KEY
"""
from __future__ import annotations
import argparse, json, os, re, time
from dataclasses import asdict

import sympy as sp

from core.units import parse_unit, dims_equal
from stages.latex_to_sympy import parse
from stages.structure_check import assign, T
from stages.candidates import (build_snippets, textlayer_check, context_sentences,
                               find_candidates, key)

PROMPT_VERSION = 'var.v2'
BATCH = 8

UNIVERSAL_CONSTANTS = ('gas constant', 'faraday', 'boltzmann', 'avogadro', 'planck',
                       'gravitational acceleration', 'speed of light')

ROLE_HELP = {
    'constant': 'a fixed number characterising the system or a physical law: kinetic constant, '
                'yield, affinity constant, rate constant, physical constant',
    'operating_input': 'a quantity the experimenter sets or supplies: feed or outlet flow rate, '
                       'inlet or feed concentration, temperature setpoint',
    'state_or_computed': 'a quantity that changes during the run: a concentration, a volume, '
                         'a cell density, a rate or factor computed from other quantities',
    'unclear': 'the snippets do not let you decide',
}

SCHEMA = {
    'type': 'object',
    'properties': {
        'symbols': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'symbol': {'type': 'string',
                               'description': 'The symbol name exactly as given in the list.'},
                    'candidate': {'type': ['integer', 'null'],
                                  'description': 'Number of the snippet that DEFINES this symbol, '
                                                 'or null if none of its snippets does.'},
                    'meaning': {'type': ['string', 'null'],
                                'description': 'What the quantity is, in the words of the chosen '
                                               'snippet, at most 12 words. Null if candidate is null.'},
                    'unit_string': {'type': ['string', 'null'],
                                    'description': "The unit exactly as printed in the chosen snippet, "
                                                   "e.g. 'mM' or 'h^{-1}'. '-' only if the snippet says "
                                                   "dimensionless. Null if the snippet prints no unit."},
                    'role': {'type': 'string', 'enum': list(ROLE_HELP)},
                },
                'required': ['symbol', 'candidate', 'meaning', 'unit_string', 'role'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['symbols'],
    'additionalProperties': False,
}

SYSTEM = f"""\
You read snippets from a scientific paper and decide which snippet DEFINES each symbol \
of a mechanistic process model.

For every symbol you are given numbered snippets. Choose among THOSE snippets only.

Rules:
1. A snippet defines a symbol if it says what the quantity is: a nomenclature line, a \
"where X is ..." clause, a table row naming the parameter, or a sentence that describes \
the quantity an equation computes ("The specific growth rate is determined by ..." \
before "mu = ..."). A snippet that only uses or mentions the symbol does not define it.
   For a symbol that an equation COMPUTES, the sentence just before that equation (marked \
"text before the equation") counts as its definition when it says what the equation \
describes. Example: "The mass balances on glucose and glutamine around the bioreactor are:" \
before d([GLN])/dt = ... defines [GLN]: pick it, meaning "glutamine (mass balance variable)".
   The same letter can mean different things in different parts of a paper (V can be a volume \
in the model and a covariance matrix in the statistics section). Pick the snippet that fits how \
the symbol is used in the equation shown.
2. If no snippet defines the symbol, answer candidate null, meaning null, unit_string null. \
This is a correct and expected answer. Do NOT use your own knowledge of the model to \
fill the gap.
3. meaning: the snippet's own words, shortened. Do not add information that is not in it.
4. unit_string: copy the unit characters exactly as they appear in the chosen snippet. \
Null if that snippet prints no unit. Never infer a unit.
5. role, one of exactly these values:
{chr(10).join(f'   {k}: {v}' for k, v in ROLE_HELP.items())}
   The line "computed by eq N" tells you the model computes this symbol.
6. Answer every symbol in the list, once, using the symbol name exactly as given.

Return JSON only."""


def model_symbols(formulas):
    laws, failed = {}, []
    for i, f in enumerate(formulas):
        try:
            laws[i], _ = parse(f['latex'])
        except Exception as ex:
            failed.append((f.get('eq', i + 1), str(ex)[:80]))
    asg = assign(laws)
    syms = {}
    for i, e in laws.items():
        for s in sorted(e.free_symbols - {T}, key=str):
            d = syms.setdefault(str(s), {'computed_by': [], 'used_by': []})
            (d['computed_by'] if asg[i][0] == s else d['used_by']).append(i)
    return syms, failed


_UNIT_IN_PARENS = re.compile(r'\(([^()]*(?:\([^()]*\))?[^()]*)\)')

def printed_units(text: str):
    """Parenthesised groups in a snippet that parse as units (for G4)."""
    out = []
    for g in _UNIT_IN_PARENS.findall(text):
        g2 = g.replace('$', '').strip()
        if not g2 or re.fullmatch(r'[\d\s.,;]+', g2) or len(g2) > 25:
            continue
        if '_' in g2 or '\\' in g2 or re.search(r'\b(and|or|respectively|et al)\b', g2):
            continue        # symbol notation like (K_{lysis}) or a list: pint would read K as kelvin, N as newton
        r = parse_unit(g2)
        if r.ok and (r.dims or g2.lower() in ('dimensionless', '-')):
            out.append((g2, r.dims))
    return out


def _norm_unit(u: str) -> str:
    return re.sub(r'[\s${}\\^]', '', u or '')


def prompt_for(batch, symbols, cands, snip, formulas):
    lines = []
    for name in batch:
        info = symbols[name]
        eqn = lambda idx: formulas[idx].get('eq') or idx + 1
        how = (f"computed by eq {', '.join(str(eqn(i)) for i in info['computed_by'])}"
               if info['computed_by'] else 'not computed by any equation')
        uses = ', '.join(str(eqn(i)) for i in info['used_by'])
        lines.append(f'SYMBOL {name}  ({how}{"; used in eq " + uses if uses else ""})')
        shown = info['computed_by'] or info['used_by'][:1]
        for i in shown:
            tex = formulas[i]['latex'].strip().strip('$').strip()
            lines.append(f'  equation {eqn(i)}: {tex}')
        for sid, lv in cands[name]:
            s = snip[sid]
            tag = 'text before the equation' if lv == 'context' else s.kind.replace('_', ' ')
            lines.append(f'  [{sid}] (p.{s.page + 1}, {tag}) {s.text}')
        if not cands[name]:
            lines.append('  (no snippets found)')
        lines.append('')
    return 'Symbols and their candidate snippets:\n\n' + '\n'.join(lines)


def gate(name, ans, info, cand_list, snip, has_table_value, units_seen):
    flags = []
    ids = {sid: lv for sid, lv in cand_list}
    c = ans.get('candidate')
    rec = {'symbol': name, 'role': ans.get('role'), 'meaning': ans.get('meaning'),
           'unit_string': ans.get('unit_string'), 'source': None, 'dims': None, 'flags': flags}
    # G1
    if c is not None and c not in ids:
        flags.append(f'G1 picked snippet {c}, which is not one of this symbol\'s candidates: answer discarded')
        c = None; rec.update(meaning=None, unit_string=None)
    # G7
    if c is None:
        flags.append('G7 not defined in this paper (no snippet defines it)')
    else:
        s = snip[c]
        rec['source'] = {'snippet': c, 'page': s.page + 1, 'kind': s.kind, 'quote': s.text,
                         'match': ids[c], 'textlayer': s.textlayer}
        # G2
        if s.textlayer == 'disagrees':
            flags.append(f'G2 reader and PDF text layer disagree on this snippet: {"; ".join(s.textlayer_diff)}')
        elif s.textlayer == 'unverifiable':
            flags.append('G2 no PDF text layer: snippet not independently verified')
        # G6
        if ids[c] == 'partial':
            flags.append('G6 paper names a longer symbol than the equation (e.g. KI_lac vs I_lac): '
                         'the equation reader may have split one name into two')
        elif ids[c] == 'confusable':
            flags.append('G6 symbol matched only after OCR-confusable letters (I/l/1, O/0)')
        # G3
        u = rec['unit_string']
        if u is not None:
            said_dimless = u.strip() == '-' and 'dimensionless' in s.text.lower()
            if not said_dimless and _norm_unit(u) not in _norm_unit(s.text):
                flags.append(f'G3 unit "{u}" does not appear in the chosen snippet')
            r = parse_unit(u)
            if r.ok:
                rec['dims'] = r.dims
            else:
                flags.append(f'G3 unit "{u}" does not parse: {r.error}')
    # G4
    distinct = []
    for u, d in units_seen:
        if not any(dims_equal(d, d2) for _, d2 in distinct):
            distinct.append((u, d))
    if len(distinct) > 1:
        flags.append('G4 the paper prints conflicting units for this symbol: '
                     + ' vs '.join(f'"{u}"' for u, _ in distinct))
    # G5
    role = rec['role']
    if info['computed_by'] and role in ('constant', 'operating_input'):
        flags.append(f'G5 an equation computes this symbol, but the role given is "{role}"')
    if not info['computed_by'] and role == 'state_or_computed':
        flags.append('G5 role "state_or_computed", but no equation computes this symbol '
                     '(missing equation, or a misread symbol?)')
    if has_table_value and role not in ('constant', None):
        flags.append(f'G5 a table gives this symbol a numeric value, but the role given is "{role}"')
    if role == 'unclear' and not info['computed_by']:
        flags.append('G5 role unclear')
    if c is not None and ids.get(c) == 'context':
        rec['implied'] = True       # definition read from the sentence before the equation: worth a human look
    # KG node. Structure decides for computed symbols: they are Variables whatever the LLM said
    # (a disagreement is already flagged by G5 above).
    rec['node'] = 'Variable' if info['computed_by'] else ('Parameter' if role == 'constant' else 'Variable')
    rec['fixed'] = bool(rec['meaning'] and any(k in rec['meaning'].lower() for k in UNIVERSAL_CONSTANTS))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paper_dir')
    ap.add_argument('--dry-run', action='store_true', help='no LLM: write prompts and code-only checks')
    ap.add_argument('--answers', help='JSON file of LLM answers to gate (re-gate without new calls)')
    ap.add_argument('--resume', action='store_true',
                    help='keep answers already in variables_llm_answers.json; call the LLM only for missing symbols')
    ap.add_argument('--model', default=os.environ.get('LLM_MODEL', 'openai/gpt-oss-120b'))
    ap.add_argument('--pause', type=float, default=8.0, help='seconds between calls (rate limit)')
    a = ap.parse_args()
    P = a.paper_dir

    formulas = json.load(open(f'{P}/paddle.json'))['formulas']
    B = json.load(open(f'{P}/paddle_blocks.json'))
    blocks = B['blocks']
    symbols, failed = model_symbols(formulas)
    snips = build_snippets(blocks)
    textlayer_check(snips, blocks, f'{P}/paper.pdf', B.get('dpi', 150))
    snip = {s.id: s for s in snips}
    ctx = context_sentences(blocks, snips, [f['bbox'] for f in formulas])
    cands = find_candidates(symbols, snips, ctx)
    names = sorted(symbols, key=lambda n: (not symbols[n]['computed_by'], n.lower()))

    has_value = {n: any(snip[sid].kind == 'table_row' and re.search(r'\|[^|]*\d', snip[sid].text)
                        for sid, lv in cands[n] if lv != 'context') for n in names}
    units_seen = {n: [u for sid, lv in cands[n] if lv != 'context' for u in printed_units(snip[sid].text)]
                  for n in names}

    batches = [names[i:i + BATCH] for i in range(0, len(names), BATCH)]
    prompts = [prompt_for(b, symbols, cands, snip, formulas) for b in batches]

    print(f'{len(names)} symbols, {len(snips)} snippets, {len(batches)} LLM calls of up to {BATCH} symbols')
    if failed:
        print('equations that did not parse (their symbols are missing):', failed)
    if not symbols:
        raise SystemExit('No equation parsed, so there are no symbols to define. Nothing written. '
                         'Check that stages/latex_to_sympy.py is the current version.')

    answers = {}
    if a.answers:
        for item in json.load(open(a.answers)):
            answers[item['symbol']] = item
    elif a.dry_run:
        with open(f'{P}/variables_prompts.txt', 'w') as f:
            f.write(SYSTEM + '\n\n' + ('\n' + '=' * 80 + '\n').join(prompts))
        print(f'dry run: prompts written to {P}/variables_prompts.txt')
    else:
        from core.harness import Harness, log_call
        key_ = os.environ.get('GROQ_API_KEY')
        if not key_:
            raise SystemExit('GROQ_API_KEY is not set in this shell')
        h = Harness('https://api.groq.com/openai/v1', a.model, api_key=key_)
        os.makedirs('logs', exist_ok=True)
        ans_path = f'{P}/variables_llm_answers.json'
        raw = []
        if a.resume and os.path.exists(ans_path):
            # only answers made with THIS prompt version: mixing versions makes the run uninterpretable
            raw = [x for x in json.load(open(ans_path))
                   if x['symbol'] in symbols and x.get('prompt_version') == PROMPT_VERSION]
            answers.update({x['symbol']: x for x in raw})
            print(f'resume: kept {len(raw)} earlier answers')
        todo = [i for i, b in enumerate(batches) if any(n not in answers for n in b)]
        for k, bi in enumerate(todo):
            b, pr = batches[bi], prompts[bi]
            r = h.call(system=SYSTEM, user=pr, schema=SCHEMA, schema_name='symbol_definitions',
                       prompt_version=PROMPT_VERSION)
            log_call('logs/runs.jsonl', os.path.basename(P.rstrip('/')), 3, r, pr)
            print(f'  call {k + 1}/{len(todo)} (batch {bi + 1}): ok={r.ok} tier={r.tier}'
                  + ('' if r.ok else f'  last error: {r.errors[-1][:120] if r.errors else "?"}'))
            if r.ok:
                for item in r.data['symbols']:
                    if item['symbol'] in b and item['symbol'] not in answers:
                        item['prompt_version'] = PROMPT_VERSION
                        answers[item['symbol']] = item
                        raw.append(item)
            # save after every call, so a failure late in the run loses nothing
            json.dump(raw, open(ans_path, 'w'), indent=1, ensure_ascii=False)
            if k < len(todo) - 1:
                time.sleep(a.pause)
        missing = [n for n in names if n not in answers]
        if missing:
            print(f'{len(missing)} symbols still have no answer; rerun with --resume to fill them in')

    if a.dry_run:
        print('\nCODE-ONLY CHECKS (no LLM):')
        print(f"{'symbol':14} {'computed by':12} {'#cand':5} {'best match':10} {'table value':11} printed units")
        for n in names:
            lv = [l for _, l in cands[n]]
            best = next((x for x in ['exact', 'notation', 'confusable', 'partial', 'context'] if x in lv), '-')
            cb = ','.join(str(formulas[i].get('eq')) for i in symbols[n]['computed_by']) or '-'
            us = sorted({u for u, _ in units_seen[n]})
            print(f"{n:14} {cb:12} {len(cands[n]):<5} {best:10} {str(has_value[n]):11} {us}")
        print('\nFlags that need no LLM:')
        for n in names:
            dims = []
            for u, d in units_seen[n]:
                if not any(dims_equal(d, d2) for _, d2 in dims):
                    dims.append((u, d))
            if len(dims) > 1:
                print(f"  {n:14} G4 conflicting units printed: {' vs '.join(u for u, _ in dims)}")
            if not cands[n] or all(l == 'context' for _, l in cands[n]):
                print(f"  {n:14} symbol named nowhere in the text (only context sentences): definition must be implied or missing")
            for sid, l in cands[n]:
                if l in ('partial', 'confusable'):
                    print(f"  {n:14} G6 {l} match: {snip[sid].text[:70]}")
        for s_ in snips:
            if s_.textlayer == 'disagrees':
                print(f"  text layer: p{s_.page + 1} {s_.text[:60]!r}: {s_.textlayer_diff}")
        return

    records = []
    for n in names:
        ans = answers.get(n, {'candidate': None, 'meaning': None, 'unit_string': None, 'role': None})
        rec = gate(n, ans, symbols[n], cands[n], snip, has_value[n], units_seen[n])
        if n not in answers:
            rec['flags'].insert(0, 'no LLM answer for this symbol' if (answers or a.answers) else 'dry run: no LLM answer')
        rec['computed_by_eq'] = [formulas[i].get('eq') for i in symbols[n]['computed_by']]
        rec['used_in_eq'] = [formulas[i].get('eq') for i in symbols[n]['used_by']]
        rec['candidates'] = [{'snippet': sid, 'match': lv, 'kind': snip[sid].kind, 'page': snip[sid].page + 1,
                              'textlayer': snip[sid].textlayer, 'text': snip[sid].text} for sid, lv in cands[n]]
        records.append(rec)

    out = {'paper': os.path.basename(P.rstrip('/')), 'prompt_version': PROMPT_VERSION,
           'llm_used': bool(answers) and not a.answers, 'symbols': records}
    json.dump(out, open(f'{P}/variables.json', 'w'), indent=1, ensure_ascii=False)

    # report
    print(f"\n{'symbol':14} {'node':9} {'role':17} {'unit':16} flags")
    for r in records:
        print(f"{r['symbol']:14} {r['node']:9} {str(r['role']):17} {str(r['unit_string']):16} "
              f"{len(r['flags'])}")
    flagged = [r for r in records if r['flags']]
    print(f'\n{len(flagged)}/{len(records)} symbols flagged for review:')
    for r in flagged:
        for f in r['flags']:
            print(f"  {r['symbol']:14} {f}")
    print(f'\nwritten -> {P}/variables.json')


if __name__ == '__main__':
    main()
