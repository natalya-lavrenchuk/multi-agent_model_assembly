"""sup.py -- propose paper.json: which printed equations are the model, and the Sup's metadata.

    python -m extractor.sup papers/birol_2002             # writes paper.json with status 'proposed'
    python -m extractor.sup papers/birol_2002 --approve   # after you have checked it

Code first, the LLM for what code cannot decide, a human approves.
  1. stated   sentences in which the paper lists a model's equation numbers ('Eqs. (1), (3), (9)-(18)')
              are found and expanded by code. If the LLM picks one, it is the model's boundary.
  2. needed   starting from the ODEs, every equation that computes something an included equation uses
              is pulled in, until nothing new is needed (all laws if the paper has no ODE).
  3. LLM      one call: the equations with code's tags, the paper's sentences. It settles what code
              cannot (two equations computing the same thing: the same law restated, or rival laws that
              make separate models; equations nothing needs) and writes the model assumptions (quoted; blocking only for process conditions)
              and complete model or not. The Sup's description is written later, by supmeta.py, from
              the extracted laws and symbols.
  4. checks   code never lets a needed equation drop out silently: it is added back and reported.
              Equations written with '~' (\\approx), symbols used but computed outside the model and
              every quote that is not in the paper are reported for the reviewer.

--approve first applies the reviewer's boundary edits from corrections.json (review.html).
symbols.py, laws.py and graph.py refuse a paper.json that is still 'proposed'. If an approved paper.json
already exists, the proposal goes to paper_proposed.json and the two are compared.
"""
from __future__ import annotations
import json
import os
import re
import sys

from extractor import llm
from extractor.classify import MODEL, squash
from extractor.graph import _computes
from extractor.paper import (PAPER_NOTE, TEXT_LABELS, clean, load_equations, load_snippets, prose_names,
                             read_equations, with_paper)

PROMPT = """You read a published paper on a mechanistic process model and decide which of its printed
equations make up the model. Code has already analysed the equations; each one carries a tag:
  NEEDED      it computes something another model equation uses (or it is a balance, an ODE)
  CONFLICT    it computes the same quantity as another equation: only one of them belongs to a model
  NOT USED BY OTHERS  no other equation uses what it computes: the model's output (keep it: e.g. the total
              flow or total biomass the paper predicts and fits to data), or a metric, or not part of the model
  APPROX      printed with an approximately-equal sign, and could not be read
  UPDATE      changes a quantity in place (mu_X = 0.5 mu_X (1 - tanh(...))): a factor applied on top of the law
              that computes it. Keep it with that law. If the paper also prints one equation that multiplies
              all the factors together, keep either the separate updates or the combined equation, not both
Sentences in which the paper itself lists a model's equations are given separately, already expanded.
Follow them and the NEEDED tags unless the paper clearly says otherwise (an ODE can also be background
from earlier work). Decide CONFLICT, NOT USED BY OTHERS, APPROX:
- keep an equation that gives a model output the paper predicts (e.g. total biomass), or that the paper
  says is part of the model
- leave out objective functions, error metrics and statistics, parameter-estimation set-ups, controller
  equations, soft sensors that reconstruct a state from measurements, and an earlier version of an
  equation that the paper replaces with a new one
- keep a feed or operating policy the model is run with (an exponential feed F(t), a feed rate profile): it
  defines one of the model's inputs. A controller that computes an input from measurements stays out
- leave out initial conditions written in general form (x(t0) = x0) and general model forms
  (dx/dt = f(x, u, theta)); keep a balance or rate law written out for this process
- if the paper itself lists its model equations and that list leaves out the equation for a quantity
  the model uses (pH held by a controller), follow the paper: that quantity is an input

Two equations computing the same quantity (CONFLICT) are one of two things:
- the same law restated: an integrated, closed-form or simplified form of another equation, or a
  derivation step. Keep the differential or rate form (the ODE, the rate law, the balance), even if the
  paper evaluates the integrated solution: an integrated solution is only valid under the paper's own
  operating conditions (constant pressure, constant feed), while the rate form can be reused in another
  process. Exclude the integrated form with the reason "restates eq N". Keep an algebraic equation only if
  no differential form of it is printed.
- rival laws: different hypotheses for the same quantity (Monod vs Contois growth, two kinetic models).
  Then return one model per alternative; they may share all their other equations.
An approximate, simplified or analytical solution of the same model is NOT a rival: it restates the model
(exclude it with the reason "approximates eq N"). Background equations from earlier work (a classical law
the paper starts from), derivation steps and analysis of the results are not part of the model either.
Return more than one model only for such alternatives, each solved on its own (also an initial and an
improved model).
If the paper fits many combinations (model structures x kinetic laws x inhibition terms) but reports
parameter values for only some of them (the best fits, in a table), return one model per combination it
reports values for, and nothing for the others. Fitting the same equations to several data sets or
conditions (two cell lines, two temperatures, two substrates) is ONE model: its values per data set are
recorded with the symbols, not as separate models. Only a different set of equations is another model
(a different kinetic law per data set: then put the data set in each name). A
candidate law the paper offers (one of its kinetic options) that none of the returned models uses goes
into free_laws, so it is not lost.
A model has balances or states that can be simulated. If the paper instead fits single laws on their own,
with no model around them (it compares growth-rate expressions against data: Monod, Haldane, Contois,
each fitted separately), put each such law in free_laws with its name, not in models. A paper can have
both: models, and laws it fits on their own. models may be empty.
If one of the given sentences lists the model's equations, copy that sentence into boundary_quote exactly
as given; never write a sentence of your own there, use null instead.

For each model:
- name: a few words that tell the models apart ("Monod growth", "Contois growth")
- assumptions: only those the paper states for the whole model, with the paper's words in quote. Do not
  add assumptions the paper does not state; an empty list is fine. Do not list what only names the model or
  its mechanism ("Monod-type kinetics", "complete pore blocking", "Langmuir-Hinshelwood mechanism"): that is
  what the model is, not an assumption about the process. Do not list what the model's
  laws themselves say (a mechanism, a kinetic form, which effects are included). process_condition: true only
  if it is about how the user's process is built, fed or run, something the user can check without knowing the
  model: feed composition (sterile feed, no acetate in the feed), operating mode (constant pressure, constant
  flow, batch, steady state), mixing, temperature or pressure regime (isothermal, isobaric), the equipment or
  material used, the phase of the process. false for modelling choices: quasi-equilibrium of fast steps, a
  term or heat loss neglected, lumped kinetics, a property value or correlation chosen.
- complete_model: true only if the printed equations form a closed model; false if some of the model's
  equations are only in supplementary material, an appendix not given here, or another paper.

Every equation goes into at least one model, into free_laws, or into excluded, with a short reason. Write
equation ids exactly as listed ("7", "C-11", "6.2", "p5-3"), without brackets. Ids like "6.1", "6.2" are the
separate equations of one printed block (eq 6 holds several); "p5-3" is an equation the paper prints
without a number (page 5, the third such one).""" + PAPER_NOTE

SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['models', 'excluded', 'free_laws'],
    'properties': {
        'models': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['name', 'equations', 'boundary_quote', 'assumptions', 'complete_model'],
            'properties': {
                'name': {'type': 'string'},
                'equations': {'type': 'array', 'items': {'type': 'string'}},
                'boundary_quote': {'type': ['string', 'null']},
                'assumptions': {'type': 'array', 'items': {
                    'type': 'object', 'additionalProperties': False, 'required': ['text', 'quote', 'process_condition'],
                    'properties': {'text': {'type': 'string'}, 'quote': {'type': ['string', 'null']},
                                   'process_condition': {'type': 'boolean'}}}},
                'complete_model': {'type': 'boolean'}}}},
        'excluded': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False, 'required': ['eq', 'reason'],
            'properties': {'eq': {'type': 'string'}, 'reason': {'type': 'string'}}}},
        'free_laws': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False, 'required': ['eq', 'name'],
            'properties': {'eq': {'type': 'string'}, 'name': {'type': 'string'}}}}}}

BUDGET = 9000            # answer tokens for the boundary call (reasoning + every equation placed)
BUDGET_HIGH = 32000      # at high effort the reasoning alone can pass 20k on a 50-equation paper (Seidel)
STATEMENT = re.compile(r'\bEqs?\b|\bequations?\b|\bassum|\bconsists? of\b|\bsolved\b|\bmodel (is|includes|comprises)\b', re.I)
MAX_STATEMENTS = 35
EQ_LIST = re.compile(r'\b(?:Eqs?|Equations?)\.?\s*\(?\s*\d', re.I)
MODEL_WORD = re.compile(r'\bmodel\b|\bconsists? of\b|\bsolved\b|\bsystem of\b', re.I)
key = lambda n: str(n).zfill(4)


def propose(paper_dir: str) -> dict:
    blocks = json.load(open(f'{paper_dir}/paddle_blocks.json'))['blocks']
    formulas, _ = load_equations(paper_dir)
    snippets = load_snippets(paper_dir, tables=False)
    unnumbered = [f for f in formulas if f['eq'] is None]       # OCR fragments ('67', 't^*') or inline math
    formulas = [f for f in formulas if f['eq'] is not None]
    reads = read_equations(formulas, prose_names(paper_dir))
    ids = [str(f['eq']) for f in formulas]
    st = structure(ids, formulas, reads)
    stated = stated_lists(snippets, set(ids))

    msg = with_paper(paper_dir, _message(blocks, formulas, ids, st, stated, snippets, load_snippets(paper_dir)))
    schema = json.loads(json.dumps(SCHEMA))                # equation numbers limited to the paper's: no '(7)'
    schema['properties']['models']['items']['properties']['equations']['items'] = {'type': 'string', 'enum': ids}
    schema['properties']['excluded']['items']['properties']['eq'] = {'type': 'string', 'enum': ids}
    schema['properties']['free_laws']['items']['properties']['eq'] = {'type': 'string', 'enum': ids}
    try:                                                  # one call over the whole paper: it gets a larger answer budget
        answer = llm.cached_ask(paper_dir, PROMPT, msg, schema, 'sup', 'medium', label='boundary',
                                refresh='--refresh' in sys.argv, max_tokens=max(llm.MAX_TOKENS, BUDGET_HIGH if llm.EFFORT == 'high' else BUDGET))
    except llm.JSONFailed:                                # the reasoning used up the budget: think less, answer in full
        print('  boundary: answer did not fit, asking again with less reasoning (effort low)')
        answer = llm.cached_ask(paper_dir, PROMPT, msg, schema, 'sup', 'low', label='boundary (effort low)',
                                refresh='--refresh' in sys.argv, max_tokens=max(llm.MAX_TOKENS, BUDGET), keep_effort=True)
    if answer is None or llm.DRY_RUN:                     # dry run: report the size, write nothing
        raise SystemExit(llm.usage_line())
    left_out = {e['eq']: e['reason'] for e in answer['excluded'] if e['eq'] in ids}
    models = []
    for m in answer['models']:                            # an equation in another model was a choice, not an omission
        other = {n for o in answer['models'] if o is not m for n in o['equations']} - set(m['equations'])
        models.append(_model(m, snippets, ids, st, stated, left_out, other))
    for m in models:
        _attach_to_laws(m, formulas, ids, blocks)
    free = [x for x in answer.get('free_laws', []) if x['eq'] in ids]
    in_models = {n for m in models for n in m['equations']} | {x['eq'] for x in free}
    excluded = [e for e in answer['excluded'] if e['eq'] in ids and e['eq'] not in in_models]
    excluded += [{'eq': n, 'reason': 'not placed by the LLM'} for n in ids
                 if n not in in_models and n not in {e['eq'] for e in excluded}]
    checks = [c for m in models for c in m.pop("checks")] + _approx_checks(st, in_models) + \
             [f"unnumbered formula on p.{f['page'] + 1} not considered: {clean(f['latex'])[:60]}" for f in unnumbered]
    return {**_identity(paper_dir, blocks), 'status': 'proposed', 'proposed_by': MODEL,
            **(_flat(models[0]) if len(models) == 1 else {'models': models}),
            **({'free_laws': free} if free else {}),
            'excluded': sorted(excluded, key=lambda e: key(e['eq'])), 'checks': checks,
            'code_analysis': {'stated': [{'sentence': s, 'equations': e} for s, e in stated],
                              'tags': {n: st['tag'][n] for n in ids}}}


# ----------------------------------------------------------------------------- code: structure
def structure(ids, formulas, reads) -> dict:
    """Tags per equation. computes: what each law computes (graph.py's rule). Core = the ODEs plus,
    repeatedly, the one equation that computes a symbol the core uses; if two equations compute it,
    both are CONFLICT and neither is pulled in. Without ODEs (steady state, rate laws) every law is core."""
    read = dict(zip(ids, reads))
    laws = [n for n in ids if read[n]['form'] != 'unknown' and not read[n].get('notation')]
    comp = dict(zip(ids, _computes(reads)))
    approx = {n for n, f in zip(ids, formulas) if read[n]['form'] == 'unknown' and '\\approx' in f['latex']}
    producers, updates = {}, {}
    for n in laws:
        if read[n].get('updates'):                       # mu_X = 0.5 mu_X (...): stacks on mu_X, no rival to it
            updates.setdefault(read[n]['updates'], []).append(n)
        elif comp[n]:
            producers.setdefault(comp[n], []).append(n)
    core = {n for n in laws if read[n]['form'] in ('ODE', 'PDE')} or set(laws)
    conflict, why = set(), {}
    while True:
        used = {s for n in core for s in read[n]['rhs'] + read[n]['lhs']}
        new = {n for s in used for n in updates.get(s, []) if n not in core}
        for n in new:
            why[n] = f'updates {read[n]["updates"]}, used by the model'
        for s in used:
            p = producers.get(s, [])
            if len(p) > 1:
                conflict |= set(p)
            elif len(p) == 1 and p[0] not in core:
                new.add(p[0])
                why[p[0]] = f'computes {s}, used by the model'
        if not new:
            break
        core |= new
    superseded = {n: f'C-{n}' for n in laws if comp.get(n) and comp.get(f'C-{n}') == comp[n]}
    tag = {}
    for n in ids:
        if n in superseded:
            tag[n] = f'SUPERSEDED: the corrigendum corrects it as {superseded[n]}; use {superseded[n]}'
            continue
        if n in approx:
            tag[n] = 'APPROX'
        elif read[n]['form'] == 'unknown':
            tag[n] = 'not an equation'
        elif read[n].get('notation'):
            tag[n] = f'NOTATION: names the parts of the vector {", ".join(read[n]["lhs"])} (keep it with the model that uses the vector)'
        elif read[n].get('updates'):
            tag[n] = (f'UPDATE of {read[n]["updates"]} (applied in printed order on top of the law that computes it): '
                      + ('NEEDED: ' + why[n] if n in core else 'not used by the model laws'))
        elif n in conflict:
            tag[n] = f'CONFLICT: computes {comp[n]}, like eq ' + ', '.join(x for x in producers[comp[n]] if x != n)
        elif n in core:
            tag[n] = 'NEEDED: ' + why.get(n, 'an ODE (the model, or background from earlier work?)'
                                          if read[n]['form'] != 'algebraic' else 'a model law')
        else:
            tag[n] = (f'NOT USED BY OTHERS: computes {comp[n] or "?"}; the model output (keep) or not part of '
                      f'the model (leave out)')
    return {'tag': tag, 'core': core - conflict, 'conflict': conflict, 'approx': approx, 'superseded': superseded,
            'computes': comp, 'read': read, 'producers': producers, 'updates': updates}


def stated_lists(snippets, ids) -> list[tuple[str, list[str]]]:
    """Sentences that list a model's equations, with the numbers expanded: 'Eq. 2-3, 5-8, 18-20' ->
    2, 3, 5, 6, 7, 8, 18, 19, 20. Only numbers that are equations of this paper are kept."""
    out = []
    for s in snippets:
        t = s['text']
        m = EQ_LIST.search(t)
        if s['kind'] != 'sentence' or not m or not MODEL_WORD.search(t):
            continue
        nums = []
        for a, b, c in re.findall(r'(\d+)\s*\)?\s*[-–—]\s*\(?\s*(\d+)|(\d+)', t[m.start():]):
            if a and 0 < int(b) - int(a) < 60:
                nums += [str(k) for k in range(int(a), int(b) + 1)]
            elif c:
                nums.append(c)
        nums = [n for n in dict.fromkeys(nums) if n in ids]
        if len(nums) >= 3:                                # a list of a model's equations, not 'see Eq. 4'
            out.append((t, nums))
    return out


# ----------------------------------------------------------------------------- the LLM's answer, checked
def _model(m, snippets, ids, st, stated, left_out=None, elsewhere=frozenset()) -> dict:
    """The LLM's model, then code's safeguards: a stated list the LLM quotes is the boundary; a NEEDED
    equation the model leaves out is added back; anything changed or doubtful goes into checks.
    left_out: eq -> the LLM's reason for excluding it. Code never adds back an equation the LLM excluded
    with a reason (Nagy A1 belongs to the analytical method, not to the model); it says so in checks."""
    left_out = left_out or {}
    pre_checks = []
    norm = lambda n: re.sub(r'^\(\s*(.*?)\s*\)$', r'\1', str(n).strip())   # '(7)' -> '7'
    m['equations'] = [norm(n) for n in m['equations']]
    for i, n in enumerate(m['equations']):                # a corrected equation replaces the original (code)
        if n in st['superseded']:
            m['equations'][i] = st['superseded'][n]
            pre_checks.append(f"'{m['name']}': eq {n} replaced by its corrigendum version {st['superseded'][n]}")
    eqs = [n for n in dict.fromkeys(m['equations']) if n in ids]
    checks = pre_checks + [f"'{m['name']}': eq {n} does not exist" for n in m['equations'] if n not in ids]
    quote = m['boundary_quote']
    match = next(((s, e) for s, e in stated if quote and len(squash(quote)) >= 12 and squash(quote) in squash(s)), None)
    if match:                                             # a sentence that lists two models' equations together
        dup = {}                                          # ('the detailed Eqs. (4)-(9) and simplified Eqs. (11)-(16)')
        for n in match[1]:                                # is not one model's boundary: two of its equations would
            if st['computes'].get(n):                     # compute the same quantity
                dup.setdefault(st['computes'][n], []).append(n)
        clash = [v for v in dup.values() if len(v) > 1]
        if clash:
            checks.append(f"'{m['name']}': the quoted sentence lists eqs {match[1]}, which compute the same "
                              f"quantity twice ({', '.join('/'.join(v) for v in clash[:3])}): not one model's "
                              f"boundary, so the LLM's own list is kept")
            match, quote = None, None
    if match:                                             # a stated list that drops what the LLM's list needs is a
        def open_(lst):                                   # list of one part only (Owhondah: the 2R balances, not
            have = {st['computes'][n] for n in lst if st['computes'].get(n)}   # the kinetics they use)
            return {sym for n in lst if not st['read'][n].get('notation') for sym in st['read'][n]['rhs']
                    if sym not in have and st['producers'].get(sym)}
        lost = open_(match[1]) - open_(eqs)
        if lost and set(match[1]) < set(eqs) | set(match[1]):
            checks.append(f"'{m['name']}': the paper's sentence lists eqs {match[1]}, but that leaves "
                          f"{', '.join(sorted(lost))} without the equation computing it, which the LLM's list has: "
                          f"the LLM's list is kept")
            match = None
    if match:                                             # the paper's own list is the boundary
        if set(match[1]) != set(eqs):
            checks.append(f"'{m['name']}': the LLM listed {sorted(eqs, key=key)}, the paper's sentence lists "
                          f"{match[1]}; the paper's list is used")
        eqs = list(match[1])
    elif quote:
        checks.append(f"'{m['name']}': boundary quote is not one of the paper's sentences: {quote[:80]}")
    while True:                                           # never drop a needed equation silently: what the chosen
        have = {st['computes'][n] for n in eqs if st['computes'].get(n)}       # equations use and only one
        need = {}                                                               # equation computes is added back
        for n in eqs:
            for sym in ([] if st['read'][n].get('notation') else st['read'][n]['rhs']):   # a vector's parts: no use
                p = st['producers'].get(sym, [])
                if sym not in have and len(p) == 1 and p[0] not in eqs:
                    need[p[0]] = (sym, n)
        if not need:
            break
        for n, (sym, user) in need.items():
            if match:
                checks.append(f"'{m['name']}': eq {n} computes {sym}, used by eq {user}, but is not in the paper's list")
            elif n in elsewhere:                          # the LLM put it in another model, not this one
                checks.append(f"'{m['name']}': {sym} (used by eq {user}) is computed by eq {n}, which the LLM put in "
                              f"another model but not this one: an input here, or should eq {n} be in?")
            elif n in left_out:
                checks.append(f"'{m['name']}': {sym} (used by eq {user}) is computed only by eq {n}, which the LLM "
                              f"left out ({left_out[n][:70]}): an input of this model, or should eq {n} be in?")
            else:
                eqs.append(n)
                checks.append(f"'{m['name']}': eq {n} added back by code (computes {sym}, used by eq {user})")
        if match or all(n in left_out or n in elsewhere for n in need):
            break
        if any(n in left_out for n in need):          # only the ones not excluded were added: look again
            need = {n: v for n, v in need.items() if n not in left_out}
    by_symbol = {}
    for n in eqs:
        if st['computes'].get(n) and not st['read'][n].get('updates'):   # an update stacks on, it is no rival
            by_symbol.setdefault(st['computes'][n], []).append(n)
    checks += [f"'{m['name']}': eqs {', '.join(v)} all compute {s}: keep one" for s, v in by_symbol.items() if len(v) > 1]
    computed = {st['computes'][n] for n in eqs if st['computes'].get(n)}
    used = {s for n in eqs if not st['read'][n].get('notation') for s in st['read'][n]['rhs']}
    said = {c.split(': ', 1)[1].split(' (used by')[0] for c in checks if '(used by eq' in c}   # explained above   # already explained
    checks += [f"'{m['name']}': {s} is used, but the equation computing it (eq {', '.join(st['producers'][s])}) "
               f"is not in the model" for s in sorted(used - computed) if s in st['producers'] and s not in said]
    assumptions = []
    for a in m['assumptions']:
        q, page = _quoted(a['quote'], snippets)
        if not q:                                         # paraphrased: the paper's own sentence, if one clearly matches
            q, page = _assumption_sentence(a['text'] + ' ' + (a['quote'] or ''), snippets)
        if q:                                             # stated by the paper: kept; blocking only if a process condition
            pc = bool(a.get('process_condition'))         # (same rule as laws.py): a modelling choice is a warning
            assumptions.append({'text': a['text'], 'quote': q, 'page': page, 'process_condition': pc,
                                'severity': 'blocking' if pc else 'warning', 'reasoned': False})
        else:
            checks.append(f"'{m['name']}': assumption dropped, not found in the paper: {a['text'][:80]}")
    return {'name': m['name'], 'equations': sorted(eqs, key=key),
            'equations_quote': match[0] if match else None,
            'boundary_basis': 'stated' if match else 'inferred',
            'complete_model': m['complete_model'],
            'model_assumptions': assumptions, 'checks': list(dict.fromkeys(checks))}


def parse_equations(text: str) -> list[str]:
    """'4, C-5, 7-10, C-11–C-16' -> ['4', 'C-5', '7', '8', '9', '10', 'C-11', ..., 'C-16']."""
    out = []
    for part in [x.strip() for x in text.split(',') if x.strip()]:
        r = re.fullmatch(r'([A-Za-z]+-)?(\d+)\s*[-–]\s*(?:[A-Za-z]+-)?(\d+)', part)
        out += [f"{r.group(1) or ''}{i}" for i in range(int(r.group(2)), int(r.group(3)) + 1)] if r else [part]
    return out


def _boundary_corrections(paper_dir, cfg) -> list[str]:
    """Apply the reviewer's boundary edits from review.html (corrections.json) to paper.json before
    approving: per model 'equations' ('4, 7-10, C-11–C-16'; empty drops the model) and 'name';
    'new_models' adds Sups ([{name, equations, based_on}]: assumptions and 'complete model' are copied
    from the model it is based on). Returns what changed. New models are removed from corrections.json
    once applied, so approving again does not add them twice."""
    from extractor.paper import load_equations, models
    path = f'{paper_dir}/corrections.json'
    corr_all = json.load(open(path)) if os.path.exists(path) else {}
    corr = corr_all.get('sup', {})
    known = {str(f['eq']) for f in load_equations(paper_dir)[0] if f['eq'] is not None}
    changes = []
    if not cfg.get('models') and corr_all.get('new_models'):  # one model becomes the first of several
        first = {k: cfg.pop(k) for k in list(cfg) if k in ('equations', 'equations_quote', 'boundary_basis',
                                                            'complete_model', 'model_assumptions', 'model_name')}
        first['name'] = first.pop('model_name', None) or 'model 1'
        cfg['models'] = [first]
    targets = cfg['models'] if cfg.get('models') else [cfg]
    by_id = {}
    for m, mm in zip(targets, models(cfg)):
        by_id[mm['id']] = m
        c = corr.get(mm['id'], {})
        if c.get('name') and c['name'] != m.get('name', m.get('model_name')):
            changes.append(f"{mm['id']}: renamed to '{c['name']}'")
            m['name' if cfg.get('models') else 'model_name'] = c['name']
        eqs = c.get('equations')
        if eqs is None:
            continue
        if not eqs.strip():                               # emptied in the page: this model is dropped
            m['_drop'] = True
            changes.append(f"{mm['id']}: dropped")
            continue
        new = parse_equations(eqs)
        if [str(x) for x in m.get('equations') or []] != new:
            changes.append(f"{mm['id']}: equations {m.get('equations')} -> {new}")
            m['equations'], m['boundary_basis'] = new, 'human'
    for nm in corr_all.get('new_models', []):
        base = by_id.get(nm.get('based_on'), {})
        eqs = parse_equations(nm.get('equations', ''))
        if not eqs:
            continue
        cfg['models'].append({'name': nm.get('name') or f"model {len(cfg['models']) + 1}", 'equations': eqs,
                              'boundary_basis': 'human', 'complete_model': base.get('complete_model'),
                              'model_assumptions': json.loads(json.dumps(base.get('model_assumptions', [])))})
        changes.append(f"new model '{cfg['models'][-1]['name']}': {eqs}")
    for m in targets if not cfg.get('models') else cfg['models']:
        unknown = [e for e in m.get('equations') or [] if e not in known]
        if unknown and not m.get('_drop'):
            changes.append(f"WARNING {m.get('name', m.get('model_name'))}: no equation {unknown} in the paper")
    if cfg.get('models'):
        cfg['models'] = [m for m in cfg['models'] if not m.get('_drop')]
        if len(cfg['models']) == 1:                       # one model left: the plain one-Sup layout
            only = cfg.pop('models')[0]
            only['model_name'] = only.pop('name', None)
            cfg.update(only)
            changes.append(f"one model left: Sup id {cfg['sup']}")
    if corr_all.pop('new_models', None) is not None:      # applied: do not add them again next time
        json.dump(corr_all, open(path, 'w'), indent=1, ensure_ascii=False)
    return changes


def _approx_checks(st, chosen=None) -> list[str]:
    out = [f'eq {n} is printed with ~ (\\approx) and could not be read as a law; if it is part of the model, '
           f'its LaTeX needs fixing by hand' for n in sorted(st['approx'], key=key)]
    for v, ns in st.get('updates', {}).items():           # several updates of one rate: a factor applied twice?
        ns = [n for n in ns if chosen is None or n in chosen]
        if len(ns) > 1:
            out.append(f'{v} is updated in place by eqs {", ".join(ns)}, in that order: check that no factor is '
                       f'applied twice (e.g. one equation per factor and another that multiplies them all)')
    return out


def _attach_to_laws(m, formulas, ids, blocks):
    """An assumption printed in the sentence that introduces one equation ('The rate of pore blockage is
    assumed to be proportional to ... :' then eq 7) belongs to that law, not to the whole model: it gets
    'eq'. graph.py then writes it on the law, so reusing another law of the model is not blocked by it."""
    from extractor.laws import _position, _text_near
    intro = {}
    for n, f in zip(ids, formulas):
        k = _text_near(blocks, _position(blocks, f), -1)
        if k is not None and n in m['equations']:
            intro.setdefault(k, n)                         # the first equation after that text
    for a in m['model_assumptions']:
        q = squash(a['quote'])
        hit = next((n for k, n in intro.items() if q and q in squash(blocks[k]['content'])), None)
        if hit:
            a['eq'] = hit


def _assumption_sentence(text, snippets):
    """The paper's sentence that states this assumption, when the LLM paraphrased it: a sentence with
    'assum' that shares most content words with it (Jaccard >= 0.45 on 5-letter word stems, so a long
    sentence that merely mentions the same nouns does not win). (sentence, page) or (None, None)."""
    stop = {'assum', 'that', 'this', 'with', 'from', 'each'}
    words = lambda t: {w.rstrip('s')[:5] for w in re.findall(r'[a-z]{4,}', t.lower())} - stop   # 'blocks' = 'block'
    want = words(text)
    best = max(((len(want & words(s['text'])) / max(len(want | words(s['text'])), 1), s) for s in snippets
                if re.search(r'assum', s['text'], re.I)), key=lambda t: t[0], default=(0, None))
    return (best[1]['text'], best[1]['page'] + 1) if best[0] >= 0.45 else (None, None)


def _quoted(quote, snippets):
    """(quote, page) if the quote is in the paper (letters and digits compared), else (None, None)."""
    q = squash(quote)
    hit = next((s for s in snippets if len(q) >= 12 and q in squash(s['text'])), None) if quote else None
    return (quote, hit['page'] + 1) if hit else (None, None)


def _flat(m) -> dict:
    return {k: v for k, v in m.items() if k != 'name'} | {'model_name': m['name']}


def _identity(paper_dir, blocks) -> dict:
    """Sup id = the folder name (author_year, set when the paper was added); title and DOI from the OCR."""
    title = next((b['content'].strip() for b in blocks if b['label'] == 'doc_title'), None)
    text = ' '.join(b['content'] or '' for b in blocks if b['page'] == 0) or ' '.join(b['content'] or '' for b in blocks)
    doi = re.search(r'10\.\d{4,9}/[^\s,;)\]]+', text)
    return {'sup': os.path.basename(os.path.normpath(paper_dir)), 'title': title,
            'doi': doi.group(0).rstrip('.') if doi else None}


def _message(blocks, formulas, ids, st, stated, snippets, snippets_all=()) -> str:
    """Title, abstract, every equation with code's tag and the text just before it, the sentences that
    list equations, and the sentences about the model. Short: the free tier allows 8000 tokens a minute."""
    title = next((b['content'] for b in blocks if b['label'] == 'doc_title'), '')
    abstract = next((b['content'] for b in blocks if b['label'] == 'abstract'), '')[:1000]
    eqs = [f'({n}) [{st["tag"][n]}] p.{f["page"] + 1}  ...{_text_before(blocks, f)}\n    {clean(f["latex"])[:240]}'
           for n, f in zip(ids, formulas)]
    lists = [f'- "{s[:300]}"  -> eqs {", ".join(e)}' for s, e in stated]
    said = [s for s in snippets if s['kind'] == 'sentence' and STATEMENT.search(s['text'])
            and all(s['text'] != t for t, _ in stated)][:MAX_STATEMENTS]
    return (f'Title: {title}\n\nAbstract: {abstract}\n\nEquations (number, code tag, page, text just before, LaTeX):\n'
            + '\n'.join(eqs) + '\n\nSentences in which the paper lists equations:\n' + ('\n'.join(lists) or '(none)')
            + '\n\nOther sentences about the equations, the model and its assumptions:\n'
            + '\n'.join(f'- (p.{s["page"] + 1}) {s["text"][:220]}' for s in said)
            + _fit_tables(snippets_all))


FIT_TABLE = re.compile(r'paramet|estimat|fitt|fitted|best|model|rmse|error|kinetic', re.I)


def _fit_tables(snippets, max_chars=5000, row_chars=150) -> str:
    """Tables that say which models were fitted and with what values (Owhondah Table 5: the best fit for
    each structure and data set). The boundary depends on them when a paper fits many combinations but
    reports values for only some. Each table's title is printed once, then its rows, shortened."""
    tables = {}
    for x in snippets:
        if x['kind'] != 'table_row':
            continue
        title, _, row = x['text'].partition(': ') if x['text'][:5].lower() == 'table' else ('(untitled table)', '', x['text'])
        if FIT_TABLE.search(title) or FIT_TABLE.search(row[:60]):
            tables.setdefault((x['page'], title), []).append(row)
    out, size = [], 0
    for (page, title), rows in tables.items():
        part = f'- (p.{page + 1}) {title[:160]}\n' + '\n'.join(f'    {r[:row_chars]}' for r in rows)
        if size + len(part) > max_chars:
            break
        out.append(part)
        size += len(part)
    return '\n\nTables about fitted models:\n' + '\n'.join(out) if out else ''


def _text_before(blocks, f) -> str:
    """The last 140 characters of text printed before the equation on its page."""
    k = next((i for i, b in enumerate(blocks) if b['page'] == f['page'] and b['bbox'] == f['bbox']), None)
    for b in reversed(blocks[:k] if k is not None else []):
        if b['page'] != f['page']:
            break
        if b['label'] in TEXT_LABELS and b['content'].strip():
            return b['content'].strip()[-140:]
    return ''


def main(paper_dir: str):
    path = f'{paper_dir}/paper.json'
    if '--approve' in sys.argv:
        cfg = json.load(open(path))
        for change in _boundary_corrections(paper_dir, cfg):
            print('  from corrections.json:', change)
        blocks = json.load(open(f'{paper_dir}/paddle_blocks.json'))['blocks']   # law-level assumptions, for the
        formulas = [f for f in load_equations(paper_dir)[0] if f['eq'] is not None]   # approved equation list
        for m in cfg.get('models') or ([cfg] if cfg.get('equations') else []):   # free laws only: no model
            for a in m.get('model_assumptions', []):
                a.pop('eq', None)
            _attach_to_laws(m, formulas, [str(f['eq']) for f in formulas], blocks)
            for a in m.get('model_assumptions', []):
                if a.get('eq'):
                    print(f"  assumption belongs to eq {a['eq']} only: {a['text'][:60]}")
        cfg['status'] = 'approved'
        json.dump(cfg, open(path, 'w'), indent=1, ensure_ascii=False)
        print(f'approved -> {path}')
        return

    prop = propose(paper_dir)
    existing = json.load(open(path)) if os.path.exists(path) else None
    slug = re.sub(r'[^a-z0-9]+', '-', llm.MODEL.split('/')[-1].lower()).strip('-')   # one file per model compared
    out = path if existing is None or existing.get('status') == 'proposed' else f'{paper_dir}/paper_proposed_{slug}.json'
    json.dump(prop, open(out, 'w'), indent=1, ensure_ascii=False)

    for x in prop['code_analysis']['stated']:
        print(f"paper states: eqs {', '.join(x['equations'])}   (\"{x['sentence'][:80]}...\")")
    for m in prop.get('models', [prop]):
        print(f"model: {m.get('name') or m.get('model_name')} | eqs {', '.join(m['equations'])} | "
              f"boundary {m['boundary_basis']} | complete: {m['complete_model']} | "
              f"{len(m['model_assumptions'])} quoted assumptions")
    print('excluded:', ', '.join(f"{e['eq']} ({e['reason'][:45]})" for e in prop['excluded']))
    for c in prop['checks']:
        print('CHECK:', c)
    if out != path:
        from extractor.paper import models as _models
        approved = [set(map(str, a.get('equations') or [])) for a in _models(existing)]
        for m in prop.get('models', [prop]):
            new = set(m['equations'])
            old = max(approved, key=lambda a: len(a & new), default=set())   # the approved model closest to it
            print(f"vs the approved paper.json ({m.get('name') or m.get('model_name')}): "
                  + ('same equations' if old == new else f'only in proposal {sorted(new - old, key=key)}, '
                                                         f'only in approved {sorted(old - new, key=key)}'))
    print(f'-> {out}' + ('' if out != path else '   (status: proposed; check it, then --approve)'))


if __name__ == '__main__':
    llm.DRY_RUN = '--dry-run' in sys.argv
    main(sys.argv[1])
