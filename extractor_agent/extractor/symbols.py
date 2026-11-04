"""symbols.py -- every symbol of the model: description, unit, value, Parameter or Variable.

    python -m extractor.symbols papers/kontoravdi_2010              # run (answers cached)
    python -m extractor.symbols papers/kontoravdi_2010 --dry-run    # how many tokens, send nothing
    python -m extractor.symbols papers/kontoravdi_2010 --refresh    # ignore the cache

Writes papers/<id>/symbols.json, which laws.py and graph.py read. Replaces classification.json.

Code first:
    which laws compute and use each symbol (paper.py), a value printed as 'symbol = number unit',
    the table row that gives the symbol a number, and the 3 snippets that best define it.
    A symbol some law computes is a Variable: the LLM is not asked about its class.
The LLM, in batches of BATCH symbols, only the equations and snippets of that batch:
    description   the paper's words if it defines the symbol (stated, quoted), otherwise its own
                  judgement from the equations (inferred)
    unit, value   as printed (stated) or inferred from the equations; null if it cannot tell
    class         parameter | variable, for symbols no law computes (same rules as classify.py)
    estimation    parameters: estimated here | from literature | set | not said
    measured      variables: measured | not said
    initial_value variables: the starting value of the run the model was fitted to or simulated
                  (these three only with a quote; added 26 Oct for Layer 3: which parameters to
                  estimate, which outputs have data, how to reproduce the run)
    uncertainty   parameters: the reported uncertainty of the value, as printed (every number checked)
    same_as       the symbol this one is the same quantity as, written differently (S and s)
Checks by code: a quote must be in the paper; a stated unit must be in the cited snippet; estimation,
measured and initial_value without a quote found in the paper are dropped to 'not said'.
Anything failing a check becomes inferred. status = accepted when class and description are
stated (or class is set by structure); everything else is listed for review.
"""
from __future__ import annotations
import json
import re
import sys

from extractor import llm, triage
from extractor.classify import _rank, _which, squash
from extractor.paper import PAPER_NOTE, load_model, load_snippets, mention, table_value, with_paper

BATCH = 10                # symbols per call; a batch the model cannot answer in full is split in two
SNIPPETS_PER_SYMBOL = 3
SNIPPET_CHARS = 260

PROMPT = """You document the symbols of a published mechanistic process model.

For each symbol listed under "Symbols", give:

description  One short line: what the quantity is, in the paper's terms ("viable cell density",
             "Monod saturation constant for glucose"). If a snippet defines the symbol, set
             description_basis "stated", give the snippet id and copy the defining words exactly
             as quote. If no snippet defines it, describe it from its role in the equations, set
             description_basis "inferred", snippet_id and quote null.
unit         As printed in the paper (nomenclature, table, text), e.g. "mmol L^-1", "h^-1", "-" for
             dimensionless; unit_basis "stated". If the paper gives none, infer it from the equations
             only if that is unambiguous (unit_basis "inferred"); otherwise null and "unknown".
value        The numerical value the paper reports for the model named under "Model" (parameter
             tables, "K = 0.5 mM"), as printed, with its unit; null if none. Papers often print
             several sets: an original and an improved model, fits to different experiments, a
             literature value and a fitted one. Give only the set that belongs to the named model.
             Table rows start with their table's title, which usually says which model or
             experiment the set belongs to. If you cannot tell which set is the named model's,
             give every candidate as "value unit (p.N, table)" separated by " ; ".
value_number The value as a plain number in the unit you give under unit, without the error: 0.0072 for
             "7.2 +- 0.4 x 10^-3 h^-1" with unit "h^-1"; 6.3 for "6.3 +- 1.0" with unit
             "10^-12 mmol cell^-1 h^-1" (a table that prints a scale factor next to the unit: the unit
             keeps the factor). null when value is null or when you list several candidates.
value_set    "this model" if the table or text ties the value to the named model; "only one set"
             if the paper prints a single value for it; "unclear" if you could not tell (then
             list the candidates); null when value is null.
values_by_model  When the paper gives this symbol a separate value for each model or law listed under
             "Model" (Monod and Contois each with their own mu_max): one entry per model, name exactly
             as listed. Then value is null. Empty list otherwise.
values_by_case   When the paper gives separate values for different data sets or conditions of the same
             model (per protein, per pH, per substrate, per experiment): one entry per case, case = a
             few words ("ribonuclease A, pH 5"), value as printed. Then value is null and value_set
             null. Empty list otherwise. A transformed value (ln k_b = 45.8) is given as printed.
class        Only for symbols marked NOT COMPUTED. First write class_reason (one sentence), then
             class, parameter or variable:
               parameter  a constant of a law: rate constants, maximum rates, saturation and
                          inhibition constants, yields, maintenance coefficients, efficiencies,
                          activation energies, exponents, transfer coefficients, physical properties
                          (also saturation concentrations, solubilities, equilibrium constants, even
                          when they depend on temperature), universal constants. Being "not computed" does not make it a variable.
               variable   a condition of how the process is run, or a process quantity that changes:
                          flow rates, feed or inlet concentrations, feed temperature, set-points,
                          manipulated or measured operating quantities, and concentrations, pH or
                          temperature the model uses without computing them.
             The test: does the paper fit or estimate it, report a value for it, or treat it as a
             property of the material or the system (a fraction, a specific resistance, a particle
             size, a proportionality constant)? Then parameter. Does the operator set or measure it
             for each run (pressure, feed concentration, flow rate, temperature)? Then variable.
             A number in a table does not decide it either way: tables list fitted constants but
             also feed conditions, flow rates and initial conditions. class_basis "stated" only if
             the paper says what the symbol is. class_reason must name which of these it is, not
             only "not computed".
             For symbols marked COMPUTED: class_reason "computed by a law", class "variable",
             class_basis "stated".

Three more fields. Answer them ONLY from what the paper says, with the paper's exact words as the
matching quote. "not said" (or null) is the expected answer whenever the paper does not say it: never
fill one in from what is usual for this kind of model or process. For a value in a table, quote the
table's title.
estimation      Parameters only: "estimated" (fitted to data in this paper), "from literature" (taken
                from another work, cited or said so), "set" (fixed by the authors without fitting: a
                chosen value, a physical constant, a property), or "not said". Variables: "not said".
measured        Variables only: "measured" if the paper measured this quantity in the experiments the
                model is fitted to or compared with, otherwise "not said". Parameters: "not said".
initial_value   Variables only: the starting value the paper gives for the run the model was fitted to
                or simulated (a state's initial value, a set input such as a feed concentration or flow),
                with its unit ("2 L", "300 g/L"); null if the paper gives none.
estimation_quote, measured_quote, initial_quote: the words that support each answer; null when the
                answer is "not said" or null.
uncertainty     Parameters with a value: the uncertainty the paper reports for that value, as printed
                (standard deviation, "± 0.4", a confidence interval "0.35-0.66", a relative standard
                deviation "15.2 %"); null if the paper reports none. Never compute one yourself.
same_as         If this symbol is the same quantity as another symbol of the model written differently
                (the paper writes substrate concentration both as S and as s; K_s and K_S for the same
                constant), that other symbol exactly as listed under "All symbols of the model"; otherwise
                null. Only when the paper's text or the equations make clear it is the same quantity: two
                symbols that differ only in case are often different quantities (k_1 rate constant,
                K_1 equilibrium constant).

Answer every listed symbol exactly once, writing the symbol exactly as given. Do not answer other
symbols.""" + PAPER_NOTE

SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['symbols'],
    'properties': {'symbols': {'type': 'array', 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['symbol', 'description', 'description_basis', 'snippet_id', 'quote', 'unit', 'unit_basis',
                     'value', 'value_number', 'value_set', 'values_by_model', 'values_by_case', 'class_reason', 'class', 'class_basis',
                     'estimation', 'estimation_quote', 'measured', 'measured_quote', 'initial_value', 'initial_quote',
                     'uncertainty', 'same_as'],
        'properties': {
            'symbol': {'type': 'string'},
            'description': {'type': 'string'},
            'description_basis': {'type': 'string', 'enum': ['stated', 'inferred']},
            'snippet_id': {'type': ['integer', 'null']},
            'quote': {'type': ['string', 'null']},
            'unit': {'type': ['string', 'null']},
            'unit_basis': {'type': 'string', 'enum': ['stated', 'inferred', 'unknown']},
            'value': {'type': ['string', 'null']},
            'value_number': {'type': ['number', 'null']},
            'value_set': {'type': ['string', 'null'], 'enum': ['this model', 'only one set', 'unclear', None]},
            'values_by_case': {'type': 'array', 'items': {
                'type': 'object', 'additionalProperties': False, 'required': ['case', 'value'],
                'properties': {'case': {'type': 'string'}, 'value': {'type': 'string'}}}},
            'values_by_model': {'type': 'array', 'items': {
                'type': 'object', 'additionalProperties': False, 'required': ['model', 'value'],
                'properties': {'model': {'type': 'string'}, 'value': {'type': 'string'}}}},
            'class_reason': {'type': 'string'},           # before class: the reason is written first
            'class': {'type': 'string', 'enum': ['parameter', 'variable']},
            'class_basis': {'type': 'string', 'enum': ['stated', 'inferred']},
            'estimation': {'type': 'string', 'enum': ['estimated', 'from literature', 'set', 'not said']},
            'estimation_quote': {'type': ['string', 'null']},
            'measured': {'type': 'string', 'enum': ['measured', 'not said']},
            'measured_quote': {'type': ['string', 'null']},
            'initial_value': {'type': ['string', 'null']},
            'initial_quote': {'type': ['string', 'null']},
            'uncertainty': {'type': ['string', 'null']},
            'same_as': {'type': ['string', 'null']}}}}}}


def describe(paper_dir: str, refresh: bool = False) -> dict:
    formulas, reads, _ = load_model(paper_dir)
    eqs = [str(f.get('eq') or i + 1) for i, f in enumerate(formulas)]
    laws = list(zip(eqs, reads))
    computed_by = {}
    for n, r in laws:
        for s in r['lhs']:
            computed_by.setdefault(s, []).append(n)
    symbols = list(dict.fromkeys(s for _, r in laws for s in r['lhs'] + r['rhs']))
    snippets = load_snippets(paper_dir)
    by_id = {s['id']: s for s in snippets}
    printed = {r['lhs'][0]: r['value'] for _, r in laws if r.get('value') and len(r['lhs']) == 1}
    intro = _intros(paper_dir, formulas, eqs)
    names = _model_names(paper_dir)

    answers = {}

    def run(batch, k, retry=True):
        msg = with_paper(paper_dir, _message(batch, laws, computed_by, snippets, intro, names)
                         + '\n\nAll symbols of the model (for same_as): ' + ', '.join(symbols))
        try:
            schema = json.loads(json.dumps(SCHEMA))      # symbol limited to this batch's symbols
            schema['properties']['symbols']['items']['properties']['symbol'] = {'type': 'string', 'enum': list(batch)}
            schema['properties']['symbols']['items']['properties']['same_as'] = {'type': ['string', 'null'], 'enum': symbols + [None]}
            if names:                                     # values per model: only the names listed under Model
                schema['properties']['symbols']['items']['properties']['values_by_model']['items']['properties']['model'] = \
                    {'type': 'string', 'enum': names}
            out = llm.cached_ask(paper_dir, PROMPT, msg, schema, 'symbols', 'low',
                                 label=f'symbols {k + 1}-{k + len(batch)} of {len(symbols)}', refresh=refresh)
        except llm.JSONFailed:
            if len(batch) == 1:
                print(f'  {batch[0]}: no valid answer, left for review')
                return
            print(f'  symbols {k + 1}-{k + len(batch)}: answer did not fit, splitting in two')
            h = len(batch) // 2
            run(batch[:h], k)
            run(batch[h:], k + h)
            return
        for a in (out or {}).get('symbols', []):
            s = _which(a['symbol'], batch)
            if s is not None and s not in answers:
                answers[s] = a
        missing = [s for s in batch if s not in answers]
        if out and missing and retry:                     # the LLM skipped some: ask once more for those only
            print(f'  {len(missing)} symbols skipped, asking again: {missing}')
            run(missing, k, retry=False)

    for k in range(0, len(symbols), BATCH):
        run(symbols[k:k + BATCH], k)

    result = {}
    for s in symbols:
        a = answers.get(s)
        entry = {'computed_by': computed_by.get(s, []),
                 'used_in': [n for n, r in laws if s in r['rhs']],
                 'table': (table_value(s, snippets) or [None])[0]}
        entry.update(_checked(a, s in computed_by, by_id, printed.get(s)) if a else
                     {'class': 'variable', 'status': 'pending', 'review': ['no answer from the LLM']})
        result[s] = entry
    return {'model': llm.MODEL, 'symbols': result}


def _message(batch, laws, computed_by, snippets, intro=None, names=None) -> str:
    """The equations that contain the batch symbols, each with the sentence that introduces it ('The
    heavy- and light-chain mRNA balances are'), then per symbol: computed or not, and the snippets
    that best define it. Papers often define parameters in a nomenclature but states only through
    that sentence."""
    intro = intro or {}
    eqs = [f'({n}) {r["latex"]}' + (f'      <- "{intro[n]}"' if intro.get(n) else '')
           for n, r in laws if set(batch) & set(r['lhs'] + r['rhs'])]
    blocks = []
    for s in batch:
        head = f'{s}   [COMPUTED by eq {", ".join(computed_by[s])}]' if s in computed_by else f'{s}   [NOT COMPUTED]'
        names_ = [s] + ([b] if (b := _without_index(s)) != s else [])   # nu_i is written nu in the tables
        hits = [x for x in snippets if any(mention(n, x['text']) in ('exact', 'notation', 'confusable') for n in names_)]
        hits.sort(key=lambda x: _rank(s, x))
        shown = hits[:SNIPPETS_PER_SYMBOL]
        rows = [x for x in hits if x['kind'] == 'table_row' and re.search(r'\|[^|]*\d', x['text']) and x not in shown]
        rows.sort(key=lambda x: 0 if VALUE_TABLE.search(x['text'].split(': ', 1)[0]) else 1)   # fitted values first
        shown += rows[:3]                                  # a table row with a number: where values come from
        lines = [f'  [{x["id"]}] ({x["kind"]}, p.{x["page"] + 1}) {_short(x)}' for x in shown]
        blocks.append(head + '\n' + ('\n'.join(lines) or '  (no snippet mentions this symbol)'))
    head = f'Model: {" | ".join(names)}\n\n' if names else ''
    return head + 'Equations:\n' + '\n'.join(eqs) + '\n\nSymbols:\n\n' + '\n\n'.join(blocks)


VALUE_TABLE = re.compile(r'estimat|fitted|\bfit\b|parameter|optim|result', re.I)


def _without_index(sym: str) -> str:
    """nu_{i} -> nu, k_{ads,i} -> k_{ads}: tables often drop the generic index (one row per protein instead)."""
    t = re.sub(r',\s*[ijk]\s*\}', '}', sym)
    t = re.sub(r'_\{\s*[ijk]\s*\}$|_[ijk]$', '', t)
    return t


def _short(x) -> str:
    """A snippet for the prompt. A table row carries its table's title in front; a long title (Anane: 200
    characters) must not push the row's values out, so the title is shortened, not the row."""
    if x['kind'] == 'table_row' and ': ' in x['text']:
        title, row = x['text'].split(': ', 1)
        return (title[:90] + ('...' if len(title) > 90 else '') + ': ' + row)[:SNIPPET_CHARS + 200]
    return x['text'][:SNIPPET_CHARS]


def _model_names(paper_dir) -> list[str]:
    """The names of the paper's Sups (paper.json), and of each free-floating law, so the LLM can pick the
    right parameter set or give one value per model."""
    try:
        cfg = json.load(open(f'{paper_dir}/paper.json'))
    except (OSError, ValueError):
        return []
    from extractor.paper import models
    out = []
    for m in models(cfg):                                 # free-floating laws are fitted one by one: list each
        out += [n for n in m.get('law_names', {}).values() if n] if m.get('free') else [m['name']] if m.get('name') else []
    return list(dict.fromkeys(out))


def several_values(v) -> bool:
    """True when the value lists more than one candidate set ('17.64 (p.6) ; 100 (p.14)')."""
    return bool(v) and ';' in str(v)


def _intros(paper_dir, formulas, eqs) -> dict:
    """eq -> the last 160 characters of the text printed before it (across other equations of the
    same section), from laws.py's reading of the page."""
    from extractor.laws import _position, _text_near
    blocks = json.load(open(f'{paper_dir}/paddle_blocks.json'))['blocks']
    out = {}
    for n, f in zip(eqs, formulas):
        k = _text_near(blocks, _position(blocks, f), -1)
        while k is not None and len(blocks[k]['content'].strip()) < 15:   # 'where', 'and': look further back
            k = _text_near(blocks, k, -1)
        if k is not None:
            out[n] = ' '.join(blocks[k]['content'].split())[-160:]
    return out


def _checked(a: dict, computed: bool, by_id: dict, printed_value: str | None) -> dict:
    """The LLM's answer after the code checks. 'llm' keeps what it said."""
    review = []
    found = _find(a['quote'], a['snippet_id'], by_id) if a['description_basis'] == 'stated' else None
    d_basis = 'stated' if found else 'inferred'
    if a['description_basis'] == 'stated' and not found:
        review.append('description quote not found in the paper')
    unit, u_basis = a['unit'], a['unit_basis']
    if u_basis == 'stated' and unit and not (_unit_printed(unit, a['snippet_id'], by_id) or found and _unit_printed(unit, found['id'], by_id)):
        u_basis = 'inferred'
    if computed:
        cls, c_basis, c_reason = 'variable', 'structure', 'computed by a law'
    else:
        cls, c_basis, c_reason = a['class'], a['class_basis'], a['class_reason']
        other = 'variable' if cls == 'parameter' else 'parameter'
        if other in c_reason.lower() and cls not in c_reason.lower():
            review.append(f'class {cls} but the reason says {other}: {c_reason}')
        if c_basis == 'inferred':
            review.append(f'class inferred: {c_reason}')
    if d_basis == 'inferred':
        review.append('description inferred')
    value = printed_value or a['value']
    unclear = several_values(value) or (bool(value) and a.get('value_set') == 'unclear' and not printed_value)
    if unclear:
        review.append('value: not clear which set belongs to this model; keep the right one')
    if u_basis in ('inferred', 'unknown'):
        review.append(f'unit {u_basis}')
    # estimation / measured / initial value: kept only with a quote found in the paper, else 'not said'
    quoted = lambda q: _find(q, None, by_id) if q else None
    est = a.get('estimation', 'not said') if cls == 'parameter' else 'not said'
    meas = a.get('measured', 'not said') if cls == 'variable' else 'not said'
    init = a.get('initial_value') if cls == 'variable' else None
    if est != 'not said' and not quoted(a.get('estimation_quote')):
        review.append(f'estimation "{est}" dropped: no quote found'); est = 'not said'
    if meas != 'not said' and not quoted(a.get('measured_quote')):
        review.append('measured dropped: no quote found'); meas = 'not said'
    if init and not quoted(a.get('initial_quote')):
        review.append(f'initial value "{init}" dropped: no quote found'); init = None
    # uncertainty: every number in it must be printed in the paper (no computed spreads)
    unc = a.get('uncertainty') if cls == 'parameter' else None
    if unc and not all(any(squash(n) in squash(x['text']) for x in by_id.values()) for n in re.findall(r'\d+(?:\.\d+)?', unc)):
        review.append(f'uncertainty "{unc}" dropped: a number in it is not in the paper'); unc = None
    same = a.get('same_as') if a.get('same_as') not in (None, a.get('symbol')) else None
    if same:
        review.append(f'same quantity as {same} (merged when linking)')
    extra = {'uncertainty': unc, 'same_as': same, 'estimation': est, 'estimation_quote': a.get('estimation_quote') if est != 'not said' else None,
             'measured': meas, 'measured_quote': a.get('measured_quote') if meas != 'not said' else None,
             'initial_value': init, 'initial_quote': a.get('initial_quote') if init else None}
    return {**extra, 'description': a['description'], 'description_basis': d_basis,
            'quote': a['quote'] if found else None, 'page': found['page'] + 1 if found else None,
            'unit': unit, 'unit_basis': u_basis,
            'value': value, 'value_number': None if unclear or printed_value else a.get('value_number'),
            'value_set': 'unclear' if unclear else a.get('value_set'),
            'values_by_model': {x['model']: x['value'] for x in a.get('values_by_model') or []},
            'values_by_case': {x['case']: x['value'] for x in a.get('values_by_case') or []},
            'class': cls, 'class_basis': c_basis, 'class_reason': c_reason,
            'status': 'accepted' if d_basis == 'stated' and c_basis != 'inferred' and not unclear else 'pending',
            'review': review, 'llm': a}


def _find(quote, snippet_id, by_id):
    """The snippet that holds the quote (the cited one first; the LLM sometimes cites the wrong id)."""
    q = squash(quote)
    if len(q) < 6:
        return None
    cited = by_id.get(snippet_id)
    return cited if cited and q in squash(cited['text']) else \
        next((x for x in by_id.values() if q in squash(x['text'])), None)


def _unit_printed(unit, snippet_id, by_id) -> bool:
    u = squash(unit)
    cited = by_id.get(snippet_id)
    return bool(u) and (cited is not None and u in squash(cited['text'])
                        or any(u in squash(x['text']) for x in by_id.values() if x['kind'] != 'sentence'))


def main(paper_dir: str, refresh: bool):
    result = describe(paper_dir, refresh)
    print(llm.usage_line())
    if llm.DRY_RUN:
        return
    json.dump(result, open(f'{paper_dir}/symbols.json', 'w'), indent=1, ensure_ascii=False)
    syms = result['symbols']
    n = lambda f: sum(1 for v in syms.values() if f(v))
    print(f"{len(syms)} symbols: {n(lambda v: v['class'] == 'parameter')} parameter, "
          f"{n(lambda v: v['class'] == 'variable')} variable | descriptions: "
          f"{n(lambda v: v.get('description_basis') == 'stated')} stated, "
          f"{n(lambda v: v.get('description_basis') == 'inferred')} inferred | units: "
          f"{n(lambda v: v.get('unit_basis') == 'stated')} stated, {n(lambda v: v.get('unit_basis') == 'inferred')} inferred, "
          f"{n(lambda v: v.get('unit_basis') == 'unknown')} unknown | {n(lambda v: bool(triage.must_check(v)))} to check, {n(lambda v: bool(v.get('review')) and not triage.must_check(v))} for information")
    print(f'-> {paper_dir}/symbols.json')


if __name__ == '__main__':
    llm.DRY_RUN = '--dry-run' in sys.argv
    main(sys.argv[1], '--refresh' in sys.argv)
