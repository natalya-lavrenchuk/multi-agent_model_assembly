"""classify.py -- for every symbol no law computes: Parameter or Variable?

    export GROQ_API_KEY=...
    python -m extractor.classify papers/kontoravdi_2010                 # normal run
    python -m extractor.classify papers/kontoravdi_2010 --hide-tables   # test: LLM sees no tables
    python -m extractor.classify --compare run1.json run2.json ...      # which answers change between runs

Writes papers/<id>/classification.json, which graph.py uses for the pipeline graph.

Symbols a law computes are not asked about: structure already makes them Variables.
The LLM always answers, and says what the answer rests on:
    stated     the paper says it; the quote must appear word for word in the snippet
    inferred   the paper is silent; the LLM names the modelling convention it used
status = accepted for a stated answer whose quote checks out; everything else goes to review.

When the paper has tables, the answer is also compared with the table rule
(Parameter = has a numeric table value), which is an answer key for --hide-tables runs.
"""
from __future__ import annotations
import json
import os
import re
import sys

from extractor.llm import MODEL, ask
from extractor.paper import load_model, load_snippets, mention, table_value
BATCH = 6               # symbols per call: the free Groq tier allows 8000 tokens per minute
SNIPPETS_PER_SYMBOL = 4

PROMPT = """You classify symbols from a published mechanistic process model.

Every symbol below appears in the model's equations, and no equation computes it.
For each one decide: parameter or variable.

Definitions:
  parameter   a constant of the model during a run: rate constants, maximum rates,
              saturation (Monod, Contois) and inhibition constants, yields, maintenance
              coefficients, activation energies, exponents, heat and mass transfer
              coefficients, physical properties and universal constants (R, densities,
              heat capacities, saturation concentrations of a dissolved gas).
  variable    supplied from outside the model, and may change during a run or between
              runs: feed and outlet flow rates, feed or inlet concentrations, feed
              temperature, coolant flow, agitation power, aeration, set-points, other
              manipulated or measured operating quantities; also quantities of the
              process that change during a run (concentrations, pH, temperature) when
              the model uses them without computing them.

A number in a table does not make a symbol a parameter: tables also list feed
concentrations, feed temperatures, flow rates and initial conditions. Ask: is this a
constant of a law, or a condition of how the process is run or a quantity that changes?

Use the paper first. If a snippet states what the symbol is ("feed flow rate",
"Contois saturation constant"), set basis to "stated", give the snippet id, and copy the
supporting words exactly. If the paper does not say, decide from the symbol's role in the
equations and from standard modelling convention: set basis to "inferred" and name the
convention, e.g. "denominator of a Monod-type term, conventionally a saturation constant".
Always give an answer. Never leave a symbol unclassified.

Answer only the symbols listed under "Symbols", each exactly once, writing the symbol exactly
as given. Do not answer other symbols that appear in the equations."""

SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['symbols'],
    'properties': {'symbols': {'type': 'array', 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['symbol', 'class', 'basis', 'snippet_id', 'quote', 'reason'],
        'properties': {
            'symbol': {'type': 'string'},
            'class': {'type': 'string', 'enum': ['parameter', 'variable']},
            'basis': {'type': 'string', 'enum': ['stated', 'inferred']},
            'snippet_id': {'type': ['integer', 'null']},
            'quote': {'type': ['string', 'null']},
            'reason': {'type': 'string'}}}}}}


def classify(paper_dir: str, hide_tables: bool = False) -> dict:
    formulas, reads, _ = load_model(paper_dir)
    reads = list(zip([f.get('eq') or i + 1 for i, f in enumerate(formulas)], reads))
    computed = {s for _, r in reads for s in r['lhs']}
    todo = sorted({s for _, r in reads for s in r['rhs']} - computed)
    snippets = load_snippets(paper_dir, tables=not hide_tables)
    by_id = {s['id']: s for s in snippets}

    answers, duplicates, extras = {}, [], []
    for k in range(0, len(todo), BATCH):
        batch = todo[k:k + BATCH]
        for a in ask(PROMPT, _message(batch, reads, snippets), SCHEMA, 'classification')['symbols']:
            s = _which(a['symbol'], batch)
            if s is None:
                extras.append(a)                          # a symbol we did not ask about
            elif s in answers:
                duplicates.append(a)                      # the same symbol answered twice: first answer kept
            else:
                answers[s] = _check(a, by_id)
        print(f'  {min(k + BATCH, len(todo))}/{len(todo)} symbols')

    out = {'model': MODEL, 'tables_shown': not hide_tables,
           'duplicate_answers': duplicates, 'extra_answers': extras, 'symbols': {}}
    for s in todo:
        out['symbols'][s] = answers.get(s) or {'class': 'variable', 'basis': 'inferred',
                                                'quote': None, 'page': None, 'status': 'pending',
                                                'reason': 'no answer from the LLM; kept as Variable'}
    return out


def _message(batch, reads, snippets) -> str:
    """Equations that contain the batch symbols, then per symbol the snippets that mention it."""
    eqs = [f'({n}) {r["latex"]}' for n, r in reads if set(batch) & set(r['lhs'] + r['rhs'])]
    blocks = []
    for s in batch:
        hits = [x for x in snippets if mention(s, x['text']) in ('exact', 'notation', 'confusable')]
        hits.sort(key=lambda x: _rank(s, x))                     # stable: reading order within each rank
        lines = [f'  [{x["id"]}] ({x["kind"]}, p.{x["page"] + 1}) {x["text"][:300]}' for x in hits[:SNIPPETS_PER_SYMBOL]]
        blocks.append(f'{s}\n' + ('\n'.join(lines) or '  (no snippet mentions this symbol)'))
    return 'Equations:\n' + '\n'.join(eqs) + '\n\nSymbols and the snippets that mention them:\n\n' + '\n\n'.join(blocks)


UNICODE_GREEK = dict(zip('αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ',
                         ['alpha', 'beta', 'gamma', 'delta', 'varepsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa',
                          'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi',
                          'omega', 'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Phi', 'Psi', 'Omega']))


UNICODE_SUB = 'ₐₑₒₓₔₕₖₗₘₙₚₛₜᵢᵣᵤᵥⱼ₀₁₂₃₄₅₆₇₈₉'
SUB_TO_ASCII = str.maketrans(UNICODE_SUB, 'aeoxəhklmnpstiruvj0123456789')


def _which(answered: str, batch: list[str]) -> str | None:
    """The batch symbol the LLM meant. It may rewrite notation ('\\alpha_1' for '\\alpha_{1}', 'μ_max')."""
    if answered in batch:
        return answered
    for ch, name in UNICODE_GREEK.items():
        answered = answered.replace(ch, '\\' + name)
    sub = re.search(f'[{UNICODE_SUB}]+$', answered)                # 'μₚ' -> '\\mu_{p}'
    if sub:
        answered = answered[:sub.start()] + '_{' + sub.group(0).translate(SUB_TO_ASCII) + '}'
    answered = re.sub(r'_([A-Za-z0-9,]+)', r'_{\1}', answered)     # 'm_glc' -> 'm_{glc}'
    answered = re.sub(r'\^([A-Za-z0-9*])(?![A-Za-z0-9}])', r'^{\1}', answered)   # 'mu_m^A' -> '^{A}'
    if answered in batch:                                         # exact after rewriting: '\\mu_{m}^{A}'
        return answered
    plain = lambda x: re.sub(r'\^\{[^}]*\}', '', x)              # a superscript is part of the name:
    return next((s for s in batch if plain(s) == s and mention(s, answered) == 'exact'), None) \
        if '^' not in answered else next((s for s in batch if s == answered), None)


DEFINITION_CUE = re.compile(r'\b(where|is the|are the|denotes?|represents?|stands for)\b|=\s*\\text', re.I)


def _rank(symbol: str, snip: dict) -> int:
    """Which snippets the LLM sees first: 0 a definition (a nomenclature line, or 'where ... is the ...'
    with the symbol written as math), 1 the symbol written as math, 2 prose that only shares letters
    (the article 'a', an author footnote 'b')."""
    text = snip['text']
    math = [m for m in re.findall(r'\$([^$]*)\$', text) if not m.strip().startswith('^')]   # not footnote marks
    first = text.split()[0] if text.split() else ''
    as_math = any(mention(symbol, m) == 'exact' for m in math + [first])
    if as_math and (snip['kind'] == 'line' or DEFINITION_CUE.search(text)):
        return 0
    return 1 if as_math else 2


def squash(t: str | None) -> str:
    """Letters and digits only, LaTeX and Unicode spellings made equal: how quotes are compared with the paper."""
    for ch, name in UNICODE_GREEK.items():               # 'α_1' and '$ \\alpha_{1} $' compare equal; '×' and '\\times' too
        t = (t or '').replace(ch, name)
    t = (t or '').replace('×', 'times').replace('·', 'cdot').replace('−', '-')   # Unicode vs LaTeX
    t = re.sub(r'\\(text|mathrm|mathit|mathbf|operatorname|left|right|[bB]igg?)\b', '', t)   # formatting only
    t = re.sub(r'\^\s*\{*\s*[a-z*†‡]\s*\}*', '', t)                                       # footnote marks: ^{a}, ^{*}
    t = re.sub(r'var(epsilon|phi|theta)', r'\1', t.lower())
    return re.sub(r'[^a-z0-9]', '', t)


def _check(a: dict, by_id: dict) -> dict:
    """Keep 'stated' only if the quote is really in the paper: in the cited snippet, or (the LLM
    sometimes cites the wrong id) in another snippet of the paper; otherwise downgrade to 'inferred'.
    Compared on letters and digits only, so LaTeX markup ('K $ _{A} $' vs 'K_A') does not matter.
    The LLM's own answer is kept under 'llm'."""
    q = squash(a['quote'])
    cited = by_id.get(a['snippet_id'])
    found = None
    if a['basis'] == 'stated' and len(q) >= 8:
        found = cited if cited and q in squash(cited['text']) else \
            next((x for x in by_id.values() if q in squash(x['text'])), None)
    ok = found is not None
    reason = a['reason'] if ok or a['basis'] == 'inferred' else a['reason'] + ' [quote not found in the paper]'
    return {'class': a['class'], 'basis': 'stated' if ok else 'inferred',
            'quote': a['quote'] if ok else None, 'page': found['page'] + 1 if ok else None,
            'status': 'accepted' if ok else 'pending', 'reason': reason, 'llm': a}


def main(paper_dir: str, hide_tables: bool):
    result = classify(paper_dir, hide_tables)
    json.dump(result, open(f'{paper_dir}/classification.json', 'w'), indent=1, ensure_ascii=False)

    syms = result['symbols'].values()
    n = lambda **match: sum(all(v[k] == x for k, x in match.items()) for v in syms)
    print(f"{len(syms)} symbols: {n(**{'class': 'parameter'})} parameter, {n(**{'class': 'variable'})} variable | "
          f"{n(basis='stated')} stated, {n(basis='inferred')} inferred | "
          f"{n(status='pending')} for review")

    if result['duplicate_answers'] or result['extra_answers']:
        print(f"ignored: {len(result['duplicate_answers'])} duplicate answers, {len(result['extra_answers'])} answers "
              f"for symbols not asked about (kept in classification.json)")
    key = load_snippets(paper_dir)                      # answer key from the tables, if the paper has any
    if any(s['kind'] == 'table_row' for s in key):
        wrong = [s for s, v in result['symbols'].items() if (v['class'] == 'parameter') != bool(table_value(s, key))]
        print(f'vs table rule (answer key): {len(syms) - len(wrong)}/{len(syms)} agree; differ: {wrong}')
    print(f'-> {paper_dir}/classification.json')


def compare_runs(paths: list[str]):
    """One line per symbol: the answer in each run (P = parameter, V = variable; lower case =
    inferred). Symbols whose class changes are marked."""
    runs = [json.load(open(p))['symbols'] for p in paths]
    code = lambda v: 'P' if v['class'] == 'parameter' else 'V'
    show = lambda v: code(v) if v['basis'] == 'stated' else code(v).lower()
    changed = 0
    print(f"{'symbol':18} " + ' '.join(f'run{i + 1}' for i in range(len(runs))))
    for s in runs[0]:
        answers = [r.get(s) for r in runs]
        flips = len({code(a) for a in answers if a}) > 1
        changed += flips
        print(f"{s:18} " + ' '.join(f'{show(a) if a else "-":>4}' for a in answers) + ('   <- changes' if flips else ''))
    print(f'\n{changed}/{len(runs[0])} symbols change between runs (P/V; case = stated/inferred, not counted)')


if __name__ == '__main__':
    if sys.argv[1] == '--compare':
        compare_runs(sys.argv[2:])
    else:
        main(sys.argv[1], '--hide-tables' in sys.argv)
