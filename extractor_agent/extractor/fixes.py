"""fixes.py -- a corrected form for each law flagged as a possible misprint, for a person to approve.

    python -m extractor.fixes papers/kontoravdi_2010              # run (answer cached)
    python -m extractor.fixes papers/kontoravdi_2010 --dry-run

Run after laws.py. Only laws with a possible misprint (laws.json 'issue') are sent, all in one call per paper,
after the whole paper. Per law the LLM answers:
    action          'correct' only when the paper itself shows the intended form (another equation written
                    correctly, the text, the units, a corrigendum); 'keep as printed' for notation, a unit-list
                    problem, or when unsure
    corrected_latex the corrected equation, changing only what is needed, with the printed symbols
    reason, quote   why, and the paper's words that show the intended form
Checks by code: the corrected equation must be read by the same reader as the printed ones; it may only use
symbols the model already has (no new quantities); the quote must be in the paper (else basis 'inferred').
Writes papers/<id>/fixes.json. graph.py writes an approved or proposed correction next to the printed equation
(mkg:latexCorrected), never instead of it; the review page shows it with a 'use corrected form' box.
The OCR fixes in paddle.json are different: they restore what the paper prints. These change what it prints.
"""
from __future__ import annotations
import json
import re
import os
import sys

from extractor import llm
from extractor.classify import squash
from extractor.graph import corrected
from extractor.paper import PAPER_NOTE, load_model, load_snippets, prose_names, read_equation, with_paper

PROMPT = """You check equations of a published mechanistic model that an earlier step flagged as possible
misprints. For each listed equation you get the equation as printed, the reason it was flagged, and what its
symbols mean.

action           "correct" only if the paper itself shows what the equation should be: another equation that
                 writes the same thing correctly, the text describing it, the units, the model's other
                 equations, or a corrigendum. "keep as printed" if the flag is about notation or formatting, a
                 unit listed inconsistently elsewhere (the equation itself is fine), or if you are not sure.
                 Never correct from what you believe the standard form is unless the paper supports it.
corrected_latex  For "correct": the whole corrected equation in LaTeX, written like the printed one, changing
                 only what is needed, and using only symbols that appear in the model (the list is given).
                 null for "keep as printed".
reason           One sentence: what was wrong and what the correction is, or why it is kept as printed.
quote            The paper's exact words (or an equation's text) that show the intended form; null if none.

Answer every listed equation exactly once.""" + PAPER_NOTE

SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['fixes'],
          'properties': {'fixes': {'type': 'array', 'items': {
              'type': 'object', 'additionalProperties': False,
              'required': ['eq', 'action', 'corrected_latex', 'reason', 'quote'],
              'properties': {'eq': {'type': 'string'},
                             'action': {'type': 'string', 'enum': ['correct', 'keep as printed']},
                             'corrected_latex': {'type': ['string', 'null']},
                             'reason': {'type': 'string'}, 'quote': {'type': ['string', 'null']}}}}}}


def propose(paper_dir: str, refresh: bool = False) -> dict:
    syms, laws = corrected(paper_dir)
    formulas, reads, _ = load_model(paper_dir)
    eqs = [str(f.get('eq')) for f in formulas]
    flagged = [n for n in eqs if laws.get(n, {}).get('issue')]
    if not flagged:
        return {'model': llm.MODEL, 'fixes': {}}
    by = dict(zip(eqs, zip(formulas, reads)))
    lines = []
    for n in flagged:
        f, r = by[n]
        meaning = '; '.join(f"{s}: {syms.get(s, {}).get('description', '?')} [{syms.get(s, {}).get('unit') or '?'}]"
                            for s in dict.fromkeys(r['lhs'] + r['rhs']))
        lines.append(f"({n}) {f['latex'].strip()}\n  flagged: {laws[n]['issue']}\n  symbols: {meaning}")
    model_syms = sorted(syms)
    msg = with_paper(paper_dir, 'Flagged equations:\n\n' + '\n\n'.join(lines)
                     + '\n\nSymbols of the model: ' + ', '.join(model_syms))
    schema = json.loads(json.dumps(SCHEMA))
    schema['properties']['fixes']['items']['properties']['eq'] = {'type': 'string', 'enum': flagged}
    a = llm.cached_ask(paper_dir, PROMPT, msg, schema, 'fixes', 'medium', label=f'fixes for eqs {", ".join(flagged)}',
                       refresh=refresh)
    if a is None:
        return {'model': llm.MODEL, 'fixes': {}}
    texts = [squash(x['text']) for x in load_snippets(paper_dir)]
    names, known = prose_names(paper_dir), set(syms)        # the reader's names (as for the printed equations); the model's symbols
    out = {}
    for x in a['fixes']:
        checks = []
        fx = {'action': x['action'], 'printed_latex': by[x['eq']][0]['latex'].strip(), 'corrected_latex': None,
              'reason': x['reason'], 'quote': None, 'basis': 'inferred', 'checks': checks, 'status': 'proposed'}
        q = x.get('quote')
        if q and len(squash(q)) >= 6 and any(squash(q) in t for t in texts):
            fx['quote'], fx['basis'] = q, 'stated'
        if x['action'] == 'correct' and x.get('corrected_latex'):
            tex = re.sub(r'\\\[|\\\]', '', x['corrected_latex']).strip().strip('$').replace('\\displaystyle', '').strip()   # no \[ \] inside $$
            try:
                r = read_equation(f'$$ {tex} $$', names)
                new = _new_symbols(r, read_equation(by[x['eq']][0]['latex'], names), known)
                if new:
                    checks.append(f'the correction brings in symbols the model does not have: {sorted(new)}')
                fx['corrected_latex'] = f'$$ {tex} $$'
            except Exception as e:
                checks.append(f'corrected equation could not be read: {e}')
        if x['action'] == 'correct' and not fx['corrected_latex']:
            fx['action'] = 'keep as printed'
            checks.append('no usable corrected equation: kept as printed')
        out[x['eq']] = fx
    return {'model': llm.MODEL, 'fixes': out}



INDEPENDENT = {'t', 'z', 'x', 'r'}     # time and space coordinates: never in the model's symbol list


def _new_symbols(corr: dict, printed: dict, known: set) -> set:
    """Symbols the correction adds that neither the printed equation nor the model has: a sign the
    proposed fix invents something. Coordinates (t, z, ...) and integration variables do not count."""
    have = set(printed['lhs'] + printed['rhs']) | known | set(corr.get('wrt', [])) | set(printed.get('wrt', [])) | INDEPENDENT
    return set(corr['lhs'] + corr['rhs']) - have


def recheck(paper_dir: str):
    """Redo the code checks on the saved proposals (no LLM call)."""
    path = f'{paper_dir}/fixes.json'
    d = json.load(open(path))
    names, known = prose_names(paper_dir), set(json.load(open(f'{paper_dir}/symbols.json'))['symbols'])
    for f in d['fixes'].values():
        if f.get('action') != 'correct' or not f.get('corrected_latex'):
            continue
        f['checks'] = [c for c in f.get('checks', []) if 'symbols not in the model' not in c and 'symbols the model does not have' not in c]
        try:
            new = _new_symbols(read_equation(f['corrected_latex'], names), read_equation(f['printed_latex'], names), known)
        except Exception as e:
            f['checks'].append(f'corrected equation could not be read: {e}'); continue
        if new:
            f['checks'].append(f'the correction brings in symbols the model does not have: {sorted(new)}')
    json.dump(d, open(path, 'w'), indent=1, ensure_ascii=False)

def main(paper_dir: str, refresh: bool):
    result = propose(paper_dir, refresh)
    print(llm.usage_line())
    if llm.DRY_RUN:
        return
    try:  # hand corrections (basis 'human') and the OCR misreads found in review survive a rerun
        old = json.load(open(f'{paper_dir}/fixes.json'))
        result['fixes'].update({n: f for n, f in old['fixes'].items() if f.get('basis') == 'human'})
        if old.get('misreads'):
            result['misreads'] = old['misreads']
    except (FileNotFoundError, KeyError, ValueError):
        pass
    json.dump(result, open(f'{paper_dir}/fixes.json', 'w'), indent=1, ensure_ascii=False)
    for n, f in result['fixes'].items():
        print(f"eq {n}: {f['action']}" + (f" -> {f['corrected_latex']}" if f['corrected_latex'] else '')
              + (f"  [{'; '.join(f['checks'])}]" if f['checks'] else ''))
        print(f"    {f['reason']}")
    print(f'-> {paper_dir}/fixes.json')


if __name__ == '__main__':
    llm.DRY_RUN = '--dry-run' in sys.argv
    main(sys.argv[1], '--refresh' in sys.argv)
