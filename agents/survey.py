"""survey.py -- Layer 1 over every rate law in the graph, put into Asprey's balances. Code only, except the
LaTeX -> SymPy translation (one cached LLM call per law, the same as the agents use).

    python survey.py --dry-run       # how many translations are not cached yet, nothing sent
    python survey.py                 # writes runs/survey_rate_laws.json and prints the table

The symbol mapping is done by code from the graph's roles, not by the modeler agent, using the problem record
(problem.py): the law's computed quantity -> the unknown (r), and each role the record lists -> its model symbol
(for the yeast case 'substrate' -> x2, 'biomass' -> x1). A law that needs any other variable (an inhibitor
such as lactate, oxygen, a product) cannot be used with Asprey's two measured states, and is reported so.
Then the same checks as for any candidate: units, structure, positivity (limiting behaviour).
For the thesis: how many of the graph's rate laws are usable for this case study, and why the rest are not.
"""
from __future__ import annotations
import json
import os
import sys
from collections import Counter

import candidates
import kg_query
import translate
from extractor import llm

import problem


def survey(base: dict, p: dict) -> list:
    """Every law in the graph that computes something of the unknown's dimension, mapped by the graph's roles
    (p['roles']: role -> model symbol) and checked, for the problem p."""
    rows = []
    for i, r in enumerate(kg_query.find_rate_laws(p['unknown_dimension'])):
        row = {'id': i, 'law': f"{r['source']} eq {r['eq']}", 'iri': r['iri'], 'latex': r['latex']}
        law_tr = translate.translate_model(kg_query.get_law(r['iri']))
        L = law_tr['laws'][0] if law_tr.get('laws') else {}
        if 'rhs' not in L:
            row.update(status='not translated', reason='; '.join(L.get('problems', [])) or 'no translation (dry run)')
            rows.append(row)
            continue
        if L['problems']:
            row.update(status='translation rejected', reason='; '.join(L['problems']))
            rows.append(row)
            continue
        names = law_tr['names']
        mapping = {L['target']: p['unknown']}
        for role, sym in r['roles'].items():
            if role in p['roles'] and sym in names:
                mapping[names[sym]] = p['roles'][role]
        variables = {str(s) for s in L['rhs'].free_symbols} - set(law_tr['parameters']) - {'t'}
        missing = sorted(v for v in variables if v not in mapping)
        if missing:
            inv = {v: k for k, v in names.items()}
            what = [f"{v} ({law_tr['symbols'][inv[v]].get('description', '')[:40]})" for v in missing]
            row.update(status='needs other variables', reason='needs ' + ', '.join(what))
            rows.append(row)
            continue
        cand = candidates.swap(base, r['iri'], mapping, law_tr)
        row['mapping'] = mapping
        row['rate_law'] = cand['laws'][-1]['rhs_text'] if cand.get('laws') else ''
        if cand['ok']:
            row.update(status='usable', reason='')
        else:
            kind = ('positivity' if any('below zero' in p for p in cand['problems']) else
                    'units' if any('unit' in p or 'dimension' in p for p in cand['problems']) else 'structure')
            row.update(status=f'rejected: {kind}', reason='; '.join(cand['problems']))
        rows.append(row)
    return rows


if __name__ == '__main__':
    llm.DRY_RUN = '--dry-run' in sys.argv
    # python survey.py [data file]: the problem record of that data file (default: the yeast case)
    files = [a for a in sys.argv[1:] if a.endswith('.json')]
    p = problem.of(json.load(open(files[0])) if files else {})
    base = translate.translate(p['base_model'])
    rows = survey(base, p)
    for row in rows:
        print(f"[{row['id']:2}] {row['law'][:36]:36} {row['status']:24} {row['reason'][:110]}")
    print('\n', dict(Counter(r['status'] for r in rows)))
    print(llm.usage_line())
    if not llm.DRY_RUN:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'survey_rate_laws.json')
        json.dump(rows, open(out, 'w'), indent=1, ensure_ascii=False)
        print('wrote', out)
