"""score_readers.py -- score equation readers (Marker, Paddle, ...) against gold_laws.json.

Compares at the SymPy level, not the string level: two LaTeX strings that typeset the
same equation ('F_{o u t}' vs 'F_{out}') count as equal; a different symbol does not.
Readers are matched to gold rows by reading order, not by the equation number they report.

Usage (from extractor_agent/):
  python -m stages.score_readers papers/kontoravdi_2010
"""
import json, sys
import sympy as sp
from stages.latex_to_sympy import parse

def canon(tex):
    try:
        eq, _ = parse(tex)
        return sp.srepr(eq)
    except Exception:
        return None

def main(paper_dir):
    gold = json.load(open(f'{paper_dir}/gold_laws.json'))['laws']
    readers = {'marker': [g['latex_marker'] for g in gold]}
    try:
        readers['paddle'] = [f['latex'] for f in json.load(open(f'{paper_dir}/paddle.json'))['formulas']]
    except FileNotFoundError:
        pass
    names = list(readers)
    score = dict.fromkeys(names, 0)
    print(f"{'eq':>3} | " + ' | '.join(f'{n:7}' for n in names) + ' | readers agree?')
    for i, g in enumerate(gold):
        ref = canon(g['latex_printed'])
        got = [canon(readers[n][i]) if i < len(readers[n]) else None for n in names]
        cells = []
        for n, c in zip(names, got):
            ok = c is not None and c == ref
            score[n] += ok
            cells.append('ok' if ok else ('FAIL' if c is None else 'WRONG'))
        agree = len(set(got)) == 1 and got[0] is not None
        print(f"{g['eq']:>3} | " + ' | '.join(f'{c:7}' for c in cells) + f" | {'yes' if agree else 'NO -> flag'}")
    print(' '.join(f'{n}: {score[n]}/{len(gold)}' for n in names))

if __name__ == '__main__':
    main(sys.argv[1])
