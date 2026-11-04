"""structure_check.py -- model-level checks on a set of laws from ONE Sup. Code only, no LLM.

What it does
  1. Assigns each law the ONE variable it computes.
       direct       the left side has one candidate   (Q_glc = ...  ->  Q_glc)
       elimination  the left side has several; drop the ones another law already
                    computes, repeat until settled    (d(V X_v)/dt -> X_v, because eq 1 computes V)
       ambiguous    still more than one candidate -> flag
  2. Reports what the assignment implies. How a finding is READ depends on the Sup kind:

     finding                         full_model                       law_collection
     ------------------------------  -------------------------------  ---------------------------
     variable computed by 2+ laws    ERROR (over-determined)          rivals (expected)
     used but never computed         inputs/parameters, or a misread  inputs/parameters (expected)
     computed but never used         model output, or a misread       expected
     near-name pair (see below)      likely misread                   likely misread
     square (every law has its own   expected; if not, flag           not expected
     variable)

  near-name pair: X is computed but never used, Y is used but never computed, and
  their names differ by one subscript/letter (R_t vs R_L). In a full model that
  pattern is the fingerprint of an OCR misread on one side.

What it cannot do
  - Tell parameters from inputs (needs the variable step and the prose).
  - Decide whether 'K I_lac' is one name or K times I_lac (needs the prose definition).
  - Check dimensions (separate step, needs units).

Usage (from extractor_agent/):
  python -m stages.structure_check papers/kontoravdi_2010/gold_laws.json --field latex_printed
  python -m stages.structure_check papers/kontoravdi_2010/paddle.json --kind full_model
"""
from __future__ import annotations
import argparse, difflib, json
import sympy as sp
from stages.latex_to_sympy import parse

T = sp.Symbol('t')

def candidates(eq):
    lhs = eq.lhs
    if isinstance(lhs, sp.Derivative):
        lhs = lhs.expr
    return sorted(lhs.free_symbols - {T}, key=str)

def assign(laws):
    """laws: {label: Eq}. Returns {label: (symbol or None, how, candidates)}."""
    cand = {k: candidates(e) for k, e in laws.items()}
    out = {k: (c[0], 'direct', c) for k, c in cand.items() if len(c) == 1}
    changed = True
    while changed:
        changed = False
        taken = {v[0] for v in out.values()}
        for k, c in cand.items():
            if k in out:
                continue
            left = [s for s in c if s not in taken]
            if len(left) == 1:
                out[k] = (left[0], 'elimination', c); changed = True
    for k, c in cand.items():
        if k not in out:
            out[k] = (None, 'ambiguous' if c else 'no candidate', c)
    return out

def near(a, b):
    a, b = str(a), str(b)
    return a != b and difflib.SequenceMatcher(None, a, b).ratio() >= 0.6

def check(rows, kind):
    laws, failed = {}, []
    for label, tex in rows:
        try:
            laws[label], _ = parse(tex)
        except Exception as ex:
            failed.append((label, str(ex)[:70]))
    asg = assign(laws)
    computed = {}
    for k, (s, _, _) in asg.items():
        if s is not None:
            computed.setdefault(s, []).append(k)
    used = {}
    for k, e in laws.items():
        for s in e.free_symbols - {T}:
            if s != asg[k][0]:
                used.setdefault(s, []).append(k)

    multi = {s: ks for s, ks in computed.items() if len(ks) > 1}
    never_computed = sorted((s for s in used if s not in computed), key=str)
    never_used = sorted((s for s in computed if s not in used), key=str)
    pairs = [(x, y) for x in never_used for y in never_computed if near(x, y)]
    square = not failed and not multi and all(v[0] is not None for v in asg.values())

    full = kind == 'full_model'
    print(f'Sup kind: {kind}   laws: {len(rows)}   parsed: {len(laws)}')
    if failed:
        print('\nPARSE FAILED (reader or converter problem, see reader agreement):')
        for k, m in failed: print(f'  eq {k}: {m}')
    print('\nComputes:')
    for k, (s, how, c) in asg.items():
        extra = f'   candidates {[str(x) for x in c]}' if how != 'direct' else ''
        print(f'  eq {k:>3} -> {str(s):14} [{how}]{extra}')
    print(f"\nComputed by 2+ laws ({'ERROR' if full else 'rivals'}):", {str(s): ks for s, ks in multi.items()} or 'none')
    print(f"Computed, never used ({'output or misread' if full else 'expected'}):", [str(s) for s in never_used] or 'none')
    print(f"Used, never computed ({'inputs/parameters, or misread' if full else 'expected'}): {len(never_computed)}")
    print('  ', ' '.join(str(s) for s in never_computed))
    print('Near-name pairs (computed-not-used ~ used-not-computed, likely misread):',
          [f'{x} ~ {y}' for x, y in pairs] or 'none')
    if full:
        print('Square (each law has its own variable):', 'yes' if square else 'NO')
    return dict(assign=asg, multi=multi, never_used=never_used, never_computed=never_computed,
                pairs=pairs, failed=failed, square=square)

def load_rows(path, field):
    d = json.load(open(path))
    if 'laws' in d:                                    # gold_laws.json
        return [(g['eq'], g[field]) for g in d['laws']]
    return [(i, f['latex']) for i, f in enumerate(d['formulas'], 1)]   # reader output: reading order

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--field', default='latex_printed', help='for gold_laws.json: latex_printed or latex_marker')
    ap.add_argument('--kind', default='full_model', choices=['full_model', 'law_collection'])
    a = ap.parse_args()
    check(load_rows(a.path, a.field), a.kind)
