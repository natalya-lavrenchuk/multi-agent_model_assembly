"""cloud_tasks/design_rescore.py -- scores the design study from design.py's saved outputs. Code only, no LLM.

    python cloud_tasks/design_rescore.py run [--hr]     # -> runs/design_study/rescored.jsonl (resumable)
    python cloud_tasks/design_rescore.py report

Why (FINDINGS.md): design.fit_all (agents/design.py:185) keeps only some keys of each fit and drops 'p' and
'fits_within_noise'. agents._adequate (agents/agents.py:333-337) reads both, so design_study.outcome() raises
KeyError: 'p' (or, with no marginal fit, finds nothing adequate because fits_within_noise is missing). design.main
has already saved every fit set to runs/design_<truth>_seed<n>.json, so this script puts the two keys back with
fit.py's own definitions (p = len(tr['parameters']) = len(theta), fit.py line 410; fits_within_noise = chi2 <= chi2_limit,
fit.py line 480) and applies the unchanged rule (design_study.outcome = agents.adequacy + agents.code_verdict).
No system code is changed and no fit is redone. --hr runs design.main --loop --designs <the D_HR-best design>
(fits come from the fit cache when they exist) and scores it the same way.
"""
from __future__ import annotations
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design_study as ds  # noqa: E402  (chdirs to agents/, puts it on sys.path)

AG = ds.AG
OUT = os.path.join(AG, 'runs', 'design_study', 'rescored.jsonl')


def restore(fits: dict) -> dict:
    out = {}
    for n, f in fits.items():
        f = dict(f)
        if f.get('ok'):
            f['p'] = len(f['theta'])
            f['fits_within_noise'] = f['chi2'] <= f['chi2_limit']
        out[n] = f
    return out


def one(path: str, hr: bool) -> dict:
    m = re.search(r'design_([a-z]+)_seed(\d+)\.json$', path)
    truth, seed = m.group(1), int(m.group(2))
    o = json.load(open(path))
    row = {'truth': truth, 'seed': seed, 'before': ds.outcome(restore(o['before']), truth)}
    for kind in ('designed', 'random (control)'):
        L = o['loop'][kind]
        row[kind] = {'design': L['design'], 'D': L['D'], **ds.outcome(restore(L['after']), truth)}
    row['best_D'], row['best_D_HR'] = o['ranking'][0]['D'], o['ranking'][0]['D_HR']
    if hr:
        import design
        hr_name = max(o['ranking'], key=lambda r: r['D_HR'])['name']
        dpath = os.path.join(AG, 'runs', 'design_study', f'{truth}_seed{seed}.json')
        keep = open(path).read()                         # design.main rewrites this file; keep the --loop output
        try:
            o2 = design.main([dpath, '--rivals', ds.PAIRS[truth], '--loop', '--designs', hr_name])
        finally:
            open(path, 'w').write(keep)
        L = next(v for k, v in o2['loop'].items() if k.startswith('designed'))
        row['hunter-reiner'] = {'design': hr_name, **ds.outcome(restore(L['after']), truth)}
    return row


def run(hr: bool) -> None:
    done = set()
    if os.path.exists(OUT):
        done = {(r['truth'], r['seed']) for r in map(json.loads, open(OUT)) if ('hunter-reiner' in r) or not hr}
    for path in sorted(glob.glob(os.path.join(AG, 'runs', 'design_*_seed*.json'))):
        m = re.search(r'design_([a-z]+)_seed(\d+)\.json$', path)
        if (m.group(1), int(m.group(2))) in done:
            continue
        t0 = time.time()
        try:
            row = one(path, hr)
        except Exception as e:
            row = {'truth': m.group(1), 'seed': int(m.group(2)), 'error': f'{type(e).__name__}: {e}'}
        row['seconds'] = round(time.time() - t0)
        open(OUT, 'a').write(json.dumps(row, default=float) + '\n')
        print(f"{row['truth']:8} seed {row['seed']}: before {row.get('before', {}).get('outcome')}, designed "
              f"{row.get('designed', {}).get('outcome')}, random {row.get('random (control)', {}).get('outcome')}"
              f"{', HR ' + str(row['hunter-reiner']['outcome']) if 'hunter-reiner' in row else ''} {row.get('error', '')}",
              flush=True)


def report() -> None:
    rows = {}
    for r in map(json.loads, open(OUT)):                  # the last line per case wins (an --hr pass adds HR)
        rows[(r['truth'], r['seed'])] = r
    R = list(rows.values())
    bad = [r for r in R if 'error' in r]
    R = [r for r in R if 'error' not in r]
    print(f'{len(R)} cases, {len(bad)} errors (rescored from design.py outputs; see FINDINGS.md)')
    for truth in sorted({r['truth'] for r in R}):
        S = [r for r in R if r['truth'] == truth]
        print(f'\n== truth {truth}: {len(S)} cases, seeds {min(r["seed"] for r in S)}-{max(r["seed"] for r in S)}')
        for kind in ('before', 'designed', 'random (control)', 'hunter-reiner'):
            K = [r[kind]['outcome'] for r in S if kind in r]
            if K:
                n = len(K)
                print(f'  {kind:17} ' + '  '.join(f'{o} {K.count(o)}/{n}' for o in ('right', 'undecided', 'wrong', 'none'))
                      + f"   right {ds.wilson(K.count('right'), n)}")


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == 'run':
        run('--hr' in a)
    elif a and a[0] == 'report':
        report()
    else:
        print(__doc__)
