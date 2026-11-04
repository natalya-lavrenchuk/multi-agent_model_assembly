"""cloud_tasks/design_study.py -- design.py's closed loop over many noise seeds. Code only, no LLM.

    python cloud_tasks/design_study.py run [--seeds 101-112] [--truths monod,contois,andrews,aiba] [--hr]
    python cloud_tasks/design_study.py report

For every truth and seed: make data from the hidden truth (data.make, the same generator as the batch and
stats_study), then run design.main(--loop) with the rival pair for that truth (Monod vs Contois for monod and
contois, Andrews vs Aiba for andrews and aiba). design.main fits both rivals, ranks the 320 designs by the
Buzzi-Ferraris criterion D, "runs" the best one and a random one (the control) on the hidden truth, and refits.
Here every fit set (before, after designed, after random) goes through the same fixed rule as stats_study
(agents.adequacy + agents.code_verdict), giving outcome right / undecided / wrong / none.
--hr: also run the design that the noise-only Hunter-Reiner criterion D_HR ranks first (for the HR vs BF table).
Resumable: a case whose design file exists is skipped. Data: runs/design_study/; design.py's own output stays at
runs/design_<truth>_seed<n>.json; one summary line per case in runs/design_study/summary.jsonl.
"""
from __future__ import annotations
import json
import math
import os
import sys
import time

AG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AG)
os.chdir(AG)
import agents             # noqa: E402
import data as datamod    # noqa: E402
import design             # noqa: E402

OUT = os.path.join(AG, 'runs', 'design_study')
PAIRS = {'monod': 'monod,contois', 'contois': 'monod,contois', 'andrews': 'andrews,aiba', 'aiba': 'andrews,aiba'}


def outcome(fits: dict, truth: str) -> dict:
    log = {'candidates': [{'rate_id': n, 'law': n, 'built': True, 'fit': f} for n, f in fits.items()]}
    ad = agents.adequacy(log)
    rule = agents.code_verdict(ad, 'C' + truth)
    picked = rule['final'][1:] if rule['final'] else ''
    o = ('none' if rule['verdict'] == 'none adequate' else 'undecided' if rule['verdict'] == 'indistinguishable'
         else 'right' if picked == truth else 'wrong')
    return {'outcome': o, 'picked': picked, 'weights': {a['label'][1:]: a['weight'] for a in ad},
            'chi2': {n: (f.get('chi2'), f.get('chi2_limit')) for n, f in fits.items()}}


def one(truth: str, seed: int, hr: bool) -> dict:
    dpath = os.path.join(OUT, f'{truth}_seed{seed}.json')
    if not os.path.exists(dpath):
        tr, theta = datamod.scenario(truth)
        d, t = datamod.make(tr, seed, theta)
        t['scenario'] = truth
        json.dump(d, open(dpath, 'w'), indent=1)
        json.dump(t, open(dpath.replace('.json', '.truth.json'), 'w'), indent=1)
    out = design.main([dpath, '--rivals', PAIRS[truth], '--loop'])
    row = {'truth': truth, 'seed': seed, 'before': outcome(out['before'], truth)}
    for kind in ('designed', 'random (control)'):
        L = out['loop'][kind]
        row[kind] = {'design': L['design'], 'D': L['D'], **outcome(L['after'], truth)}
    top = out['ranking'][0]
    row['best_D'], row['best_D_HR'] = top['D'], top['D_HR']
    if hr:
        hr_name = max(out['ranking'], key=lambda r: r['D_HR'])['name']
        o2 = design.main([dpath, '--rivals', PAIRS[truth], '--loop', '--designs', hr_name])
        L = next(v for k, v in o2['loop'].items() if k.startswith('designed'))
        row['hunter-reiner'] = {'design': hr_name, **outcome(L['after'], truth)}
    return row


def run(truths: list, seeds: list, hr: bool) -> None:
    os.makedirs(OUT, exist_ok=True)
    summ = os.path.join(OUT, 'summary.jsonl')
    done = {(r['truth'], r['seed']) for r in map(json.loads, open(summ))} if os.path.exists(summ) else set()
    for seed in seeds:                      # seed-major: every truth gets cases even if the night ends early
        for truth in truths:
            if (truth, seed) in done:
                continue
            t0 = time.time()
            try:
                row = one(truth, seed, hr)
            except Exception as e:          # one bad case must not stop the night: record it, carry on
                row = {'truth': truth, 'seed': seed, 'error': f'{type(e).__name__}: {e}'}
            row['seconds'] = round(time.time() - t0)
            open(summ, 'a').write(json.dumps(row, default=float) + '\n')
            print(f"{truth:8} seed {seed}: before {row.get('before', {}).get('outcome')}, designed "
                  f"{row.get('designed', {}).get('outcome')}, random {row.get('random (control)', {}).get('outcome')}"
                  f"  {row['seconds']} s {row.get('error', '')}", flush=True)


def wilson(k: int, n: int) -> str:
    if n == 0:
        return ''
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f'(95% CI {c - h:.0%}-{c + h:.0%})'


def report() -> None:
    R = [json.loads(l) for l in open(os.path.join(OUT, 'summary.jsonl'))]
    bad = [r for r in R if 'error' in r]
    R = [r for r in R if 'error' not in r]
    print(f'{len(R)} cases, {len(bad)} errors')
    for truth in sorted({r['truth'] for r in R}):
        S = [r for r in R if r['truth'] == truth]
        print(f'\n== truth {truth}: {len(S)} cases')
        for kind in ('before', 'designed', 'random (control)', 'hunter-reiner'):
            K = [r[kind]['outcome'] for r in S if kind in r]
            if K:
                n = len(K)
                print(f'  {kind:17} ' + '  '.join(f'{o} {K.count(o)}/{n}' for o in ('right', 'undecided', 'wrong', 'none'))
                      + f"   right {wilson(K.count('right'), n)}")


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == 'run':
        s = a[a.index('--seeds') + 1] if '--seeds' in a else '101-112'
        lo, hi = (int(x) for x in s.split('-'))
        truths = a[a.index('--truths') + 1].split(',') if '--truths' in a else ['andrews', 'aiba', 'monod', 'contois']
        run(truths, list(range(lo, hi + 1)), '--hr' in a)
    elif a and a[0] == 'report':
        report()
    else:
        print(__doc__)
