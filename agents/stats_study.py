"""stats_study.py -- how good is the statistical layer on its own? Code only, no LLM, many noise seeds.

    python stats_study.py run [--seeds 101-130] [--truths monod,contois,andrews,aiba]   # resumable
    python stats_study.py report                                                        # the tables

For every truth and seed: make data from the hidden truth (data.make, a new noise draw), fit the four candidate
laws that matter for these truths (Monod, Contois, Andrews, Aiba) and apply the same fixed rule the agents' code
uses (agents.adequacy / code_verdict: adequate = fits within the noise, every t > 1, no runaway parameter;
decided when one adequate candidate has Akaike weight >= 0.9 among the adequate ones).

What it measures, per truth:
  (fit.WEIGHTS decides the weighting; results go to runs/stats/ for 'model', runs/stats_measured_weights/ for
  'measured', so the two can be compared)
  false rejection   how often the TRUE law is not adequate (chi2 test at 95%: should be about 5%, plus any
                    imprecise-parameter or fitting failures)
  coverage          how often each true parameter value is inside the fitted 95% interval (should be about 95%)
  decided right     the rule picks the true law; decided wrong: it picks another; undecided: the adequate
                    candidates are within 0.9 of each other (indistinguishable); none: nothing adequate
  wrong adequate    how often each WRONG law is adequate (power against it is 1 minus this)
This is the ceiling for the agents: with these candidates fitted, the code alone decides this well. The agents'
own task is harder (they must also find the candidates); the gap is what the search costs.
Seeds start at 101 so they never repeat the batch's data sets (seeds 1-3). Results: runs/stats/<truth>.jsonl.
"""
from __future__ import annotations
import os
import sys
# macOS: the fits run in forked processes (fit.py). After an HTTPS call (urllib loads Apple's system frameworks),
# fork() can abort unless this is set before Python starts, so the script restarts itself once with it set.
if __name__ == '__main__' and sys.platform == 'darwin' and os.environ.get('OBJC_DISABLE_INITIALIZE_FORK_SAFETY') != 'YES':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.execv(sys.executable, [sys.executable] + sys.argv)
import json
import math
import os
import sys
import time

import agents
import candidates
import data as datamod
import fit
import translate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs', 'stats' if fit.WEIGHTS == 'model' else 'stats_measured_weights')
KG = 'http://example.org/kg/'


def rivals(base: dict) -> dict:
    return {'monod': base, 'contois': translate.translate('asprey_2000_contois_kinetics'),
            'andrews': candidates.swap(base, KG + 'zentou_2019_andrews/law/zentou_2019_eq5', {'mu': 'r', 'S_': 'x2'}),
            'aiba': candidates.swap(base, KG + 'muloiwa_2020_free_laws/law/muloiwa_2020_eq11', {'mu': 'r', 'S_': 'x2'})}


def one(truth: str, seed: int, cands: dict) -> dict:
    tr, theta = datamod.scenario(truth)
    d, t = datamod.make(tr, seed, theta)
    rows, log = {}, {'candidates': []}
    for name, c in cands.items():
        f = fit.fit(c, d)
        rows[name] = {k: f.get(k) for k in ('ok', 'chi2', 'chi2_limit', 'fits_within_noise', 'theta', 'ci95',
                                            't_values', 'warnings', 'aic', 'seconds', 'error', 'method')}
        log['candidates'].append({'rate_id': name, 'law': name, 'built': True, 'fit': f})
    ad = agents.adequacy(log)
    rule = agents.code_verdict(ad, 'C' + truth)
    picked = rule['final'][1:] if rule['final'] else ''
    outcome = ('none' if rule['verdict'] == 'none adequate' else 'undecided' if rule['verdict'] == 'indistinguishable'
               else 'right' if picked == truth else 'wrong')
    true_fit = rows[truth]
    cover = {}
    if true_fit['ok'] and true_fit['ci95']:
        true_theta = t['theta']
        for p, (lo, hi) in true_fit['ci95'].items():
            if p in true_theta:
                cover[p] = lo <= true_theta[p] <= hi
    return {'truth': truth, 'seed': seed, 'outcome': outcome, 'picked': picked,
            'adequate': [o['label'][1:] for o in ad], 'weights': {o['label'][1:]: o['weight'] for o in ad},
            'true_adequate': truth in [o['label'][1:] for o in ad], 'coverage': cover, 'fits': rows}


def run(truths: list, seeds: list) -> None:
    os.makedirs(OUT, exist_ok=True)
    base = translate.translate('asprey_2000_monod_kinetics')
    cands = rivals(base)
    for truth in truths:
        path = os.path.join(OUT, f'{truth}.jsonl')
        done = {json.loads(l)['seed'] for l in open(path)} if os.path.exists(path) else set()
        for seed in seeds:
            if seed in done:
                continue
            t0 = time.time()
            r = one(truth, seed, cands)
            open(path, 'a').write(json.dumps(r, default=float) + '\n')
            print(f"{truth:8} seed {seed}: {r['outcome']:9} adequate {r['adequate']}  "
                  f"{time.time() - t0:.0f} s", flush=True)


def wilson(k: int, n: int) -> tuple:
    if n == 0:
        return (float('nan'), float('nan'))
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def report() -> None:
    import glob
    for path in sorted(glob.glob(os.path.join(OUT, '*.jsonl'))):
        R = [json.loads(l) for l in open(path)]
        n, truth = len(R), R[0]['truth']
        print(f"\n== truth {truth}: {n} seeds")
        fr = sum(not r['true_adequate'] for r in R)
        lo, hi = wilson(fr, n)
        print(f"  false rejection of the true law: {fr}/{n} = {fr / n:.0%}  (95% CI {lo:.0%}-{hi:.0%}; expected ~5%)")
        for o in ('right', 'undecided', 'wrong', 'none'):
            k = sum(r['outcome'] == o for r in R)
            print(f"  rule outcome {o:10} {k}/{n} = {k / n:.0%}")
        for other in ('monod', 'contois', 'andrews', 'aiba'):
            if other != truth:
                k = sum(other in r['adequate'] for r in R)
                print(f"  wrong law {other:8} adequate {k}/{n} = {k / n:.0%}  (power against it {1 - k / n:.0%})")
        params = sorted({p for r in R for p in r['coverage']})
        for p in params:
            v = [r['coverage'][p] for r in R if p in r['coverage']]
            print(f"  95% interval covers the true {p:7}: {sum(v)}/{len(v)} = {sum(v) / len(v):.0%}")
        w = [r['weights'].get(truth) for r in R if truth in r['weights']]
        if w:
            print(f"  mean Akaike weight of the true law among adequate: {sum(w) / len(w):.2f}")


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == 'run':
        s = args[args.index('--seeds') + 1] if '--seeds' in args else '101-130'
        a, b = (int(x) for x in s.split('-'))
        truths = args[args.index('--truths') + 1].split(',') if '--truths' in args else ['andrews', 'aiba', 'monod', 'contois']
        run(truths, list(range(a, b + 1)))
    elif args and args[0] == 'report':
        report()
    else:
        print(__doc__)
