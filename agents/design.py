"""design.py -- experiment design for model discrimination. Code only, no LLM.

    python design.py runs/aiba_seed2.json --rivals andrews,aiba            # rank new experiments for two rivals
    python design.py runs/aiba_seed2.json --rivals andrews,aiba --loop     # + closed loop on the hidden truth
    python design.py runs/aiba_seed2.json --from-log runs/batch/agents_aiba_seed2_confirm.json --loop

When two (or more) rival laws both fit the data, which NEW experiment would tell them apart best?
  1. Each rival is fitted to the data (fit.fit: parmest, same weights and tests as everywhere else).
  2. Candidate experiments: one fed-batch experiment of 20 h, sampled hourly, inside the ranges of the case study
     (Asprey and Macchietto 2000): dilution u1 in 0.05-0.2 1/h, feed concentration u2 in 5-35 g/L, starting
     biomass 1-10 g/L; feed held constant or switched once at 10 h. 320 designs.
  3. For each design, both rivals are simulated with their fitted parameters, and the design is scored by the
     Hunter-Reiner criterion: D = sum over readings of ((y_A - y_B) / sigma)^2, sigma the measurement noise
     (data.sd) at the mean of the two predictions, plus (Buzzi-Ferraris and Forzatti) the variance of each
     rival's prediction from its own parameter uncertainty (sensitivities and the parameter covariance on the
     data so far). D is roughly the chi-square a NEW experiment would add against the wrong rival if the other
     were true: D >> 1 separates them, D ~ 1 cannot. Without the parameter term (Hunter-Reiner, also reported),
     the ranking favoured knife-edge designs (4 Nov: substrate running out at an uncertain moment), on which even
     the TRUE law fitted badly.
     With more than two rivals, a design scores its smallest D over the pairs (every pair must be separated).
  4. --loop (synthetic data only): the best design is "run" on the hidden truth (simulated, with noise drawn as
     in data.py), added to the data, and the rivals are refitted; the same is done for a design chosen at random
     from the grid, as the control. Reported: chi-square, p-value and Akaike weight of each rival before and
     after. The question for the thesis: does a designed experiment turn "indistinguishable" into a decision, and
     does it do better than an experiment chosen without design?
Rivals: --rivals names from RIVALS below, or --from-log a run log, whose adequate candidates (confirmation
round) are taken as the rivals, rebuilt from their law and mapping. Results: runs/design_<data>.json.
Limits: designs are fixed-shape (constant or one switch), not optimised continuously; the prediction variance is
linearised (sensitivities), so far from the current estimates it is only approximate.
"""
from __future__ import annotations
import os
import sys
# macOS: the fits run in forked processes; see agents.py
if __name__ == '__main__' and sys.platform == 'darwin' and os.environ.get('OBJC_DISABLE_INITIALIZE_FORK_SAFETY') != 'YES':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.execv(sys.executable, [sys.executable] + sys.argv)
import copy
import itertools
import json
import math

import numpy as np

import build
import candidates
import data as datamod
import fit
import translate

HERE = os.path.dirname(os.path.abspath(__file__))
KG = 'http://example.org/kg/'
RIVALS = {
    'monod': None, 'contois': 'contois',
    'andrews': (KG + 'zentou_2019_andrews/law/zentou_2019_eq5', {'mu': 'r', 'S_': 'x2'}),
    'aiba': (KG + 'muloiwa_2020_free_laws/law/muloiwa_2020_eq11', {'mu': 'r', 'S_': 'x2'}),
    'moser': (KG + 'muloiwa_2020_free_laws/law/muloiwa_2020_eq8', {'mu': 'r', 'S_': 'x2'}),
}
U1 = [0.05, 0.1, 0.15, 0.2]
U2 = [5.0, 10.0, 20.0, 35.0]
X10 = [1.0, 3.0, 5.0, 8.0, 10.0]
TF, SWITCH = 20, 10


def rival(name: str, base: dict) -> dict:
    spec = RIVALS[name]
    if spec is None:
        return base
    if spec == 'contois':
        return translate.translate('asprey_2000_contois_kinetics')
    return candidates.swap(base, spec[0], spec[1])


def designs() -> list:
    """4 dilution rates x 5 starting biomasses x 16 feed profiles (4 constant, 12 switched once at 10 h) = 320
    designs, named D001..."""
    out = []
    for u1, x10 in itertools.product(U1, X10):
        for a, b in itertools.product(U2, U2):
            steps = [(0, a)] if a == b else [(0, a), (SWITCH, b)]
            out.append({'tf': TF, 'samples': list(range(TF + 1)), 'x0': {'x1': x10, 'x2': 0.01},
                        'inputs': {'u1': [(0, u1)], 'u2': steps}})
    for i, e in enumerate(out, 1):
        e['name'] = f'D{i:03d}'
    return out


def describe(e: dict) -> str:
    u2 = e['inputs']['u2']
    feed = f"u2 {u2[0][1]:g} g/L" + (f", switched to {u2[1][1]:g} g/L at {u2[1][0]} h" if len(u2) > 1 else '')
    return f"start biomass {e['x0']['x1']:g} g/L, dilution u1 {e['inputs']['u1'][0][1]:g} 1/h, feed {feed}, {e['tf']} h hourly"


def simulate(tr: dict, e: dict, theta: dict) -> dict | None:
    sim = build.integrate(tr, e, theta)
    if sim is None:
        return None
    return {s: np.array([sim(t)[0][s] for t in e['samples']]) for s in ('x1', 'x2')}


REL_STEP = 1e-3                                           # finite-difference step for sensitivities


def _stack(pred: dict) -> np.ndarray:
    return np.concatenate([pred['x1'], pred['x2']])


def sensitivities(tr: dict, e: dict, theta: dict) -> tuple:
    """Predictions (x1 and x2 stacked) and d prediction / d theta (finite differences), or (None, None)."""
    p0 = simulate(tr, e, theta)
    if p0 is None:
        return None, None
    y0, cols = _stack(p0), []
    for k in theta:
        th = dict(theta)
        h = REL_STEP * max(abs(theta[k]), 1e-6)
        th[k] = theta[k] + h
        p1 = simulate(tr, e, th)
        if p1 is None:
            return y0, None
        cols.append((_stack(p1) - y0) / h)
    return y0, np.column_stack(cols)


def covariance(tr: dict, data: dict, theta: dict) -> np.ndarray | None:
    """Parameter covariance (J'WJ)^-1 on the data the rival was fitted to (W = 1/sigma^2, sigma from the model
    curve, as in fit.py). Computed here so the design code needs nothing from the fit beyond theta."""
    rows = []
    for e in data['experiments']:
        y0, S = sensitivities(tr, e, theta)
        if S is None:
            return None
        rows.append(S / datamod.sd(np.clip(y0, 0, None))[:, None])
    J = np.vstack(rows)
    try:
        return np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(J.T @ J)


def divergence(models: list, e: dict) -> tuple:
    """models: [(label, tr, theta, cov)]. Per pair of rivals:
       D_HR = sum ((y_A - y_B) / sigma)^2                         (Hunter and Reiner 1965: noise only)
       D_BF = sum (y_A - y_B)^2 / (sigma^2 + var_A + var_B)       (Buzzi-Ferraris and Forzatti: + the spread of
              each rival's prediction from its own parameter uncertainty, var = diag(S C S'), S = sensitivities)
    A design that separates the rivals only where their predictions are themselves uncertain (a knife-edge, such
    as the moment substrate runs out) scores high on D_HR but not on D_BF. Ranking uses D_BF (min over pairs)."""
    pr = {}
    for lab, tr, th, C in models:
        y, S = sensitivities(tr, e, th)
        if y is None or S is None or C is None:
            return 0.0, 0.0, {}
        pr[lab] = (y, np.einsum('ij,jk,ik->i', S, C, S))
    per, hr = {}, {}
    for (a, (ya, va)), (b, (yb, vb)) in itertools.combinations(pr.items(), 2):
        sig2 = datamod.sd(np.clip((ya + yb) / 2, 0, None)) ** 2
        hr[f'{a} vs {b}'] = float(np.sum((ya - yb) ** 2 / sig2))
        per[f'{a} vs {b}'] = float(np.sum((ya - yb) ** 2 / (sig2 + np.clip(va, 0, None) + np.clip(vb, 0, None))))
    return min(per.values()), min(hr.values()), per


def rank(models: list) -> list:
    rows = []
    for e in designs():
        D, D_hr, per = divergence(models, e)
        rows.append({'design': e, 'D': D, 'D_HR': D_hr, 'per_pair': per, 'text': describe(e)})
    return sorted(rows, key=lambda r: -r['D'])


def run_experiment(truth_tr: dict, theta: dict, e: dict, seed: int) -> dict:
    """The new experiment 'run' on the hidden truth: simulated finely, noise as in data.make."""
    rng = np.random.default_rng(seed)
    m = build.build(truth_tr, e, theta, nfe_per_hour=8)
    if not build.solve(m):
        raise RuntimeError('the true model did not solve for the designed experiment')
    clean = build.trajectories(m, e['samples'])
    y = {s: [max(0.0, float(v)) for v in np.asarray(clean[s]) + rng.normal(0, 1, len(clean[s])) * datamod.sd(clean[s])]
         for s in datamod.MEASURED}
    return {**copy.deepcopy(e), 'y': y}


def fit_all(rivals: list, data: dict) -> dict:
    fits = [dict(fit.fit(tr, data), sup=lab) for lab, tr in rivals]
    fit.compare(fits)
    return {f['sup']: {k: f.get(k) for k in ('ok', 'chi2', 'chi2_limit', 'p_value', 'fit_class', 'theta', 't_values',
                                             'aic', 'akaike_weight', 'error', 'warnings')} for f in fits}


def show(title: str, fits: dict) -> None:
    print(f'  {title}')
    for lab, f in fits.items():
        if f['ok']:
            print(f"    {lab:10} chi2 {f['chi2']:7.1f} / limit {f['chi2_limit']:.1f}  p {f['p_value']:.3f} "
                  f"({f['fit_class']})  Akaike weight {f.get('akaike_weight', 0):.3f}")
        else:
            print(f"    {lab:10} fit failed: {(f.get('error') or '')[:110]}")


def rivals_from_log(path: str, base: dict) -> list:
    log = json.load(open(path))
    conf = log.get('confirmation') or {}
    labels = [o['label'] for o in conf.get('adequate', [])]
    by = {f"C{c['rate_id']}": c for c in log['candidates'] if c.get('built')}
    out = []
    for lab in labels:
        c = by[lab]
        tr = base if c['iri'].endswith('asprey_2000_eq17') else (
            translate.translate('asprey_2000_contois_kinetics') if c['iri'].endswith('asprey_2000_eq18')
            else candidates.swap(base, c['iri'], c['mapping']))
        out.append((f"{lab} {c['law'].split(' eq ')[0][:18]} eq {c['law'].split(' eq ')[-1]}", tr))
    return out


def main(args: list) -> dict:
    data_path = args[0]
    data = json.load(open(data_path))
    base = translate.translate('asprey_2000_monod_kinetics')
    if '--from-log' in args:
        rivals = rivals_from_log(args[args.index('--from-log') + 1], base)
    else:
        rivals = [(n, rival(n, base)) for n in args[args.index('--rivals') + 1].split(',')]
    if len(rivals) < 2:
        raise SystemExit('need at least two rivals')
    print(f'rivals: {", ".join(l for l, _ in rivals)}')
    before = fit_all(rivals, data)
    show('fitted to the data as it is:', before)
    models = [(lab, tr, before[lab]['theta'], covariance(tr, data, before[lab]['theta'])) for lab, tr in rivals
              if before[lab]['ok']]
    ranked = rank(models)
    print(f'\n  best experiments of {len(ranked)} (D = expected chi-square separating the rivals, counting noise and '
          f'parameter uncertainty; >> 1 separates; D_HR counts noise only):')
    for r in ranked[:5]:
        print(f"    {r['design']['name']}  D {r['D']:8.1f}  (D_HR {r['D_HR']:8.1f})   {r['text']}")
    by_hr = sorted(ranked, key=lambda r: -r['D_HR'])[:3]
    print('    best by noise only (D_HR): ' + '; '.join(f"{r['design']['name']} D_HR {r['D_HR']:.0f} but D {r['D']:.1f}"
                                                   for r in by_hr))
    Ds = np.array([r['D'] for r in ranked])
    existing = [divergence(models, e)[0] for e in data['experiments']]
    print(f'    (median design D {np.median(Ds):.1f}; the existing three experiments: '
          f"{', '.join(f'{x:.1f}' for x in existing)})")
    out = {'data': os.path.basename(data_path), 'rivals': [l for l, _ in rivals], 'before': before,
           'ranking': [{'name': r['design']['name'], 'D': r['D'], 'D_HR': r['D_HR'], 'text': r['text'],
                        'per_pair': r['per_pair']} for r in ranked], 'existing_D': existing}
    if '--loop' in args:
        truth_file = data_path.replace('.json', '.truth.json')
        truth = json.load(open(truth_file))                     # synthetic data only: the hidden truth runs it
        truth_tr, theta = datamod.scenario(truth['scenario'])
        theta = truth['theta'] or theta or build.paper_values(truth_tr)
        rng = np.random.default_rng(1000 + int(data.get('seed', 0)))
        random_design = ranked[int(rng.integers(len(ranked)))]
        out['loop'] = {'truth': truth['scenario']}
        tries = [('designed', ranked[0]), ('random (control)', random_design)]
        if '--designs' in args:                                  # chosen designs instead (e.g. the next best ones)
            names = args[args.index('--designs') + 1].split(',')
            tries = [(f'designed ({n})', next(r for r in ranked if r['design']['name'] == n)) for n in names]
        for kind, r in tries:
            new = run_experiment(truth_tr, theta, {**r['design'], 'name': 'E4'}, seed=10_000 + int(data.get('seed', 0)))
            d2 = {**data, 'experiments': data['experiments'] + [new]}
            after = fit_all(rivals, d2)
            print(f"\n  after adding the {kind} experiment {r['design']['name']} (D {r['D']:.1f}): {r['text']}")
            show('refitted on 4 experiments:', after)
            out['loop'][kind] = {'design': r['design']['name'], 'D': r['D'], 'text': r['text'], 'after': after}
        print(f"\n  hidden truth: {truth['scenario']}")
    tag = '_' + args[args.index('--designs') + 1].replace(',', '_') if '--designs' in args else ''
    path = os.path.join(HERE, 'runs', f"design_{os.path.basename(data_path).replace('.json', '')}{tag}.json")
    json.dump(out, open(path, 'w'), indent=1, default=float)
    print('wrote', path)
    return out


if __name__ == '__main__':
    main(sys.argv[1:])
