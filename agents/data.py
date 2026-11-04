"""data.py -- synthetic experiments from a hidden "true" model. Code only, no LLM.

    python data.py andrews --seed 1      # writes runs/andrews_seed1.json (agents) and runs/andrews_seed1.truth.json

The true model is a Sup from the graph, built exactly like any candidate (translate + build, finer time grid),
with the paper's parameter values. The agents get only the data file: measured outputs, operating conditions
and noise level. Which model and which values made it is in the truth file, read only by the scoring code.

Noise: Gaussian, standard deviation REL_SD * value + ABS_SD (5% plus a small floor, so near-zero substrate
readings are not trusted to 0.1%). The same formula, from the measured value, is what the fitting uses.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

import build
import translate

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
REL_SD, ABS_SD = 0.05, 0.01                                     # 5% of the reading + 0.01 g/L
MEASURED = ['x1', 'x2']                                         # biomass and substrate, sampled every hour

# Three 20 h fed-batch experiments inside the paper's ranges (dilution 0.05-0.2 1/h, feed 5-35 g/L,
# starting biomass 1-10 g/L). Experiment 1 is the paper's first one, extended to 20 h.
EXPERIMENTS = [
    {'name': 'E1', 'tf': 20, 'samples': list(range(21)), 'x0': {'x1': 5.0, 'x2': 0.01},
     'inputs': {'u1': [(0, 0.2)], 'u2': [(0, 15.0)]}},
    {'name': 'E2', 'tf': 20, 'samples': list(range(21)), 'x0': {'x1': 2.0, 'x2': 0.01},
     'inputs': {'u1': [(0, 0.05), (10, 0.15)], 'u2': [(0, 35.0), (10, 10.0)]}},
    {'name': 'E3', 'tf': 20, 'samples': list(range(21)), 'x0': {'x1': 8.0, 'x2': 0.01},
     'inputs': {'u1': [(0, 0.15), (7, 0.08), (14, 0.2)], 'u2': [(0, 5.0), (7, 25.0)]}},
]


DESCRIPTION = (
    "Baker's yeast (Saccharomyces cerevisiae) grown in a fed-batch bioreactor. Biomass concentration x1 and "
    "substrate (glucose) concentration x2 are measured every hour, in g/L. The operator sets the dilution rate u1 "
    "(1/h) and the substrate concentration in the feed u2 (g/L), both held constant between switches. Three "
    "experiments of 20 h with different starting biomass and input profiles. Measurement error is about 5% of "
    "the reading plus 0.01 g/L. The growth kinetics are unknown: find the rate law that explains the data.")


def sd(values) -> np.ndarray:
    return REL_SD * np.abs(np.asarray(values, float)) + ABS_SD


def make(true, seed: int, theta: dict | None = None) -> tuple[dict, dict]:
    """true: a Sup id, or a model dict (translate.translate or candidates.swap output)."""
    tr = translate.translate(true) if isinstance(true, str) else true
    if not tr['ok']:
        raise ValueError(f"{tr['sup']} does not pass the checks")
    theta = theta or build.paper_values(tr)
    rng = np.random.default_rng(seed)
    exps = []
    for e in EXPERIMENTS:
        m = build.build(tr, e, theta, nfe_per_hour=8)
        if not build.solve(m):
            raise RuntimeError(f"true model did not solve for {e['name']}")
        clean = build.trajectories(m, e['samples'])
        y = {s: [max(0.0, float(v)) for v in np.asarray(clean[s]) + rng.normal(0, 1, len(clean[s])) * sd(clean[s])]
             for s in MEASURED}
        exps.append({**e, 'y': y})
    import problem
    data = {'seed': seed, 'description': DESCRIPTION, 'problem': {**problem.YEAST, 'description': DESCRIPTION},
            'measured': MEASURED, 'noise': {'rel_sd': REL_SD, 'abs_sd': ABS_SD},
            'units': {'x1': 'g / l', 'x2': 'g / l', 'u1': '1 / h', 'u2': 'g / l', 't': 'h'},
            'experiments': exps}
    truth = {'seed': seed, 'true_model': tr['sup'], 'law_iri': tr.get('law_iri'), 'theta': theta}
    return data, truth


def save(data: dict, truth: dict, name: str = 'run') -> str:
    """runs/<name>_seed<n>.json for the agents, runs/<name>_seed<n>.truth.json for scoring only. The name is in
    the file name, never in the data the agents read."""
    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, f"{name}_seed{data['seed']}.json")
    json.dump(data, open(path, 'w'), indent=1)
    json.dump(truth, open(path.replace('.json', '.truth.json'), 'w'), indent=1)
    return path


# Hidden truths for the experiments. Monod and Contois with the paper's values; Andrews (Zentou 2019's law put
# into Asprey's balances) with values chosen so substrate inhibition shows inside the paper's operating ranges
# (experiment E3 reaches about 11 g/L substrate). Zentou's own K_i (181 g/L) would never show here.
#   aiba        Muloiwa 2020 eq 11 (Aiba: exponential substrate inhibition), the near-twin of Andrews. KI = 6 g/L
#               gives about the same drop at 11 g/L as Andrews with Ki = 3 (exp(-11/6) = 0.16 vs 0.21).
#   notingraph  a law that is in no paper of the graph: Contois-type biomass dependence AND Andrews-type
#               inhibition, r = mumax x2 / (KS x1 + x2 + x2^2/Ki). The right answer is "nothing in the graph
#               explains this"; it tests whether the system says so or forces a law that fits badly.
KG = 'http://example.org/kg/'
SCENARIOS = {
    'monod': {'sup': 'asprey_2000_monod_kinetics', 'law': KG + 'asprey_2000_monod_kinetics/law/asprey_2000_eq17'},
    'contois': {'sup': 'asprey_2000_contois_kinetics', 'law': KG + 'asprey_2000_contois_kinetics/law/asprey_2000_eq18'},
    'andrews': {'base': 'asprey_2000_monod_kinetics', 'law': KG + 'zentou_2019_andrews/law/zentou_2019_eq5',
                'mapping': {'mu': 'r', 'S_': 'x2'},
                'theta': {'mumax': 0.5, 'KS': 0.3, 'Ki': 3.0, 'theta3': 0.559, 'theta4': 0.055}},
    'aiba': {'base': 'asprey_2000_monod_kinetics', 'law': KG + 'muloiwa_2020_free_laws/law/muloiwa_2020_eq11',
             'mapping': {'mu': 'r', 'S_': 'x2'},
             'theta': {'mumax': 0.5, 'Ks': 0.3, 'KI': 6.0, 'theta3': 0.559, 'theta4': 0.055}},
    'notingraph': {'base': 'asprey_2000_monod_kinetics', 'law': KG + 'zentou_2019_andrews/law/zentou_2019_eq5',
                   'mapping': {'mu': 'r', 'S_': 'x2'}, 'rhs': 'mumax*x2/(KS*x1 + x2 + x2**2/Ki)', 'law_iri': None,
                   'theta': {'mumax': 0.5, 'KS': 0.05, 'Ki': 3.0, 'theta3': 0.559, 'theta4': 0.055}},
}


def scenario(name: str) -> tuple[dict, dict | None]:
    s = SCENARIOS[name]
    if 'sup' in s:
        tr = translate.translate(s['sup'])
        tr['law_iri'] = s['law']
        return tr, None
    import candidates
    import sympy as sp
    tr = candidates.swap(translate.translate(s['base']), s['law'], s['mapping'])
    if 'rhs' in s:                                  # a law of our own: same parameters, a different expression
        L = tr['laws'][-1]
        L['rhs'] = sp.sympify(s['rhs'], locals={k: sp.Symbol(k) for k in ['mumax', 'KS', 'Ki', 'x1', 'x2']})
        L['rhs_text'], L['eq'] = sp.sstr(L['rhs']), 'not in the graph'
        tr['sup'] = tr['label'] = f"{s['base']} + a law not in the graph: r = {L['rhs_text']}"
        tr['law_iri'] = None
    return tr, s['theta']


if __name__ == '__main__':
    # python data.py andrews --seed 3      (scenario names: monod, contois, andrews; or a Sup id)
    seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 1
    what = sys.argv[1]
    tr, theta = scenario(what) if what in SCENARIOS else (translate.translate(what), None)
    data, truth = make(tr, seed, theta)
    truth['scenario'] = what
    if what in SCENARIOS:
        truth['law_iri'] = tr.get('law_iri')
        truth['rate_law'] = tr['laws'][-1]['rhs_text'] if what != 'monod' and what != 'contois' else \
            next(L['rhs_text'] for L in tr['laws'] if L.get('target') == 'r')
    print('wrote', save(data, truth, what))
