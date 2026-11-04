"""reference.py -- Asprey & Macchietto (2000) baker's yeast models written by hand, as the answer key for testing
the generated models. Not used by the agents. Integrated with SciPy, fully independent of Pyomo and the graph.

    dx1/dt = (r - u1 - theta4) x1                    biomass
    dx2/dt = -r x1 / theta3 + u1 (u2 - x2)           substrate
    Monod:   r = theta1 x2 / (theta2 + x2)
    Contois: r = theta1 x2 / (theta2 x1 + x2)
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

from build import input_value


def rate(kind: str, th: dict, x1: float, x2: float) -> float:
    if kind == 'monod':
        return th['theta1'] * x2 / (th['theta2'] + x2)
    return th['theta1'] * x2 / (th['theta2'] * x1 + x2)


def simulate(kind: str, exp: dict, th: dict) -> dict:
    def f(t, x):
        u1, u2 = input_value(exp['inputs']['u1'], t), input_value(exp['inputs']['u2'], t)
        r = rate(kind, th, x[0], x[1])
        return [(r - u1 - th['theta4']) * x[0], -r * x[0] / th['theta3'] + u1 * (u2 - x[1])]
    t_switch = sorted({s for steps in exp['inputs'].values() for s, _ in steps} | {exp['tf']})
    x, out = [exp['x0']['x1'], exp['x0']['x2']], {'t': list(exp['samples']), 'x1': [], 'x2': []}
    sol_t, sol_x = [], []
    for a, b in zip([0] + t_switch[:-1], t_switch):              # integrate piece by piece across input switches
        if b <= a:
            continue
        ts = [t for t in exp['samples'] if a <= t <= b]
        s = solve_ivp(f, (a, b), x, t_eval=ts, rtol=1e-10, atol=1e-12, method='LSODA')
        sol_t += list(s.t)
        sol_x += list(s.y.T)
        x = list(s.y[:, -1]) if len(s.t) and s.t[-1] == b else list(solve_ivp(f, (a, b), x, rtol=1e-10, atol=1e-12).y[:, -1])
    for t in exp['samples']:
        i = int(np.argmin([abs(t - u) for u in sol_t]))
        out['x1'].append(float(sol_x[i][0]))
        out['x2'].append(float(sol_x[i][1]))
    return out


SWITCHING = {'tf': 20, 'samples': list(range(21)), 'x0': {'x1': 3.0, 'x2': 0.01},
             'inputs': {'u1': [(0, 0.12), (5, 0.05), (12, 0.2)], 'u2': [(0, 30.0), (8, 8.0)]}}

if __name__ == '__main__':
    # python reference.py : the week-1 test. The models built from the graph must match the hand-written ones.
    import translate
    import build
    from extractor import llm
    for sup, kind in [('asprey_2000_monod_kinetics', 'monod'), ('asprey_2000_contois_kinetics', 'contois')]:
        tr = translate.translate(sup)
        print(translate.report(tr))
        if not tr['ok']:
            print('  checks failed: not built')
            continue
        th = build.paper_values(tr)
        for name, e in [('paper experiment', build.DEFAULT_EXPERIMENT), ('switching inputs', SWITCHING)]:
            g, r = build.simulate(tr, e, th), simulate(kind, e, th)
            diff = max(abs(a - b) for s in ('x1', 'x2') for a, b in zip(g[s], r[s]))
            print(f"  {name}: solved {g['ok']}, largest difference from the hand-written model {diff:.1e} g/L "
                  f"({'PASS' if g['ok'] and diff < 1e-2 else 'FAIL'})")
    print(llm.usage_line())
