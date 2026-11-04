"""build.py -- checked SymPy laws -> a Pyomo.DAE model, and simulation of one experiment. Code only, no LLM.

    python build.py asprey_2000_monod_kinetics        # translate (cached), build, simulate the default experiment

build(tr, exp, params)   tr: translate.translate(sup) output. ODE laws become derivative equations, algebraic
                         laws (r = ...) become constraints at every time point. Parameters are Pyomo Vars, fixed
                         at the given values (parmest unfixes them later). Inputs come from the experiment.
simulate(...)            builds, solves with ipopt and returns the states at the sampling times.

An experiment: {'tf': hours, 'samples': [times], 'x0': {state: value}, 'inputs': {name: [(t_start, value), ...]}}.
Inputs are piecewise constant and switch only at whole hours, which are always element boundaries, so a
switch never falls inside a collocation element.
"""
from __future__ import annotations
import os
import time
import shutil
import sys

import pyomo.environ as pyo
import sympy as sp
from pyomo.dae import ContinuousSet, DerivativeVar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PYO_FUNCS = {'exp': pyo.exp, 'log': pyo.log, 'sqrt': pyo.sqrt}

# Asprey & Macchietto (2000), first experiment: 10 h, hourly samples, inputs constant at 0.2 h^-1 and 15 g/L.
# Initial biomass is not stated for it (range 1-10 g/L); 5 g/L is our choice.
DEFAULT_EXPERIMENT = {'tf': 10, 'samples': list(range(0, 11)), 'x0': {'x1': 5.0, 'x2': 0.01},
                      'inputs': {'u1': [(0, 0.2)], 'u2': [(0, 15.0)]}}


def find_ipopt() -> str:
    exe = os.environ.get('IPOPT') or shutil.which('ipopt')       # IPOPT=/full/path/to/ipopt if it is elsewhere
    if not exe:
        try:
            import idaes  # noqa: F401  (adds ~/.idaes/bin to PATH)
        except ImportError:
            pass
        exe = shutil.which('ipopt') or os.path.expanduser('~/.idaes/bin/ipopt')
    if not os.path.exists(exe):
        raise RuntimeError('ipopt not found: pip install idaes-pse && idaes get-extensions')
    folder = os.path.dirname(exe)                               # parmest calls 'ipopt' by name: put it on PATH
    if folder not in os.environ['PATH'].split(os.pathsep):
        os.environ['PATH'] = folder + os.pathsep + os.environ['PATH']
    return exe


def input_value(steps: list, t: float) -> float:
    """Piecewise constant, left-continuous: the value on (t_k, t_k+1] is the one set at t_k."""
    v = steps[0][1]
    for start, val in steps:
        if t > start + 1e-9:
            v = val
    return v


def paper_values(tr: dict) -> dict:
    """The parameter values the graph holds (the paper's), by plain name."""
    out = {}
    for sym, S in tr['symbols'].items():
        if S['kind'] == 'Parameter' and S.get('value') not in (None, ''):
            try:
                out[tr['names'][sym]] = float(S['value'])
            except ValueError:
                pass
    return out


DEADLINE = {'t': None}    # set by fit.fit: wall-clock time after which no new integration starts


class TimeUp(Exception):
    pass


def integrate(tr: dict, exp: dict, params: dict, rtol: float = 1e-8, atol: float = 1e-10):
    """The same model integrated with SciPy (algebraic laws substituted into the ODEs), as a function of time.
    Used only as the starting point for ipopt, so the solver starts next to the solution instead of from flat
    lines. Returns None if the integration fails (the model is then started from the initial values)."""
    from scipy.integrate import solve_ivp
    laws = [L for L in tr['laws'] if 'rhs' in L]
    alg = {sp.Symbol(L['target']): L['rhs'] for L in laws if L['kind'] == 'algebraic'}
    for _ in range(len(alg)):                                       # resolve algebraic laws that use each other
        alg = {k: v.subs(alg) for k, v in alg.items()}
    odes = [L for L in laws if L['kind'] == 'ode']
    states = [L['target'] for L in odes]
    rhs = [L['rhs'].subs(alg) for L in odes]
    syms = sorted({str(s) for e in list(rhs) + list(alg.values()) for s in e.free_symbols})
    f_rhs = sp.lambdify([sp.Symbol(a) for a in syms], rhs, modules='numpy')
    f_alg = {str(k): sp.lambdify([sp.Symbol(a) for a in syms], v, modules='numpy') for k, v in alg.items()}

    def args(t, x):
        vals = dict(zip(states, x))
        out = []
        for a in syms:
            if a in vals:
                out.append(vals[a])
            elif a in params:
                out.append(params[a])
            elif a == 't':
                out.append(t)
            else:
                out.append(input_value(exp['inputs'][a], t))
        return out
    switches = sorted({s for steps in exp['inputs'].values() for s, _ in steps if 0 < s < exp['tf']})
    pieces, x = [], [exp['x0'][s] for s in states]
    try:
        for a, b in zip([0] + switches, switches + [exp['tf']]):    # integrate piece by piece across input switches
            if DEADLINE['t'] is not None and time.time() > DEADLINE['t']:
                return None                             # past the fit's time limit: do not start another piece
            sol = solve_ivp(lambda t, y: f_rhs(*args(t, y)), (a, b), x, method='LSODA', rtol=rtol, atol=atol,
                            dense_output=True)
            if not sol.success:
                return None
            pieces.append((a, b, sol.sol))
            x = list(sol.y[:, -1])
    except Exception:
        return None

    def at(t):
        for a, b, f in pieces:
            if t <= b + 1e-12:
                y = f(min(max(t, a), b))
                return dict(zip(states, y)), {k: float(g(*args(t, y))) for k, g in f_alg.items()}
        y = pieces[-1][2](pieces[-1][1])
        return dict(zip(states, y)), {k: float(g(*args(t, y))) for k, g in f_alg.items()}
    return at


def build(tr: dict, exp: dict, params: dict, nfe_per_hour: int = 4, ncp: int = 3) -> pyo.ConcreteModel:
    laws = [L for L in tr['laws'] if 'rhs' in L]
    states = [L['target'] for L in laws if L['kind'] == 'ode']
    alg = [L['target'] for L in laws if L['kind'] == 'algebraic']
    missing = [p for p in tr['parameters'] if p not in params]
    if missing:
        raise ValueError(f'no value for parameters {missing}')
    m = pyo.ConcreteModel(name=tr['sup'])
    m.t = ContinuousSet(bounds=(0, exp['tf']), initialize=sorted(set(exp['samples']) | set(range(int(exp['tf']) + 1))))
    m.S = pyo.Set(initialize=states)
    m.A = pyo.Set(initialize=alg)
    m.P = pyo.Set(initialize=tr['parameters'])
    m.x = pyo.Var(m.t, m.S, initialize=lambda m, t, s: exp['x0'][s], bounds=(0, None))   # time first: parmest wants it
    m.a = pyo.Var(m.A, m.t, initialize=0.1)
    m.dx = DerivativeVar(m.x, wrt=m.t)
    m.theta = pyo.Var(m.P, initialize=lambda m, p: params[p])
    m.theta.fix()
    args = sorted({str(s) for L in laws for s in L['rhs'].free_symbols})
    fns = {L['target']: sp.lambdify([sp.Symbol(a) for a in args], L['rhs'], modules=[PYO_FUNCS]) for L in laws}

    def value(name, t):
        if name in states:
            return m.x[t, name]
        if name in alg:
            return m.a[name, t]
        if name in tr['parameters']:
            return m.theta[name]
        if name == 't':
            return t
        if name in exp['inputs']:
            return input_value(exp['inputs'][name], t)
        raise KeyError(f'{name}: not a state, algebraic variable, parameter or input of the experiment')

    def ode(m, t, s):
        if t == m.t.first():
            return pyo.Constraint.Skip
        return m.dx[t, s] == fns[s](*[value(a, t) for a in args])
    m.ode = pyo.Constraint(m.t, m.S, rule=ode)
    m.alg = pyo.Constraint(m.A, m.t, rule=lambda m, a, t: m.a[a, t] == fns[a](*[value(n, t) for n in args]))
    for s in states:
        m.x[0, s].fix(exp['x0'][s])
    pyo.TransformationFactory('dae.collocation').apply_to(m, nfe=int(exp['tf'] * nfe_per_hour), ncp=ncp,
                                                          scheme='LAGRANGE-RADAU')
    m.obj = pyo.Objective(expr=0)
    start = integrate(tr, exp, params)                              # warm start at every collocation point
    if start is not None:
        for t in m.t:
            xs, al = start(t)
            for s in states:
                if not m.x[t, s].fixed:
                    m.x[t, s].set_value(max(float(xs[s]), 0.0))
            for a in alg:
                m.a[a, t].set_value(al[a])
        for t in m.t:
            for s in states:
                try:
                    m.dx[t, s].set_value(pyo.value(fns[s](*[value(a, t) for a in args])))
                except Exception:
                    pass
    return m


LOG = os.path.join(HERE, 'runs', 'ipopt_last_failure.log')


def solve(m: pyo.ConcreteModel, tee: bool = False) -> bool:
    """True if ipopt finds the solution. On failure nothing is loaded, False is returned and ipopt's output is
    kept in runs/ipopt_last_failure.log, so a failure never stops a run."""
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    tmp = LOG + '.tmp'
    try:
        res = pyo.SolverFactory('ipopt', executable=find_ipopt()).solve(m, tee=tee, load_solutions=False, logfile=tmp)
    except Exception as e:
        open(LOG, 'w').write(f'{m.name}: {type(e).__name__}: {e}\n')
        return False
    ok = (res.solver.status == pyo.SolverStatus.ok and
          res.solver.termination_condition == pyo.TerminationCondition.optimal)
    if ok:
        m.solutions.load_from(res)
    elif os.path.exists(tmp):
        os.replace(tmp, LOG)
    return ok


def trajectories(m: pyo.ConcreteModel, samples: list) -> dict:
    return {s: [pyo.value(m.x[t, s]) for t in samples] for s in m.S}


def simulate(tr: dict, exp: dict, params: dict) -> dict:
    m = build(tr, exp, params)
    ok = solve(m)
    return {'ok': ok, 't': list(exp['samples']), **trajectories(m, exp['samples'])}


if __name__ == '__main__':
    import translate
    from extractor import llm
    for sup in sys.argv[1:]:
        tr = translate.translate(sup)
        print(translate.report(tr))
        if not tr['ok']:
            continue
        res = simulate(tr, DEFAULT_EXPERIMENT, paper_values(tr))
        print(f"  solved: {res['ok']}")
        for i, t in enumerate(res['t']):
            print(f"   t={t:4}  " + '  '.join(f'{s}={res[s][i]:8.4f}' for s in tr['structure']['states']))
    print(llm.usage_line())
