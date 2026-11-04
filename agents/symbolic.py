"""symbolic.py -- Layer 1: checks on a model's equations before anything is solved. Code only, no LLM.

check_law(target, kind, rhs, units)   do the units balance? A sum needs terms of one dimension, exp() and log()
                                      a dimensionless argument, and the right side must have the dimension of
                                      the target (of target/time for an ODE). A symbol with no unit in the graph
                                      is solved for: its unit is inferred from the equation and reported.
check_model(laws, inputs, params)     structure: each quantity computed once, every other symbol an input or a
                                      parameter, as many equations as unknowns.

Dimensions are vectors of exponents over pint's base dimensions ([mass], [length], [time], ...). A symbol
without a unit gets a vector of unknown exponents; every rule above is a linear equation in them.
"""
from __future__ import annotations
import os
import sys

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'extractor_agent'))
from extractor.units import U                                   # the extractor's registry: cell, OD, ... defined

T = sp.Symbol('t')


def dim_of(unit: str | None) -> dict | None:
    """'g / l' -> {'[mass]': 1, '[length]': -3}; None if there is no unit or it does not parse."""
    if unit is None:
        return None
    if unit.strip() in ('dimensionless', '', '1'):
        return {}
    try:
        return {k: float(v) for k, v in U.Quantity(1, unit).dimensionality.items()}
    except Exception:
        return None


def show(d: dict) -> str:
    if not d:
        return 'dimensionless'
    top = [f"{k.strip('[]')}{'' if v == 1 else '^' + str(int(v) if v == int(v) else v)}" for k, v in d.items() if v > 0]
    bot = [f"{k.strip('[]')}{'' if v == -1 else '^' + str(int(-v) if v == int(v) else -v)}" for k, v in d.items() if v < 0]
    return (' '.join(top) or '1') + (' / ' + ' '.join(bot) if bot else '')


class _Dims:
    """Walks a SymPy expression, collecting 'these two must be equal' equations."""

    def __init__(self, known: dict, bases: list):
        self.known, self.bases, self.unknown, self.eqs, self.warnings = known, bases, {}, [], []

    def vec(self, d: dict) -> dict:
        return {b: sp.Float(d.get(b, 0)) for b in self.bases}

    def of(self, e):
        if e.is_Number:
            return self.vec({})
        if e.is_Symbol:
            if e == T:
                return self.vec({'[time]': 1})
            if self.known.get(str(e)) is not None:
                return self.vec(self.known[str(e)])
            if str(e) not in self.unknown:
                self.unknown[str(e)] = {b: sp.Symbol(f'{e}__{b.strip("[]")}') for b in self.bases}
            return self.unknown[str(e)]
        if e.is_Add:
            terms = [self.of(a) for a in e.args]
            for a, d in zip(e.args[1:], terms[1:]):
                self.eqs.append((d, terms[0], f'the terms {sp.sstr(e.args[0])} and {sp.sstr(a)} are added'))
            return terms[0]
        if e.is_Mul:
            out = self.vec({})
            for a in e.args:
                out = {b: out[b] + v for b, v in self.of(a).items()}
            return out
        if e.is_Pow:
            base, ex = e.args
            if ex.is_Number:
                return {b: v * ex for b, v in self.of(base).items()}
            # x**n with a fitted exponent (Moser: S**n, K_s in (g/L)**n): units cannot be checked by linear rules.
            # Not an error: the term gets its own unknown dimension and a warning is recorded.
            self._dimensionless(ex, f'the exponent {sp.sstr(ex)}')
            self.warnings.append(f'{sp.sstr(e)} has a fitted exponent: its units cannot be checked')
            key = f'pow{len(self.warnings)}'
            self.unknown[key] = {b: sp.Symbol(f'{key}__{b.strip("[]")}') for b in self.bases}
            return self.unknown[key]
        if isinstance(e, (sp.exp, sp.log)) or e.is_Function:
            for a in e.args:
                self._dimensionless(a, f'the argument of {e.func.__name__}()')
            return self.vec({})
        raise ValueError(f'no dimension rule for {sp.srepr(e)[:60]}')

    def _dimensionless(self, e, why):
        self.eqs.append((self.of(e), self.vec({}), f'{why} must be dimensionless'))


def check_law(target: str, kind: str, rhs: sp.Expr, units: dict) -> dict:
    """units: symbol name -> unit string or None. Returns {'ok', 'problems', 'inferred'}."""
    known = {s: dim_of(u) for s, u in units.items()}
    bases = sorted({b for d in known.values() if d for b in d} | {'[time]', '[mass]', '[length]'})
    D = _Dims(known, bases)
    lhs = D.of(sp.Symbol(target))
    if kind == 'ode':
        lhs = {b: v - (1 if b == '[time]' else 0) for b, v in lhs.items()}
    D.eqs.append((D.of(rhs), lhs, f'the right side must have the dimension of {"d" + target + "/dt" if kind == "ode" else target}'))
    problems, system = [], []
    for a, b, why in D.eqs:
        diff = [sp.nsimplify(a[k] - b[k]) for k in bases]
        if all(x.is_Number for x in diff):
            if any(x != 0 for x in diff):
                problems.append(f'{why}, but they differ: {show({k: float(a[k]) for k in bases if a[k] != 0})} '
                                f'vs {show({k: float(b[k]) for k in bases if b[k] != 0})}')
        else:
            system += [x for x in diff if x != 0]
    inferred = {}
    pow_terms = [k for k in D.unknown if k.startswith('pow')]
    if D.unknown:
        unknowns = [s for v in D.unknown.values() for s in v.values()]
        sol = sp.linsolve(system, unknowns) if system else None
        if system and sol == sp.EmptySet:
            problems.append(f'no unit for {", ".join(D.unknown)} makes the equation balance')
        for name, v in D.unknown.items():
            if not sol or sol == sp.EmptySet:
                inferred[name] = 'not determined'
                continue
            vals = dict(zip(unknowns, list(sol)[0]))
            exps = {b: vals[s] for b, s in v.items()}
            inferred[name] = (show({b: float(x) for b, x in exps.items() if x != 0})
                              if all(x.is_Number for x in exps.values()) else 'not determined')
    for k in pow_terms:
        inferred.pop(k, None)
    for name in list(inferred):
        if pow_terms and inferred[name] == 'not determined':
            inferred[name] = 'not determined (fitted exponent)'
    return {'ok': not problems, 'problems': problems, 'inferred': inferred, 'warnings': D.warnings}


def check_model(laws: list, inputs: set, params: set) -> dict:
    """laws: [{'eq', 'target', 'kind', 'rhs'}]. Structure only: who computes what, what is left over."""
    problems, computed = [], {}
    for L in laws:
        if L['target'] in computed:
            problems.append(f"{L['target']} is computed twice (eq {computed[L['target']]} and eq {L['eq']})")
        computed[L['target']] = L['eq']
    used = {str(s) for L in laws for s in L['rhs'].free_symbols} - {'t'}
    for s in sorted(used - set(computed) - inputs - params):
        problems.append(f'{s} is used but nothing computes it, and it is not an input or a parameter')
    for s in sorted((inputs | params) & set(computed)):
        problems.append(f'{s} is computed by eq {computed[s]} but is also given as an input or parameter')
    states = [L['target'] for L in laws if L['kind'] == 'ode']
    return {'ok': not problems, 'problems': problems, 'states': states,
            'algebraic': [L['target'] for L in laws if L['kind'] == 'algebraic'],
            'equations': len(laws), 'unknowns': len(computed)}


def check_positivity(laws: list, inputs: set, params: set, n: int = 2000, seed: int = 0) -> dict:
    """Limiting behaviour: a concentration that starts at or above zero must stay there. For each ODE state x,
    with x = 0 and everything else non-negative, dx/dt must not be negative; otherwise the model consumes
    what is not there (biomass or substrate below zero). Checked at n random points: other states, inputs and
    parameters drawn log-uniformly over 1e-3..1e3 (inputs also at exactly 0, e.g. no feed), because the fitted
    values are not known yet. Assumes states and inputs are non-negative quantities (true for concentrations,
    flow rates, feed concentrations). Returns {'ok', 'problems', 'examples'}."""
    import numpy as np
    odes = [L for L in laws if L.get('kind') == 'ode' and 'rhs' in L]
    alg = {sp.Symbol(L['target']): L['rhs'] for L in laws if L.get('kind') == 'algebraic' and 'rhs' in L}
    for _ in range(len(alg)):
        alg = {k: v.subs(alg) for k, v in alg.items()}
    states = [L['target'] for L in odes]
    rng = np.random.default_rng(seed)
    problems, examples = [], {}
    for L in odes:
        x = L['target']
        full = L['rhs'].subs(alg)
        rhs = full.subs(sp.Symbol(x), 0)
        if rhs.has(sp.zoo, sp.nan, sp.oo, -sp.oo):     # 0/0 or log(0) at x = 0: take the limit x -> 0+ instead
            pos = {s: sp.Symbol(str(s), positive=True) for s in full.free_symbols}
            try:
                rhs = sp.limit(full.subs(pos), pos[sp.Symbol(x)], 0, '+').subs({v: k for k, v in pos.items()})
            except Exception:
                rhs = sp.nan
            if rhs.has(sp.zoo, sp.nan) or not rhs.free_symbols and not rhs.is_finite:
                if rhs in (sp.oo,):                    # grows without bound: not negative
                    continue
                problems.append(f"{x}: the rate of change at {x} = 0 is undefined (division by zero or log of "
                                f"zero); limiting behaviour could not be checked")
                continue
        syms = sorted(str(s) for s in rhs.free_symbols if str(s) != 't')
        if not syms:
            if rhs.is_number and float(rhs) < -1e-9:
                problems.append(f"{x} can go below zero: with {x} = 0 its rate of change is {float(rhs):.3g}")
            continue
        f = sp.lambdify([sp.Symbol(s) for s in syms], rhs, modules='numpy')
        vals = {}
        for s in syms:
            v = 10 ** rng.uniform(-3, 3, n)
            if s in inputs:
                v[rng.random(n) < 0.25] = 0.0
            vals[s] = v
        with np.errstate(all='ignore'):
            d = np.asarray(f(*[vals[s] for s in syms]), float) * np.ones(n)
        bad = np.isfinite(d) & (d < -1e-9)
        if bad.any():
            i = int(np.argmax(bad))
            ex = {s: float(f'{vals[s][i]:.3g}') for s in syms}
            examples[x] = {'at': ex, 'dxdt': float(f'{d[i]:.3g}'), 'share': float(bad.mean())}
            problems.append(f"{x} can go below zero: with {x} = 0 its rate of change is still negative "
                            f"(d{x}/dt = {d[i]:.3g} at {', '.join(f'{k}={v}' for k, v in ex.items())}; "
                            f"{100 * bad.mean():.0f}% of the points checked). The rate law does not vanish when "
                            f"{x} runs out (limiting behaviour)")
    return {'ok': not problems, 'problems': problems, 'examples': examples}
