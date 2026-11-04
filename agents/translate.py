"""translate.py -- a law's LaTeX -> a SymPy expression the code can check and build. One LLM call per law, cached.

    python translate.py asprey_2000_monod_kinetics [--dry-run]

The LLM only transcribes: it gets the printed LaTeX and the law's symbols from the graph, each with a plain
name to use, and returns kind (ode / algebraic / other), target and right side. Then code checks, no LLM:
    parses       the right side is valid SymPy
    symbols      it uses exactly the law's symbols from the graph (plus t), nothing invented or dropped
    target       it computes what the graph says the law computes
    kind         ode / algebraic agrees with the form the extractor's code wrote in the graph
    units        Layer 1 (symbolic.check_law); a symbol with no unit in the graph gets one inferred
A law that fails a check is returned with its problems; nothing is fixed silently.
"""
from __future__ import annotations
import json
import os
import re
import sys

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'extractor_agent'))
from extractor import llm                                       # the extractor's cached, schema-enforced calls
import kg_query
import symbolic

CACHE = os.path.join(HERE, 'cache')                             # llm_cache/ goes in here
KIND_OF_FORM = {'ODE': 'ode', 'algebraic': 'algebraic', 'PDE': 'other', 'PDE/algebraic': 'other'}   # DAE: either
FUNCS = {'exp': sp.exp, 'log': sp.log, 'ln': sp.log, 'sqrt': sp.sqrt, 'Abs': sp.Abs, 'Max': sp.Max, 'Min': sp.Min,
         'erf': sp.erf, 'erfc': sp.erfc, 'tanh': sp.tanh, 'sinh': sp.sinh, 'cosh': sp.cosh, 'sin': sp.sin,
         'cos': sp.cos, 'atan': sp.atan, 'Heaviside': sp.Heaviside, 'Piecewise': sp.Piecewise}

SYSTEM = """You transcribe one printed equation from a scientific paper into SymPy syntax.
You get the equation's LaTeX and the list of its symbols, each with the plain name you must use.
Return:
- kind: "ode" if the left side is the time derivative of one symbol (d x/dt), "algebraic" if the left side is one
  symbol, "other" for anything else (for example d(V x)/dt, or no single left side).
- target: the plain name of the symbol on the left side (empty for "other").
- rhs: the right side in SymPy syntax, using only the plain names given, t for time, numbers, + - * / **, and
  exp, log, sqrt, Abs, Max, Min. Write multiplication explicitly with *.
- note: anything unusual (a label before the equation, a symbol you could not place), or empty.
Transcribe exactly what is printed. Do not simplify, correct, rearrange or complete the equation, even if it
looks wrong: checking it is someone else's job. Ignore labels such as "M_I(theta, u, t):" before the equation."""

SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['kind', 'target', 'rhs', 'note'],
          'properties': {'kind': {'type': 'string', 'enum': ['ode', 'algebraic', 'other']},
                         'target': {'type': 'string'}, 'rhs': {'type': 'string'}, 'note': {'type': 'string'}}}


def plain(sym: str) -> str:
    """'\\theta_{1}' -> 'theta1', 'x_{1}' -> 'x1', 'K_{glc}' -> 'Kglc', '[GLC]' -> 'GLC'."""
    s = re.sub(r'\\(mathrm|mathbf|boldsymbol|pmb|text|mathit)\b', '', sym)
    s = re.sub(r'[^A-Za-z0-9]', '', s)
    return s if s and not s[0].isdigit() else 's' + s


def names(symbols: dict) -> dict:
    """graph symbol -> plain name, unique within the model."""
    out, seen = {}, {}
    for sym in symbols:
        p = plain(sym)
        if p in ('t', 'E', 'I', 'S', 'N', 'O', 'Q', 'beta', 'gamma', 'lambda') or p in FUNCS:   # SymPy reserved
            p = p + '_'
        seen[p] = seen.get(p, 0) + 1
        out[sym] = p if seen[p] == 1 else f'{p}{seen[p]}'
    return out


def law_symbols(law: dict) -> list:
    return [s for s in [law['computes'], *law['uses'], *law['parameters']] if s]


def ask(law: dict, model: dict, nm: dict) -> dict | None:
    rows = '\n'.join(f"  {s}  ->  {nm[s]}   ({model['symbols'][s].get('description', '')})" for s in law_symbols(law))
    user = f"Equation {law['eq']}:\n{law['latex']}\n\nSymbols (as printed -> plain name to use):\n{rows}"
    return llm.cached_ask(CACHE, SYSTEM, user, SCHEMA, 'translate', label=f"eq {law['eq']}")


def check(law: dict, ans: dict, model: dict, nm: dict) -> dict:
    """The code checks on one answer. Returns the law ready for build.py, with 'problems' (empty = passed)."""
    out = {'eq': law['eq'], 'latex': law['latex'], 'kind': ans['kind'], 'target': ans['target'],
           'rhs_text': ans['rhs'], 'note': ans['note'], 'problems': [], 'inferred': {}}
    expected_kind = KIND_OF_FORM.get(law.get('form'))
    if expected_kind and ans['kind'] != expected_kind:
        out['problems'].append(f"the LLM read it as {ans['kind']}, the graph's form says {law['form']} "
                               f"({expected_kind})")
    if ans['kind'] == 'other':
        out['problems'].append('the left side is not d(x)/dt or a single symbol; not supported yet')
        return out
    allowed = {nm[s] for s in law_symbols(law)}
    # names the answer uses that are not the law's symbols become plain symbols too, so the check below can say
    # "uses S, which is not one of the law's symbols" (otherwise SymPy reads S, E, I, N, Q as its own constants)
    stray = {w for w in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', ans['rhs'])} - set(FUNCS) - allowed - {'t', 'pi'}
    try:
        rhs = sp.sympify(ans['rhs'], locals={**FUNCS, **{p: sp.Symbol(p) for p in allowed | stray}, 't': symbolic.T})
    except Exception as e:
        out['problems'].append(f'the right side does not parse: {e}')
        return out
    out['rhs'] = rhs
    used = {str(s) for s in rhs.free_symbols} - {'t'}
    if nm.get(law['computes']) != ans['target']:
        out['problems'].append(f"it computes {ans['target']}, but the graph says eq {law['eq']} computes "
                               f"{nm.get(law['computes'])}")
    expected = allowed - {ans['target']}
    if ans['kind'] == 'ode' and ans['target'] in used:
        expected |= {ans['target']}                             # dx/dt = f(x) uses x
    for s in sorted(used - allowed):
        out['problems'].append(f'uses {s}, which is not one of the law\'s symbols in the graph')
    for s in sorted(expected - used - ({ans['target']} if ans['kind'] == 'ode' else set())):
        out['problems'].append(f'drops {s}, which the graph says the law uses')
    inv = {v: k for k, v in nm.items()}
    units = {p: model['symbols'][inv[p]].get('unit') for p in allowed}
    dims = symbolic.check_law(ans['target'], ans['kind'], rhs, units)
    out['problems'] += dims['problems']
    out['inferred'] = dims['inferred']
    return out


def translate(sup: str) -> dict:
    """The whole model: every law translated and checked, plus the structure check."""
    return translate_model(kg_query.get_model(sup))


def translate_model(model: dict) -> dict:
    """Same, for a model dict from kg_query (get_model, or get_law for a single law)."""
    sup = model['sup']
    nm = names(model['symbols'])
    laws = []
    for law in model['laws']:
        ans = ask(law, model, nm)
        if ans is None:                                          # dry run
            continue
        laws.append(check(law, ans, model, nm))
    good = [L for L in laws if 'rhs' in L]
    computed = {L['target'] for L in good}
    params = {nm[s] for s, S in model['symbols'].items() if S['kind'] == 'Parameter'}
    inputs = {nm[s] for s, S in model['symbols'].items() if S['kind'] == 'Variable'} - computed
    structure = symbolic.check_model(good, inputs, params) if good else None
    return {'sup': sup, 'label': model.get('label', sup), 'names': nm, 'laws': laws, 'structure': structure,
            'parameters': sorted(params), 'inputs': sorted(inputs), 'symbols': model['symbols'],
            'ok': bool(good) and all(not L['problems'] for L in laws) and structure['ok']}


def report(t: dict) -> str:
    lines = [f"{t['sup']}: {'all checks passed' if t['ok'] else 'PROBLEMS'}"]
    for L in t['laws']:
        lines.append(f"  eq {L['eq']:5} {L['kind']:9} {L['target']} = {L['rhs_text']}")
        lines += [f"        ! {p}" for p in L['problems']]
        lines += [f"        unit inferred: {s} [{u}]" for s, u in L['inferred'].items()]
        if L['note']:
            lines.append(f"        note: {L['note']}")
    if t['structure']:
        s = t['structure']
        lines.append(f"  structure: states {s['states']}, algebraic {s['algebraic']}, inputs {t['inputs']}, "
                     f"parameters {t['parameters']}; {s['equations']} equations, {s['unknowns']} unknowns")
        lines += [f"        ! {p}" for p in s['problems']]
    return '\n'.join(lines)


if __name__ == '__main__':
    llm.DRY_RUN = '--dry-run' in sys.argv
    for sup in [a for a in sys.argv[1:] if not a.startswith('--')]:
        res = translate(sup)
        print(report(res) if res['laws'] else llm.usage_line())
    print(llm.usage_line())
