"""candidates.py -- build a candidate model by putting a rate law from anywhere in the graph into a base model.

    python candidates.py          # demo: Andrews (Zentou 2019) and Teissier (Muloiwa 2020) into Asprey's balances

swap(base, law_iri, mapping)  base: translate.translate(sup) of the model whose balances are kept (Asprey).
                              law_iri: a rate law found by kg_query.find_rate_laws().
                              mapping: the law's variables -> the base model's names, e.g. {'mu': 'r', 'S_': 'x2'}.
                              Proposed by an agent (LLM); several law symbols may map to one base symbol.
    The law's parameters become new free parameters of the candidate, named after the law (mumax, KS, Ki).
    The base law that computed the same quantity is removed. Then code checks the result, no LLM:
        mapping     every variable of the law is mapped, to a symbol the base model has
        units       Layer 1 on the new law, with the base model's units for mapped variables; parameter units
                    the paper did not give are inferred
        structure   each quantity computed once, nothing left unexplained
        positivity  limiting behaviour: no state (biomass, substrate) can be driven below zero
    A candidate that fails is returned with its problems, for the agent to read.
start_values(cand)            starting guesses for fitting: the paper's value when its unit converts to the
                              model's units, else 1.0 (the fit does not depend on good guesses; see fit.py tests).
"""
from __future__ import annotations
import copy

import sympy as sp

import kg_query
import symbolic
import translate
from symbolic import U


def _norm_symbol(s: str) -> str:
    """Printed symbol -> comparable key: '\\upmu', '\\mu', 'μ', 'mu' -> 'mu'; 'K_{S}' -> 'ks'; '[S]' -> 's'."""
    import re
    s = s.replace('μ', '\\mu').replace('µ', '\\mu')
    s = re.sub(r'\\(mathrm|mathbf|boldsymbol|pmb|mathit|text|operatorname)', '', s)
    s = re.sub(r'\\up([a-z]+)', r'\\\1', s)                  # \upmu (upright Greek) -> \mu
    return re.sub(r'[\\{}\[\]_\s^*]', '', s).lower()


def resolve_symbol(law_tr: dict, sym: str) -> str:
    """An agent's name for a law variable -> the law's plain name: the plain name itself, the printed LaTeX
    symbol, or a variant of it (\\upmu for \\mu, Unicode μ, missing braces). Unmatched names come back as given,
    and the code checks then report them."""
    names = law_tr['names']                                       # printed symbol -> plain name
    if sym in names.values():
        return sym
    if sym in names:
        return names[sym]
    key = _norm_symbol(sym)
    hits = {plain for printed, plain in names.items() if key in (_norm_symbol(printed), _norm_symbol(plain))}
    return hits.pop() if len(hits) == 1 else sym


def _free_name(name: str, taken: set) -> str:
    out, k = name, 2
    while out in taken:
        out, k = f'{name}_{k}', k + 1
    return out


def swap(base: dict, law_iri: str, mapping: dict, law_tr: dict | None = None) -> dict:
    law_tr = law_tr or translate.translate_model(kg_query.get_law(law_iri))
    L = law_tr['laws'][0]
    src = kg_query.source_of(law_iri)
    cand = {'sup': f"{base['sup']} + {src} eq {L['eq']}", 'label': f"{base['sup']} balances with {src} eq {L['eq']}",
            'base': base['sup'], 'law_iri': law_iri, 'mapping': dict(mapping), 'problems': [], 'inferred': {}}
    if L['problems'] or 'rhs' not in L:
        cand['problems'] += [f"translation of eq {L['eq']}: {p}" for p in L['problems']]
        cand['ok'] = False
        return cand
    law_params = set(law_tr['parameters'])
    law_vars = {str(s) for s in L['rhs'].free_symbols} - law_params - {'t'} | {L['target']}
    base_names = set(base['names'].values())
    for v in sorted(law_vars):
        if v not in mapping:
            cand['problems'].append(f'the law\'s variable {v} is not mapped to anything in the model')
        elif mapping[v] not in base_names:
            cand['problems'].append(f'{v} is mapped to {mapping[v]}, which the model does not have')
    target = mapping.get(L['target'])
    old = [B for B in base['laws'] if B.get('target') == target]
    if target and not old:
        cand['problems'].append(f'no law of the model computes {target}, so there is nothing to replace')
    if cand['problems']:
        cand['ok'] = False
        return cand

    # rename: variables to the model's names, parameters to fresh names
    kept = [B for B in base['laws'] if B.get('target') != target]
    taken = (set(base_names) - set(base['parameters'])) | {str(s) for B in kept if 'rhs' in B for s in B['rhs'].free_symbols}
    pmap = {}
    for p in sorted(law_params):
        pmap[p] = _free_name(p, taken)
        taken.add(pmap[p])
    rhs = L['rhs'].subs({sp.Symbol(k): sp.Symbol(v) for k, v in {**mapping, **pmap}.items() if k != L['target']},
                        simultaneous=True)
    inv_base = {v: k for k, v in base['names'].items()}
    inv_law = {v: k for k, v in law_tr['names'].items()}
    units = {}
    for s in rhs.free_symbols:
        s = str(s)
        if s in inv_base:
            units[s] = base['symbols'][inv_base[s]].get('unit')
    for p, newp in pmap.items():
        units[newp] = law_tr['symbols'][inv_law[p]].get('unit')
    units[target] = base['symbols'][inv_base[target]].get('unit')
    dims = symbolic.check_law(target, 'algebraic', rhs, units)
    new_law = {'eq': f"{src} eq {L['eq']}", 'latex': L['latex'], 'kind': 'algebraic', 'target': target, 'rhs': rhs,
               'rhs_text': sp.sstr(rhs), 'note': L['note'], 'problems': dims['problems'], 'inferred': dims['inferred']}
    laws = kept + [new_law]

    # symbols and names: the base's, plus the new parameters (with the law's descriptions, units and values)
    cand['names'] = dict(base['names'])
    cand['symbols'] = copy.deepcopy(base['symbols'])
    for p, newp in pmap.items():
        key = f'{src}:{inv_law[p]}'
        cand['names'][key] = newp
        cand['symbols'][key] = {**law_tr['symbols'][inv_law[p]], 'kind': 'Parameter', 'from': src}
    used = {str(s) for B in laws if 'rhs' in B for s in B['rhs'].free_symbols}
    params = sorted((set(base['parameters']) & used) | set(pmap.values()))
    computed = {B['target'] for B in laws}
    inputs = sorted(set(base['inputs']) & used)
    structure = symbolic.check_model([B for B in laws if 'rhs' in B], set(inputs), set(params))
    positivity = symbolic.check_positivity(laws, set(inputs), set(params)) if not structure['problems'] else \
        {'ok': True, 'problems': [], 'examples': {}}
    cand.update({'laws': laws, 'parameters': params, 'inputs': inputs, 'structure': structure,
                 'positivity': positivity, 'new_parameters': pmap, 'inferred': dims['inferred'],
                 'problems': dims['problems'] + structure['problems'] + positivity['problems']})
    cand['ok'] = not cand['problems']
    return cand


def start_values(cand: dict) -> dict:
    """Paper value converted to the model's units when possible, else 1.0."""
    inv = {v: k for k, v in cand['names'].items()}
    model_units = {symbolic.show(symbolic.dim_of(S.get('unit')) or {}): S.get('unit')
                   for S in cand['symbols'].values()
                   if 'from' not in S and S.get('unit') and symbolic.dim_of(S.get('unit')) is not None}   # base only
    out = {}
    for p in cand['parameters']:
        S = cand['symbols'].get(inv.get(p), {})
        val, unit = S.get('value'), S.get('unit')
        guess = 1.0
        try:
            if val not in (None, '') and unit:
                target_unit = model_units.get(symbolic.show(symbolic.dim_of(unit) or {}))
                if target_unit:
                    guess = float(U.Quantity(float(val), unit).to(target_unit).magnitude)
        except Exception:
            pass
        out[p] = guess
    return out


def report(cand: dict) -> str:
    lines = [f"{cand['label']}: {'all checks passed' if cand['ok'] else 'PROBLEMS'}"]
    lines += [f"   ! {p}" for p in cand['problems']]
    for B in cand.get('laws', []):
        if B['eq'] == cand['laws'][-1]['eq']:
            lines.append(f"   new law ({B['eq']}): {B['target']} = {B['rhs_text']}")
            lines += [f"      unit inferred: {s} [{u}]" for s, u in B['inferred'].items()]
    if cand.get('parameters'):
        lines.append(f"   parameters: {cand['parameters']}")
    return '\n'.join(lines)


if __name__ == '__main__':
    base = translate.translate('asprey_2000_monod_kinetics')
    KG = 'http://example.org/kg/'
    demos = [(KG + 'zentou_2019_andrews/law/zentou_2019_eq5', {'mu': 'r', 'S_': 'x2'}),
             (KG + 'muloiwa_2020_free_laws/law/muloiwa_2020_eq7', {'mu': 'r', 'S_': 'x2'})]
    for iri, mp in demos:
        c = swap(base, iri, mp)
        print(report(c))
        if c['ok']:
            print('   start values:', start_values(c))
