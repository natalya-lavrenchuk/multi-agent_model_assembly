"""build_graph.py -- turn one paper's extracted equations into a Sup subgraph (Turtle).

Follows schema/kg_schema.ttl: Sup, Law, Variable, Parameter, Metadata. No LLM anywhere.

WHAT GOES IN (and from where)
    Law        one per printed equation (reader output, e.g. paddle.json)
               formula        SymPy string from stages/latex_to_sympy.py
               computes/uses  from stages/structure_check.py (direct or by elimination)
    Metadata   per Law: equation number, source (DOI, page, the LaTeX as read = the quote),
               form (ODE / algebraic, computed by SymPy), side conditions printed next to the
               equation (e.g. 'n > 1'), status (pending, or flagged by a structural check)
               per Sup: source, form (DAE when ODEs and algebraic laws mix)
    Variable   every symbol a law computes, plus never-computed symbols with no value in any
               table (candidate inputs: feeds, inlet concentrations)
    Parameter  never-computed symbols that have a numeric value in a table row of the paper
               (code-only evidence, see stages/candidates.py). fixed=false for all: no
               universal constants in this model.

WHAT IS NOT IN YET (deliberately)
    meanings and units of symbols (variable step postponed), assumptions in prose, roles,
    law type (balance / constitutive / definition), parameter values.

Symbols are named as the equation reader wrote them, converted to identifiers
(K_glc, C_GLC for [GLC], mu_max). One node per symbol per Sup: every law that uses mu
points at the same mu node.

Usage (from extractor_agent/):
    python -m stages.build_graph papers/kontoravdi_2010 --sup kontoravdi_2005 \
        --doi 10.1016/j.compchemeng.2010.03.012 --title "Kontoravdi et al. (2005) CHO growth and mAb model"
    add --gold to build the reference graph from gold_laws.json instead of the reader output
"""
from __future__ import annotations
import argparse, json, os, re

import sympy as sp
from rdflib import Graph, Namespace, Literal, BNode, RDF, RDFS, XSD

from stages.latex_to_sympy import parse
from stages.structure_check import assign, T, near
from stages.candidates import build_snippets, find_candidates, match_level, key, tokens

MKG = Namespace('http://example.org/mkg#')


def law_form(eq) -> str:
    derivs = eq.atoms(sp.Derivative)
    if not derivs:
        return 'algebraic'
    wrt = {v for d in derivs for v, _ in d.variable_count}
    return 'ODE' if wrt == {T} else 'PDE'


def load(paper_dir: str, gold: bool):
    if gold:
        g = json.load(open(f'{paper_dir}/gold_laws.json'))
        return [{'eq': r['eq'], 'latex': r['latex_printed'], 'page': 6} for r in g['laws']], 'gold_laws.json (hand-checked)'
    f = json.load(open(f'{paper_dir}/paddle.json'))
    return f['formulas'], f"paddle.json ({f.get('reader', 'reader')})"


def table_evidence(paper_dir: str, names):
    """symbol -> text of a table row that gives it a numeric value (or None). Code only."""
    path = f'{paper_dir}/paddle_blocks.json'
    if not os.path.exists(path):
        return {n: None for n in names}
    blocks = json.load(open(path))['blocks']
    snips = [s for s in build_snippets(blocks) if s.kind == 'table_row']
    out = {}
    for n in names:
        k = key(n)
        hit = None
        for s in snips:
            levels = [lv for t in tokens(s.text) if (lv := match_level(k, t))]
            if re.search(r'\|[^|]*\d', s.text) and levels:
                hit = (f'p.{s.page + 1}: {s.text[:90]}',
                       min(levels, key=['exact', 'notation', 'confusable', 'partial'].index))
                break
        out[n] = hit
    return out


def build(paper_dir, sup_id, doi, title, gold=False):
    formulas, source_name = load(paper_dir, gold)
    laws, conds, failed = {}, {}, []
    for i, f in enumerate(formulas):
        try:
            laws[i], conds[i] = parse(f['latex'])
        except Exception as ex:
            failed.append((f.get('eq', i + 1), str(ex)[:80]))
    asg = assign(laws)

    computed, used = {}, {}
    for i, (s, how, c) in asg.items():
        if s is not None:
            computed.setdefault(str(s), []).append(i)
    for i, e in laws.items():
        for s in e.free_symbols - {T}:
            if s != asg[i][0]:
                used.setdefault(str(s), []).append(i)
    names = sorted(set(computed) | set(used))
    tab = table_evidence(paper_dir, [n for n in names if n not in computed])

    # ---- flags (structural, code only) ----
    flags = []
    for i, (s, how, c) in asg.items():
        if s is None:
            flags.append({'law': formulas[i].get('eq'), 'check': 'computes',
                          'detail': f'no single variable computed ({how}); candidates {[str(x) for x in c]}'})
    for n, ls in computed.items():
        if len(ls) > 1:
            flags.append({'symbol': n, 'check': 'computed twice',
                          'detail': f"computed by eqs {[formulas[i].get('eq') for i in ls]}"})
    never_used = sorted(n for n in computed if n not in used)        # sorted: identical output on every run
    never_comp = sorted(n for n in used if n not in computed)
    for x in never_used:
        for y in never_comp:
            if near(x, y):
                flags.append({'symbol': f'{x} ~ {y}', 'check': 'near-name pair',
                              'detail': 'computed-but-unused vs used-but-uncomputed with similar names: possible misread'})
    inputs = [n for n in never_comp if not tab.get(n)]
    for n in never_comp:
        if tab.get(n) and tab[n][1] in ('partial', 'confusable'):
            flags.append({'symbol': n, 'check': 'split name?',
                          'detail': f"table names a longer symbol ({tab[n][0][:60]}...): the reader may have split one "
                                    f"symbol into two, so a neighbouring single letter may be wrongly shared with another law"})
    for n in inputs:
        flags.append({'symbol': n, 'check': 'unclassified',
                      'detail': f"never computed and no table value: an input (feed, inlet) or an undefined parameter? used in eqs {[formulas[i].get('eq') for i in used[n]]}"})
    for f_ in failed:
        flags.append({'law': f_[0], 'check': 'parse', 'detail': f_[1]})
    flagged_laws = {f['law'] for f in flags if 'law' in f}
    for f in flags:
        if f.get('check') == 'split name?':
            for i in used.get(f['symbol'], []):
                flagged_laws.add(formulas[i].get('eq'))
    for f in flags:
        if f.get('check') == 'near-name pair':
            for side in f['symbol'].split(' ~ '):
                for i in computed.get(side, []) + used.get(side, []):
                    flagged_laws.add(formulas[i].get('eq'))

    # ---- graph ----
    g = Graph()
    KG = Namespace(f'http://example.org/kg/{sup_id}/')
    g.bind('mkg', MKG); g.bind('kg', KG)
    sup = KG['sup']
    g.add((sup, RDF.type, MKG.Sup)); g.add((sup, RDFS.label, Literal(title)))
    sm = KG['sup/meta']
    g.add((sup, MKG.hasMetadata, sm)); g.add((sm, RDF.type, MKG.Metadata))
    g.add((sm, MKG.doi, Literal(doi)))
    forms = {law_form(e) for e in laws.values()}
    g.add((sm, MKG.form, Literal('DAE' if {'ODE', 'algebraic'} <= forms else '/'.join(sorted(forms)))))
    g.add((sm, MKG.status, Literal('pending')))
    g.add((sm, RDFS.comment, Literal(f'built from {source_name} by stages/build_graph.py')))

    def var(n):
        node = KG[f'var/{n}']
        if (node, RDF.type, MKG.Variable) not in g:
            g.add((node, RDF.type, MKG.Variable)); g.add((node, MKG.symbol, Literal(n)))
            g.add((node, MKG.formulaName, Literal(n)))
            if n in inputs:
                g.add((node, RDFS.comment, Literal('never computed, no table value: candidate input (unconfirmed)')))
        return node

    def par(n):
        node = KG[f'par/{n}']
        if (node, RDF.type, MKG.Parameter) not in g:
            g.add((node, RDF.type, MKG.Parameter)); g.add((node, MKG.symbol, Literal(n)))
            g.add((node, MKG.formulaName, Literal(n)))
            g.add((node, MKG.fixed, Literal(False)))
            g.add((node, RDFS.comment, Literal(f'Parameter by code-only evidence: never computed; table row {tab[n][0]} (match: {tab[n][1]})')))
        return node

    is_param = {n for n in never_comp if tab.get(n)}
    for i, f in enumerate(formulas):
        n_eq = f.get('eq') or i + 1
        L = KG[f'law/eq{n_eq}']
        g.add((sup, MKG.includes, L)); g.add((L, RDF.type, MKG.Law))
        g.add((L, MKG.equationNumber, Literal(str(n_eq))))
        M = KG[f'law/eq{n_eq}/meta']
        g.add((L, MKG.hasMetadata, M)); g.add((M, RDF.type, MKG.Metadata))
        g.add((M, MKG.doi, Literal(doi))); g.add((M, MKG.page, Literal(str(f.get('page', 0) + 1))))
        g.add((M, MKG.quote, Literal(f['latex'].strip().strip('$').strip())))
        g.add((M, MKG.status, Literal('flagged' if n_eq in flagged_laws else 'pending')))
        if i not in laws:
            continue
        e = laws[i]
        g.add((L, MKG.formula, Literal(sp.sstr(e))))
        g.add((M, MKG.form, Literal(law_form(e))))
        if conds.get(i):
            a = BNode()
            g.add((M, MKG.hasAssumption, a)); g.add((a, MKG.text, Literal(conds[i])))
            g.add((a, MKG.reasoned, Literal(False)))
            g.add((a, MKG.quote, Literal(conds[i])))
        s_comp = asg[i][0]
        if s_comp is not None:
            g.add((L, MKG.computes, var(str(s_comp))))
        for s in sorted(e.free_symbols - {T}, key=str):
            n = str(s)
            if s == s_comp:
                continue
            if n in is_param:
                g.add((L, MKG.hasParameter, par(n)))
            else:
                g.add((L, MKG.uses, var(n)))

    summary = {
        'source': source_name, 'laws': len(formulas), 'parsed': len(laws),
        'variables': len({o for o in g.subjects(RDF.type, MKG.Variable)}),
        'parameters': len({o for o in g.subjects(RDF.type, MKG.Parameter)}),
        'computes_edges': len(list(g.triples((None, MKG.computes, None)))),
        'uses_edges': len(list(g.triples((None, MKG.uses, None)))),
        'hasParameter_edges': len(list(g.triples((None, MKG.hasParameter, None)))),
        'laws_by_form': {fm: sum(1 for e in laws.values() if law_form(e) == fm) for fm in sorted(forms)},
        'laws_flagged': sorted(flagged_laws),
        'parameters_list': sorted(is_param), 'candidate_inputs': sorted(inputs),
        'triples': len(g),
    }
    return g, flags, summary


def validate(g: Graph, schema_path: str):
    """Every mkg: term used is defined in the schema; every edge connects the right classes."""
    S = Graph().parse(schema_path)
    defined = {str(s) for s in S.subjects()}
    problems = []
    for s, p, o in g:
        for term in (p, o):
            if str(term).startswith(str(MKG)) and str(term) not in defined:
                problems.append(f'undefined term {term}')
    rng = {MKG.includes: MKG.Law, MKG.computes: MKG.Variable, MKG.uses: MKG.Variable,
           MKG.hasParameter: MKG.Parameter, MKG.hasMetadata: MKG.Metadata}
    for p, cls in rng.items():
        for s, o in g.subject_objects(p):
            if (o, RDF.type, cls) not in g:
                problems.append(f'{p.split("#")[1]} points at a non-{cls.split("#")[1]}: {o}')
    for L in g.subjects(RDF.type, MKG.Law):
        n = len(list(g.objects(L, MKG.computes)))
        if n > 1:
            problems.append(f'{L} computes {n} variables')
    return sorted(set(problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paper_dir')
    ap.add_argument('--sup', required=True)
    ap.add_argument('--doi', required=True)
    ap.add_argument('--title', required=True)
    ap.add_argument('--gold', action='store_true')
    ap.add_argument('--schema', default='schema/kg_schema.ttl')
    a = ap.parse_args()
    g, flags, summary = build(a.paper_dir, a.sup, a.doi, a.title, a.gold)
    tag = 'reference' if a.gold else 'pipeline'
    out = f'{a.paper_dir}/sup_graph_{tag}.ttl'
    g.serialize(out, format='turtle')
    problems = validate(g, a.schema)
    json.dump({'summary': summary, 'flags': flags, 'schema_problems': problems},
              open(f'{a.paper_dir}/sup_graph_{tag}_report.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    print(f'\n{len(flags)} flags:')
    for f in flags:
        print('  ', f)
    print('\nschema validation:', 'OK' if not problems else problems)
    print('written ->', out)


if __name__ == '__main__':
    main()
