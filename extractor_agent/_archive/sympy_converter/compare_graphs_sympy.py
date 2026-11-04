"""compare_graphs.py -- pipeline Sup graph vs reference Sup graph (built from gold_laws.json).

Compares what a person would check: per law, the formula and the computes edge; overall,
the Variable and Parameter sets. Usage: python -m stages.compare_graphs papers/kontoravdi_2010
"""
import json, sys
from rdflib import Graph, Namespace, RDF
MKG = Namespace('http://example.org/mkg#')

def per_law(g):
    out = {}
    for L in g.subjects(RDF.type, MKG.Law):
        n = int(str(g.value(L, MKG.equationNumber)))
        comp = g.value(L, MKG.computes)
        out[n] = {'formula': str(g.value(L, MKG.formula) or ''),
                  'computes': str(g.value(comp, MKG.symbol)) if comp else None}
    return out

def node_set(g, cls):
    return {str(g.value(x, MKG.symbol)) for x in g.subjects(RDF.type, cls)}

def main(d):
    P, R = Graph().parse(f'{d}/sup_graph_pipeline.ttl'), Graph().parse(f'{d}/sup_graph_reference.ttl')
    lp, lr = per_law(P), per_law(R)
    rows, same_f, same_c = [], 0, 0
    for n in sorted(lr):
        f_ok = lp.get(n, {}).get('formula') == lr[n]['formula']
        c_ok = lp.get(n, {}).get('computes') == lr[n]['computes']
        same_f += f_ok; same_c += c_ok
        if not (f_ok and c_ok):
            rows.append({'eq': n, 'formula_match': f_ok, 'computes_pipeline': lp.get(n, {}).get('computes'),
                         'computes_reference': lr[n]['computes'],
                         'formula_pipeline': lp.get(n, {}).get('formula'), 'formula_reference': lr[n]['formula']})
    res = {'laws': len(lr), 'formula_identical': same_f, 'computes_identical': same_c, 'differences': rows}
    for cls in ('Variable', 'Parameter'):
        sp_, sr = node_set(P, MKG[cls]), node_set(R, MKG[cls])
        res[cls] = {'pipeline': len(sp_), 'reference': len(sr),
                    'only_in_pipeline': sorted(sp_ - sr), 'only_in_reference': sorted(sr - sp_)}
    json.dump(res, open(f'{d}/graph_comparison.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == '__main__':
    main(sys.argv[1])
