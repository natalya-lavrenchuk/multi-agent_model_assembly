"""assemble.py -- KG assembler, step 1: put several papers' Sup graphs into one shared graph.

    python -m extractor.assemble papers/kontoravdi_2010 papers/anane_2017

Reads each paper's sup_graph_pipeline.ttl (built by graph.py) and writes kg/shared_graph.ttl
plus kg/shared_graph_report.json. Nothing is merged or linked yet: every node keeps the IRI
of its own Sup (kg/<sup>/...), so joining graphs cannot collide. Linking equal concepts across
Sups (sameAs, shared Variables) is the assembler's next step, and the one that needs judgement.
"""
from __future__ import annotations
import json
import os
import sys

from rdflib import Graph, RDF, URIRef

from extractor.graph import MKG, validate

OUT = 'kg'


def assemble(paper_dirs: list[str]) -> tuple[Graph, dict]:
    shared = Graph()
    shared.bind('mkg', MKG)
    per_sup = {}
    for d in paper_dirs:
        g = Graph().parse(f'{d}/sup_graph_pipeline.ttl')
        for sup in g.subjects(RDF.type, MKG.Sup):
            ns = str(sup).rsplit('/', 1)[0] + '/'           # http://example.org/kg/<sup>/
            outside = {str(s) for s in g.subjects() if isinstance(s, URIRef) and not str(s).startswith(ns)}
            per_sup[str(sup)] = {'paper': d, 'laws': len(list(g.objects(sup, MKG.includes))),
                                 'variables': len(set(g.subjects(RDF.type, MKG.Variable))),
                                 'parameters': len(set(g.subjects(RDF.type, MKG.Parameter))),
                                 'triples': len(g), 'nodes_outside_own_namespace': sorted(outside)}
        shared += g
    report = {'sups': per_sup, 'triples': len(shared),
              'triples_sum_of_parts': sum(v['triples'] for v in per_sup.values()),
              'schema_problems': validate(shared)}
    return shared, report


def main(paper_dirs: list[str]):
    g, report = assemble(paper_dirs)
    os.makedirs(OUT, exist_ok=True)
    g.serialize(f'{OUT}/shared_graph.ttl', format='turtle')
    json.dump(report, open(f'{OUT}/shared_graph_report.json', 'w'), indent=1, ensure_ascii=False)
    for sup, v in report['sups'].items():
        print(f"{sup.split('/')[-2]:18} {v['laws']:3} laws  {v['variables']:3} Variable  {v['parameters']:3} Parameter")
    print(f"shared graph: {report['triples']} triples (sum of the parts: {report['triples_sum_of_parts']}), "
          f"{len(report['schema_problems'])} schema problems -> {OUT}/shared_graph.ttl")
    for p in report['schema_problems']:
        print('   ', p)


if __name__ == '__main__':
    main(sys.argv[1:])
