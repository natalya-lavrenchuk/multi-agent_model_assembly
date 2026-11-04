"""kg_query.py -- the fixed queries ("skills") the agents use to read the knowledge graph. Code only, no LLM.

    python kg_query.py model asprey_2000_monod_kinetics     # what get_model returns
    python kg_query.py rates                                 # what find_rate_laws returns

get_model(sup)        every law of one published model (Sup) with its LaTeX, form (ODE / algebraic / PDE / DAE,
                      written by the extractor's code), description, roles, what it computes and uses, and every
                      symbol with description, unit, dimension and the paper's value. Within a Sup, nodes with
                      the same symbol are the same quantity (KG rule 2), so they are merged into one entry.
get_law(iri)          one law in the same shape (a "model" with one law), for laws taken out of their paper.
find_rate_laws(dim)   every law, in any Sup or free-floating (Muloiwa), that computes a quantity of dimension
                      dim (default per time: growth, uptake, death rates ...), with what it computes and its roles.
                      The agents choose among them; code does not judge relevance.

LaTeX: when a misprint correction or a corrected OCR reading was accepted in review, the corrected form is
returned; the printed form stays available as 'latex_printed'.
"""
from __future__ import annotations
import json
import os
import re
import sys
from functools import lru_cache

import rdflib

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.environ.get('KG_GRAPH', os.path.join(HERE, '..', 'extractor_agent', 'kg', 'shared_graph.ttl'))
KG = 'http://example.org/kg/'
M = rdflib.Namespace('http://example.org/mkg#')
PREFIX = 'PREFIX mkg: <http://example.org/mkg#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n'
FIELDS = ('description', 'unit', 'unitString', 'dimension', 'value', 'initialValue', 'basis')


@lru_cache(maxsize=1)
def graph() -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(GRAPH)
    return g


def sup_iri(sup: str) -> str:
    return f'{KG}{sup}/{sup}'


def _clean(latex: str) -> str:
    return re.sub(r'^\s*\$\$|\$\$\s*$', '', latex).strip()


def _law_info(law: rdflib.URIRef) -> dict:
    """LaTeX (corrected when accepted), form, description and roles of one law."""
    g = graph()
    meta = g.value(law, M.hasMetadata)
    printed = str(g.value(law, M.latex))
    corrected = g.value(law, M.latexCorrected)
    accepted = meta is not None and str(g.value(meta, M.correctionStatus)) == 'accepted'
    roles = {}
    if meta is not None:
        for r in g.objects(meta, M.hasRole):
            node = g.value(r, M.boundTo)
            if node is not None:
                roles[str(g.value(r, M.slot))] = str(g.value(node, M.symbol))
    return {'iri': str(law), 'eq': str(g.value(law, M.equationNumber)),
            'latex': _clean(str(corrected)) if (corrected is not None and accepted) else printed,
            'latex_printed': printed, 'corrected': bool(corrected is not None and accepted),
            'form': str(g.value(meta, M.form)) if meta is not None and g.value(meta, M.form) else None,
            'description': str(g.value(meta, M.description) or '') if meta is not None else '',
            'roles': roles, 'computes': None, 'uses': [], 'parameters': []}


def symbols_in(latex: str, candidates: list) -> set:
    """Which of the printed symbols occur in a LaTeX string, longest first, so that the s inside K_{s} is not
    taken for a symbol s. The graph links a law to the symbols of its PRINTED form (a correction keeps them, by
    design of the extractor); when a corrected form drops one (Muloiwa 2020 eqs 6 and 12: s -> S), the law as
    used must not ask for it."""
    import re

    def norm(t: str) -> str:                     # notation only: \mathrm{n m} -> nm, \upmu -> \mu, \_ -> _
        t = re.sub(r'\\(mathrm|mathbf|boldsymbol|pmb|mathit|text|operatorname)\s*', '', t)
        t = re.sub(r'\\up([a-z]+)', r'\\\1', t).replace('\\_', '_')
        t = re.sub(r'\{([A-Za-z0-9_ ]+)\}', lambda m: '{' + m.group(1).replace(' ', '') + '}', t)
        while True:                              # {{x}} -> {x}, and a single letter in braces after a wrapper
            u = re.sub(r'\{\{([^{}]*)\}\}', r'{\1}', t)
            if u == t:
                return t
            t = u
    rest, found = norm(latex), set()
    for sym in sorted({c for c in candidates if c}, key=len, reverse=True):
        token = norm(sym)
        pat = ((r'(?<![A-Za-z\\])' if token[0].isalpha() else '') + re.escape(token) +
               (r'(?![A-Za-z])' if token[-1].isalpha() else ''))
        if re.search(pat, rest):
            found.add(sym)
            rest = re.sub(pat, '#', rest)
    return found


def _collect(law_iris: list) -> tuple[list, dict]:
    """Laws with their symbols; symbols merged by printed symbol."""
    g = graph()
    laws, symbols = [], {}
    for iri in law_iris:
        law = rdflib.URIRef(iri)
        L = _law_info(law)
        for rel, key in ((M.computes, 'computes'), (M.uses, 'uses'), (M.hasParameter, 'parameters')):
            for node in g.objects(law, rel):
                sym = str(g.value(node, M.symbol))
                if key == 'computes':
                    L['computes'] = sym
                elif sym not in L[key]:
                    L[key].append(sym)
                kind = 'Parameter' if (node, rdflib.RDF.type, M.Parameter) in g else 'Variable'
                S = symbols.setdefault(sym, {'kind': kind, 'laws': []})
                S['laws'].append(L['eq'])
                for f in FIELDS:
                    v = g.value(node, M[f])
                    if v is not None:
                        if f in S and S[f] != str(v):           # same symbol, different facts in two laws
                            S.setdefault('conflicts', {})[f] = sorted({S[f], str(v)})
                        S.setdefault(f, str(v))
        if L['corrected']:                     # the law is used in its corrected form: keep the symbols it has
            keep = symbols_in(L['latex'], [L['computes'], *L['uses'], *L['parameters']])
            L['dropped_by_correction'] = [x for x in L['uses'] + L['parameters'] if x not in keep]
            L['uses'] = [x for x in L['uses'] if x in keep]
            L['parameters'] = [x for x in L['parameters'] if x in keep]
        laws.append(L)
    order = lambda e: [float(x) if x.replace('.', '').isdigit() else 0 for x in e.split('.')]
    return sorted(laws, key=lambda L: order(L['eq'])), symbols


def get_model(sup: str) -> dict:
    """{'sup', 'label', 'laws': [...], 'symbols': {symbol: {...}}}"""
    g = graph()
    label = g.value(rdflib.URIRef(sup_iri(sup)), rdflib.RDFS.label)
    if label is None:
        raise KeyError(f'no Sup {sup} in the graph')
    laws, symbols = _collect([str(l) for l in g.objects(rdflib.URIRef(sup_iri(sup)), M.includes)])
    return {'sup': sup, 'label': str(label), 'laws': laws, 'symbols': symbols}


def source_of(law_iri: str) -> str:
    """'zentou_2019_andrews' or 'muloiwa_2020_free_laws': the folder part of the law's address."""
    return law_iri.replace(KG, '').split('/')[0]


def get_law(law_iri: str) -> dict:
    laws, symbols = _collect([law_iri])
    L = laws[0]
    return {'sup': f"{source_of(law_iri)}_eq{L['eq']}", 'label': L['description'], 'laws': laws, 'symbols': symbols}


def find_rate_laws(dimension: str = '1 / [time]') -> list:
    g = graph()
    rows = g.query(PREFIX + f"""
        SELECT DISTINCT ?law ?sym ?vdesc WHERE {{
          ?law a mkg:Law ; mkg:computes ?v .
          ?v mkg:symbol ?sym ; mkg:dimension "{dimension}" .
          OPTIONAL {{ ?v mkg:description ?vdesc }}
        }}""")
    out = []
    for law, sym, vdesc in rows:
        info = _law_info(law)
        sups = [str(s).split('/')[-1] for s in g.subjects(M.includes, law)]
        out.append({'iri': str(law), 'source': source_of(str(law)), 'in_models': sups, 'eq': info['eq'],
                    'computes': str(sym), 'computes_description': str(vdesc or ''),
                    'description': info['description'], 'roles': info['roles'], 'latex': info['latex'],
                    'corrected': info['corrected']})
    return sorted(out, key=lambda r: (r['source'], r['eq']))


if __name__ == '__main__':
    if sys.argv[1] == 'rates':
        for r in find_rate_laws():
            print(f"{r['source'][:30]:30} eq {r['eq']:5} {r['computes'][:10]:10} {r['computes_description'][:45]:45} "
                  f"roles {sorted(r['roles'])}")
    else:
        print(json.dumps(get_model(sys.argv[2]), indent=1, ensure_ascii=False))
