"""
validate_paper_graph.py -- does a paper graph conform to the schema?

Exists because the schema and the extractor output drifted apart without
anyone noticing: Pass 2 wrote 13 terms the schema never defined. Run this after
every change to either. No LLM, no tokens.

Checks
  1. every mkg: class and property used is defined in the schema
  2. every law's computes / usesQuantity points at a typed Quantity node
  3. every ACCEPTED law's formula symbols are exactly the formula names of the
     quantities it computes and uses: formula and graph say the same thing
  4. no law node carries two formulas (the old L1 collision)
  5. every accepted node that can be wrong has evidence: its own
     hasJustification, or (in the shared graph) a paper node merged into it
  6. every role binding points at a Role, and every C[...] index in an
     accepted formula is a symbol the law binds to a role

Works on paper graphs and on the shared graph (indexed formulas, C[S]).

Usage:
  python validate_paper_graph.py mkg_schema.ttl paper/*.ttl
"""

from __future__ import annotations

import re
import sys

import sympy
from rdflib import Graph, Namespace, RDF, RDFS
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from mkg_ids import formula_name

MKG = Namespace("http://example.org/mkg#")

_FUNCS = {"exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt,
          "Eq": sympy.Eq, "Derivative": sympy.Derivative, "Abs": sympy.Abs,
          "Integer": sympy.Integer, "Float": sympy.Float,
          "Rational": sympy.Rational}
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def formula_symbols(s: str) -> tuple[set[str], set[str]]:
    """(quantity symbols, index symbols). A name followed by '[' is an
    indexed quantity (sympy.IndexedBase); the names inside the brackets are
    role symbols, resolved through the law's role bindings."""
    names = {n for n in _ID.findall(s) if n not in _FUNCS}
    bases = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\[", s))
    idx = {i.strip() for grp in re.findall(r"\[([^\]]+)\]", s) for i in grp.split(",")}
    local = {n: (sympy.IndexedBase(n) if n in bases else sympy.Symbol(n)) for n in names}
    expr = parse_expr(s, local_dict=local, global_dict=dict(_FUNCS),
                      transformations=standard_transformations)
    quantities = {x.name for x in expr.free_symbols
                  if isinstance(x, sympy.Symbol) and x.name not in idx}
    quantities |= {b.name for b in expr.atoms(sympy.IndexedBase)}
    return quantities, idx


def main(schema_path, paper_paths):
    schema = Graph().parse(schema_path)
    g = Graph()
    for p in paper_paths:
        g.parse(p)
    problems = []

    defined = {s for s in schema.subjects() if str(s).startswith(str(MKG))}
    used = {p for p in g.predicates() if str(p).startswith(str(MKG))}
    used |= {o for o in g.objects(None, RDF.type) if str(o).startswith(str(MKG))}
    for u in sorted(used - defined):
        problems.append(f"[1] not in schema: {u.split('#')[1]}")

    quantity_classes = set(schema.transitive_subjects(RDFS.subClassOf, MKG.Quantity))

    def is_quantity(node):
        return any(t in quantity_classes for t in g.objects(node, RDF.type))

    for law in set(g.subjects(MKG.hasFormula, None)):
        name = law.split("#")[1]
        formulas = list(g.objects(law, MKG.hasFormula))
        if len(formulas) > 1:
            problems.append(f"[4] {name}: {len(formulas)} formulas on one node")
            continue

        linked = list(g.objects(law, MKG.computes)) + list(g.objects(law, MKG.usesQuantity))
        for q in linked:
            if not is_quantity(q):
                problems.append(f"[2] {name}: points at {q.split('#')[1]}, "
                                f"which is not a typed Quantity in these files")

        if (law, MKG.hasStatus, MKG.accepted) not in g:
            continue
        expected = {formula_name(str(s)) for q in linked
                    for s in g.objects(q, MKG.hasSymbol)}
        got, idx = formula_symbols(str(formulas[0]))
        if got != expected:
            problems.append(f"[3] {name}: formula has {sorted(got)}, "
                            f"graph links {sorted(expected)}")
        bound = {str(s) for rb in g.objects(law, MKG.hasRoleSlot)
                 for s in g.objects(rb, MKG.bindsSymbol)}
        if idx != bound:
            problems.append(f"[6] {name}: formula indexes {sorted(idx)}, "
                            f"role bindings cover {sorted(bound)}")

    for rb in set(g.subjects(RDF.type, MKG.RoleBinding)):
        for r in g.objects(rb, MKG.bindsRole):
            if (r, RDF.type, MKG.Role) not in g:
                problems.append(f"[6] {rb.split('#')[1]}: bindsRole {r.split('#')[1]} is not a Role")

    needs_evidence = quantity_classes | set(schema.transitive_subjects(RDFS.subClassOf, MKG.Law)) \
        | {MKG.RoleBinding, MKG.ScopeCondition}
    for node in set(g.subjects(MKG.hasStatus, MKG.accepted)):
        if not any(t in needs_evidence for t in g.objects(node, RDF.type)):
            continue
        if not (list(g.objects(node, MKG.hasJustification))
                or list(g.subjects(MKG.mergedInto, node))):
            problems.append(f"[5] {node.split('#')[1]}: accepted with no evidence")

    print(f"{len(g)} triples, {len(set(g.subjects(MKG.hasFormula, None)))} laws")
    print("\n".join(problems) if problems else "OK: conforms")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
