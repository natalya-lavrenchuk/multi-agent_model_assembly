"""
mkg_ids.py -- the ONE place that decides what a node in a paper graph is called.

Pass 1 and Pass 2 used to build IRIs independently: Pass 1 keyed quantities on
the run's local_id (c1, c2 -- different every run), Pass 2 keyed laws on a
counter that restarted per run (L1 for both Monod eq 3 and eq 4, so loading
both files merged two equations into one node). Nothing tied a law to the
quantities it uses except a string.

Rules:
  - A doc_id is "<DOI>#<section>". The DOCUMENT is the paper (the DOI). The
    section is part of a quantity's identity, because Pass 1 runs per section
    and a symbol means one thing within a section. Ks in the Monod section and
    Ks in the Contois section are two nodes. Merging them, or not, is the
    Assembler's call.
  - A quantity is keyed on (paper, section, symbol), never on a run id, so
    every run and both passes produce the same IRI for the same symbol.
  - A law is keyed on (paper, equation block). Blocks are unique in a paper.
  - In a paper graph, the symbols INSIDE a formula are formula_name(hasSymbol)
    of the quantities the law uses. That is what links formula to nodes.
"""

from __future__ import annotations

import re
import unicodedata

PREFIXES = (
    "@prefix mkg:  <http://example.org/mkg#> .\n"
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
    "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .\n"
)

_NON_ID = re.compile(r"[^A-Za-z0-9_]")

# Greek letters by Unicode name. Without this, [^A-Za-z0-9_] -> "_" turns
# "μ" into "_" and "μmax" into "_max": two different quantities, one of them
# with no name at all.
_GREEK_NAMES = {
    "ALPHA": "alpha", "BETA": "beta", "GAMMA": "gamma", "DELTA": "delta",
    "EPSILON": "epsilon", "ZETA": "zeta", "ETA": "eta", "THETA": "theta",
    "IOTA": "iota", "KAPPA": "kappa", "LAMDA": "lambda", "LAMBDA": "lambda",
    "MU": "mu", "NU": "nu", "XI": "xi", "OMICRON": "omicron", "PI": "pi",
    "RHO": "rho", "SIGMA": "sigma", "FINAL SIGMA": "sigma", "TAU": "tau",
    "UPSILON": "upsilon", "PHI": "phi", "CHI": "chi", "PSI": "psi",
    "OMEGA": "omega",
}


def _translit(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return ch
    for prefix, case in (("GREEK SMALL LETTER ", str.lower),
                         ("GREEK CAPITAL LETTER ", str.capitalize)):
        if name.startswith(prefix):
            key = name[len(prefix):]
            if key in _GREEK_NAMES:
                return case(_GREEK_NAMES[key])
    return ch


def _ident(s: str) -> str:
    """ASCII identifier. Case preserved: S and s are different symbols."""
    s = unicodedata.normalize("NFKC", (s or "").strip())
    s = "".join(_translit(c) for c in s)
    s = _NON_ID.sub("_", s)
    return s if s and not s[0].isdigit() else "x_" + s


def split_doc_id(doc_id: str) -> tuple[str, str]:
    paper, _, section = doc_id.partition("#")
    return paper.strip(), (section.strip() or "main")


def paper_doi(doc_id: str) -> str:
    return split_doc_id(doc_id)[0]


def doc_iri(doc_id: str) -> str:
    return f"mkg:doc_{_ident(paper_doi(doc_id))}"


def formula_name(symbol: str) -> str:
    """The name a quantity has inside a paper-graph formula.
    'μmax' -> 'mumax', 'Ks' -> 'Ks', 'K_s' -> 'K_s'."""
    return _ident(symbol)


def quantity_iri(doc_id: str, symbol: str, dup: int = 0) -> str:
    paper, section = split_doc_id(doc_id)
    tail = f"_{dup + 1}" if dup else ""
    return f"mkg:{_ident(paper)}__{_ident(section)}__q_{formula_name(symbol)}{tail}"


def law_iri(doc_id: str, block_id: str | None, local_id: str) -> str:
    paper, section = split_doc_id(doc_id)
    if block_id:
        return f"mkg:{_ident(paper)}__eq_{_ident(block_id.strip('/'))}"
    # no block id: fall back to section + local id, and say so in the IRI
    return f"mkg:{_ident(paper)}__{_ident(section)}__nolocator_{_ident(local_id)}"


def check_formula_names(symbols) -> dict[str, list[str]]:
    """Symbols that would collapse to the same formula name. Must be empty
    before a formula is renamed, or two quantities become one."""
    seen: dict[str, list[str]] = {}
    for s in symbols:
        seen.setdefault(formula_name(s), []).append(s)
    return {k: v for k, v in seen.items() if len(v) > 1}
