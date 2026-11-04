"""links.py -- described links between related Sups (the linking step of the KG Assembler).

Which pairs: for every Sup, its 5 nearest Sups from other papers on the Sup map (embedding similarity,
supmap.py), plus every pair of models from the same paper, plus the hand-kept pairs.
Per pair, one LLM call reads both Sups' names, descriptions and facts and answers: related (yes/no) and a
short free paragraph on how the two models relate, what they share and how they differ, naming the models
(e.g. "Kontoravdi 2010", "Seidel 2018 (detailed kinetic model)"). No fixed relation labels: the text is
read by people and by GraphRAG-style queries. Nothing is reviewed: the text goes into the graph with the
model and date that wrote it.

The finished pairs are kept in kg/links.json, so a build only asks the LLM about pairs that are new or whose
Sups changed (their name, description or facts). A changed prompt never re-asks old pairs; --redo does.
Hand-kept links (kg/link_overrides.json: a, b as Sup ids, related, description, provenance) win over both.

cites (code, no LLM): Sup A cites Sup B when A's facts name B's first author and year.

In the graph, per related pair:
    A mkg:hasLink L, B mkg:hasLink L     L = .../kg/link/<a>__<b>, rdfs:label "Name A ~ Name B",
                                         mkg:linkDescription, mkg:similarity, mkg:provenance
    A mkg:cites B

    python -m extractor.links --dry-run     the pairs it would ask about and the tokens, no call
    python -m extractor.links --redo        ask every pair again (normally it runs inside "Build shared KG")
"""
from __future__ import annotations
import datetime
import hashlib
import json
import os
import re
import sys

from rdflib import Graph, Literal, RDFS, URIRef
from rdflib.namespace import XSD

from extractor import llm, supmap
from extractor.supmap import KG_DIR, MKG

LINK_NS = 'http://example.org/kg/link/'
STORE = os.path.join(KG_DIR, 'links.json')
OVERRIDES = os.path.join(KG_DIR, 'link_overrides.json')
PRETTY = {'vandenbussche': 'Vanden Bussche', 'lopezmeza': 'López-Meza'}
WORDS = {'monod': 'Monod', 'contois': 'Contois', 'andrews': 'Andrews', 'sma': 'SMA'}

SYSTEM = """You compare two published process models (each one "Sup") for a knowledge graph of mechanistic models
from all of chemical engineering. You get each model's name, description and facts, as extracted from its paper.

Answer:
- related: true if a modeller would find the two worth comparing (e.g. rival model structures for the same process,
  one building on, simplifying or extending the other, the same model in another operating mode, the same phenomenon
  or modelling approach in another system or field, the same unit operation with another product or conditions).
  false if they only share general words (both are "dynamic models", both "estimate parameters").
- description: if related, a short paragraph (3 to 6 sentences, at most 120 words) saying how the two models relate,
  what they share (system, phenomena, operating mode, quantities modelled, purpose) and how they differ. Refer to the
  models by the names given, never as "A" and "B". Several relations can hold at once (e.g. rival models where one
  also builds on the other): say all that apply. If not related, one sentence saying why.
Use only what the descriptions and facts say. Be specific and plain; do not guess details they do not give."""

SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['related', 'description'],
          'properties': {'related': {'type': 'boolean'}, 'description': {'type': 'string'}}}


def sid(u: str) -> str:
    return u.split('/')[-2]


def names(uris) -> dict[str, str]:
    """Sup URI -> 'Author Year', plus the model when the paper has several ('Seidel 2018 (detailed kinetic model)')."""
    parts = {u: re.match(r'([a-z]+)_(\d{4})_?(.*)', sid(u)) for u in uris}
    per_paper = {}
    for u, m in parts.items():
        if m:
            per_paper[m.group(1, 2)] = per_paper.get(m.group(1, 2), 0) + 1
    out = {}
    for u, m in parts.items():
        if not m:
            out[u] = sid(u)
            continue
        a, y, rest = m.groups()
        n = f"{PRETTY.get(a, a.capitalize())} {y}"
        if per_paper[(a, y)] > 1 and rest:
            n += ' (' + ' '.join(WORDS.get(w, w) for w in rest.split('_')) + ')'
        out[u] = n
    return out


def pairs(data: dict) -> dict[tuple[str, str], float]:
    """Unordered Sup pairs to describe, with their similarity."""
    out = {}
    for u, s in data['sups'].items():
        for n in s['near_other_papers'] + s['same_paper']:
            out[tuple(sorted((u, n['sup'])))] = n['similarity']
    return out


def text_hash(sups: dict, a: str, b: str) -> str:
    return hashlib.sha256((sups[a]['text'] + '\n\n' + sups[b]['text']).encode()).hexdigest()[:16]


def describe(sups: dict, nm: dict, a: str, b: str) -> str:
    return '\n\n'.join(f"Model \"{nm[u]}\":\n{sups[u]['text']}" for u in (a, b))


def cites(g: Graph, sups: dict) -> list[tuple[str, str]]:
    """(A, B) when A's facts name B's first author near B's year (the year in B's Sup id)."""
    years = {}
    for u in sups:
        m = re.match(r'([a-z]+)_(\d{4})', sid(u))
        if m:
            years.setdefault(m.group(1), {})[u] = m.group(2)
    out = []
    for a in sups:
        meta = g.value(URIRef(a), MKG.hasMetadata)
        facts = ' '.join(str(g.value(f, MKG.text) or '') for f in g.objects(meta, MKG.hasFact)) if meta else ''
        for author, targets in years.items():
            for b, y in targets.items():
                if sups[b]['paper'] != sups[a]['paper'] and \
                        re.search(rf'\b{author}\b[^.;]{{0,60}}?\b{y}\b', facts, re.I):
                    out.append((a, b))
    return out


def build(g: Graph, data: dict, redo: bool = False) -> tuple[Graph, dict]:
    sups = supmap.sup_texts(g)
    nm = names(sups)
    by_id = {sid(u): u for u in sups}
    store = json.load(open(STORE)).get('pairs', {}) if os.path.exists(STORE) and not redo else {}
    hand = {}
    for o in (json.load(open(OVERRIDES)).get('pairs', []) if os.path.exists(OVERRIDES) else []):
        if o.get('a') in by_id and o.get('b') in by_id:
            hand[tuple(sorted((by_id[o['a']], by_id[o['b']])))] = o
    todo = pairs(data)
    for k in hand:
        todo.setdefault(k, 0.0)                           # a hand link may join a pair the map did not propose
    made = f"written by {llm.MODEL} from the two Sups' descriptions and facts, {datetime.date.today().isoformat()}"
    out = Graph()
    out.bind('mkg', MKG)
    report = {'pairs': 0, 'linked': 0, 'not_related': 0, 'asked': 0, 'kept': 0, 'by_hand': 0, 'cites': []}
    for (a, b), sim in sorted(todo.items()):
        if a not in sups or b not in sups:
            continue
        report['pairs'] += 1
        key, h = f'{sid(a)}|{sid(b)}', text_hash(sups, a, b)
        if (a, b) in hand:
            o = hand[(a, b)]
            ans = {'related': o.get('related', True), 'description': o.get('description', ''),
                   'provenance': o.get('provenance') or 'written by hand'}
            report['by_hand'] += 1
        elif key in store and store[key].get('texts') == h:   # finished before, Sups unchanged
            ans = store[key]
            report['kept'] += 1
        else:
            r = llm.cached_ask(KG_DIR, SYSTEM, describe(sups, nm, a, b), SCHEMA, 'link_named', 'high',
                               label=f"{nm[a]} ~ {nm[b]}")
            if not r:                                     # dry run
                continue
            ans = {**r, 'provenance': made, 'texts': h}
            store[key] = ans
            report['asked'] += 1
        if not ans['related']:
            report['not_related'] += 1
            continue
        report['linked'] += 1
        L = URIRef(f"{LINK_NS}{sid(a)}__{sid(b)}")
        out.add((URIRef(a), MKG.hasLink, L))
        out.add((URIRef(b), MKG.hasLink, L))
        out.add((L, RDFS.label, Literal(f"{nm[a]} ~ {nm[b]}")))
        out.add((L, MKG.linkDescription, Literal(ans['description'])))
        out.add((L, MKG.similarity, Literal(sim, datatype=XSD.decimal)))
        out.add((L, MKG.provenance, Literal(ans.get('provenance') or made)))
    for a, b in cites(g, sups):
        out.add((URIRef(a), MKG.cites, URIRef(b)))
        report['cites'].append([sid(a), sid(b)])
    if not llm.DRY_RUN:
        json.dump({'note': 'Finished links, one per Sup pair ("a|b" Sup ids). texts = hash of both Sups\' texts: '
                           'when either Sup changes, the pair is asked again.', 'pairs': store},
                  open(STORE, 'w'), indent=1, ensure_ascii=False)
    return out, report


def main(dry: bool, redo: bool):
    llm.DRY_RUN = dry
    g = Graph().parse(os.path.join(KG_DIR, 'shared_graph.ttl'))
    _, data = supmap.build(g)                           # embeddings and group names come from the cache
    out, report = build(g, data, redo)
    print(llm.usage_line())
    print({k: (v if isinstance(v, int) else len(v)) for k, v in report.items()})
    if not dry:
        out.serialize(os.path.join(KG_DIR, 'sup_links.ttl'), format='turtle')
        json.dump(report, open(os.path.join(KG_DIR, 'sup_links.json'), 'w'), indent=1)


if __name__ == '__main__':
    main('--dry-run' in sys.argv, '--redo' in sys.argv)
