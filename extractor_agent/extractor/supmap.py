"""supmap.py -- a map of the Sups: which published models are close, which far apart.

Each Sup's text (its name, description and facts, read from the shared graph) becomes one embedding:
one call per Sup, saved in kg/embeddings/, so a rebuild only pays for new or changed Sups. Code then
compares every pair (cosine similarity), keeps each Sup's nearest neighbours, groups the Sups into
clusters (average linkage, the number of clusters chosen by silhouette) and lays them out in 2D
(classical MDS). One short LLM call per cluster names it (cached). Nothing here judges whether two
models are interchangeable: closeness is about what the papers say, not about the equations.

In the graph: a Group node per cluster (named) and Sup mkg:inGroup Group. The nearest Sups and their scores stay
in kg/sup_map.json (they choose the pairs links.py describes); the graph holds only the described links.

Writes kg/sup_map.json, kg/sup_map.ttl (added to the shared graph) and kg/sup_map.png.

    python -m extractor.supmap                 LLM_API_KEY set; LLM_EMBED_MODEL (default text-embedding-3-small)
    python -m extractor.supmap --offline       word counts instead of embeddings: to test the code, no API
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import urllib.request

import numpy as np
from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import XSD

from extractor import llm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG_DIR = os.path.join(ROOT, 'kg')
MKG = Namespace('http://example.org/mkg#')
EMBED_MODEL = os.environ.get('LLM_EMBED_MODEL', 'text-embedding-3-small')
NEIGHBOURS = 5
GROUP_NS = 'http://example.org/kg/map/'


def sup_texts(g: Graph) -> dict[str, dict]:
    """Sup URI -> its label and the text that is embedded (label, description, facts)."""
    out = {}
    for sup in sorted(g.subjects(RDF.type, MKG.Sup), key=str):
        label = str(g.value(sup, RDFS.label) or sup)
        meta = g.value(sup, MKG.hasMetadata)
        desc = str(g.value(meta, MKG.description) or '') if meta else ''
        facts = sorted(f"{g.value(f, MKG.topic) or ''}: {g.value(f, MKG.text) or ''}"
                       for f in (g.objects(meta, MKG.hasFact) if meta else []))
        sid = str(sup).split('/')[-2]
        out[str(sup)] = {'label': label, 'paper': (re.match(r'[a-z]+_\d{4}', sid) or [sid])[0],
                         'text': '\n'.join([label, desc, *facts]).strip()}
    return out


def embed(texts: list[str]) -> np.ndarray:
    """One embedding per text, cached on disk by model + text."""
    cache = os.path.join(KG_DIR, 'embeddings')
    os.makedirs(cache, exist_ok=True)
    out = []
    for t in texts:
        path = os.path.join(cache, hashlib.sha256(f'{EMBED_MODEL}\n{t}'.encode()).hexdigest()[:20] + '.json')
        if not os.path.exists(path):
            key = llm.api_key()
            if not key:
                raise SystemExit('set LLM_API_KEY first (or run with --offline to test)')
            url = llm.URL.rsplit('/chat/completions', 1)[0] + '/embeddings'
            req = urllib.request.Request(url, method='POST', data=json.dumps({'model': EMBED_MODEL, 'input': t}).encode(),
                                         headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}',
                                                  'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as r:
                reply = json.loads(r.read())
            json.dump({'model': EMBED_MODEL, 'tokens': reply.get('usage', {}).get('total_tokens'),
                       'embedding': reply['data'][0]['embedding']}, open(path, 'w'))
            print(f"  embedded {t.splitlines()[0][:60]} ({reply.get('usage', {}).get('total_tokens', '?')} tokens)")
        out.append(json.load(open(path))['embedding'])
    return np.array(out, dtype=float)


def word_counts(texts: list[str]) -> np.ndarray:
    """Offline stand-in for embeddings (tf-idf over words), only to test the rest of the code."""
    docs = [re.findall(r'[a-z]{3,}', t.lower()) for t in texts]
    vocab = sorted({w for d in docs for w in d})
    idx = {w: i for i, w in enumerate(vocab)}
    m = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for w in d:
            m[i, idx[w]] += 1
    df = (m > 0).sum(0)
    return m * np.log(len(docs) / df)


def similarity(v: np.ndarray) -> np.ndarray:
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return np.clip(v @ v.T, -1, 1)


def clusters(sim: np.ndarray) -> list[int]:
    """Average-linkage clustering on 1 - similarity; the number of clusters with the best silhouette."""
    n = len(sim)
    d = 1 - sim
    if n < 4:
        return [0] * n
    groups = [[i] for i in range(n)]
    history = {n: [g[:] for g in groups]}
    while len(groups) > 2:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                x = d[np.ix_(groups[a], groups[b])].mean()
                if best is None or x < best[0]:
                    best = (x, a, b)
        _, a, b = best
        groups[a] += groups.pop(b)
        history[len(groups)] = [g[:] for g in groups]

    def silhouette(gs):
        lab = np.empty(n, int)
        for k, g in enumerate(gs):
            lab[g] = k
        s = []
        for i in range(n):
            own = [j for j in gs[lab[i]] if j != i]
            if not own:
                s.append(0.0)
                continue
            a = d[i, own].mean()
            b = min(d[i, g].mean() for k, g in enumerate(gs) if k != lab[i])
            s.append((b - a) / max(a, b))
        return float(np.mean(s))
    k = max(range(2, min(9, n)), key=lambda k: silhouette(history[k]))
    lab = [0] * n
    for c, g in enumerate(sorted(history[k], key=lambda g: (-len(g), min(g)))):
        for i in g:
            lab[i] = c
    return lab


def layout(sim: np.ndarray) -> np.ndarray:
    """Classical MDS: 2D positions whose distances follow 1 - similarity as closely as possible."""
    d2 = (1 - sim) ** 2
    n = len(d2)
    j = np.eye(n) - np.ones((n, n)) / n
    w, v = np.linalg.eigh(-0.5 * j @ d2 @ j)
    top = np.argsort(w)[::-1][:2]
    return v[:, top] * np.sqrt(np.maximum(w[top], 0))


LABEL_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['label'],
                'properties': {'label': {'type': 'string'}}}


def name_cluster(members: list[dict], offline: bool) -> str:
    if offline:
        return ', '.join(m['paper'] for m in members[:3]) + (' …' if len(members) > 3 else '')
    user = 'Models in this group:\n\n' + '\n\n'.join(m['text'].split('\n')[1][:600] for m in members)
    a = llm.cached_ask(KG_DIR, 'You name groups of published process models. Give a short field-neutral name '
                       '(2 to 5 words) for what the models in the group have in common, e.g. the process or unit '
                       'operation they describe. No paper names.', user, LABEL_SCHEMA, 'cluster_label', 'low',
                       label='cluster name', keep_effort=True)
    return (a or {}).get('label') or ', '.join(m['paper'] for m in members[:3])


def build(g: Graph, offline: bool = False) -> tuple[Graph, dict]:
    sups = sup_texts(g)
    uris = list(sups)
    if len(uris) < 2:
        return Graph(), {}
    vec = word_counts([sups[u]['text'] for u in uris]) if offline else embed([sups[u]['text'] for u in uris])
    sim = similarity(vec)
    lab = clusters(sim)
    xy = layout(sim)
    names = {c: name_cluster([sups[u] for u, l in zip(uris, lab) if l == c], offline) for c in sorted(set(lab))}
    out = Graph()
    out.bind('mkg', MKG)
    data = {'model': 'offline word counts' if offline else EMBED_MODEL, 'clusters': names, 'sups': {}}
    group_iri = {}                                       # readable addresses: .../map/Combined_Membrane_Fouling_Models
    for c, name in names.items():
        slug = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_') or f'group_{c + 1}'
        while slug in group_iri.values():
            slug += f'_{c + 1}'
        group_iri[c] = slug
    for c, name in names.items():
        gu = URIRef(GROUP_NS + group_iri[c])
        out.add((gu, RDF.type, MKG.Group))
        out.add((gu, RDFS.label, Literal(name)))
        out.add((gu, MKG.basis, Literal('members grouped by code from embedding similarity; name written by an LLM')))
    for i, u in enumerate(uris):
        order = [j for j in np.argsort(-sim[i]) if j != i]
        near = order[:NEIGHBOURS]
        other = [j for j in order if sups[uris[j]]['paper'] != sups[u]['paper']][:NEIGHBOURS]   # for links.py
        data['sups'][u] = {'label': sups[u]['label'], 'paper': sups[u]['paper'], 'cluster': lab[i],
                           'near_other_papers': [{'sup': uris[j], 'similarity': round(float(sim[i, j]), 3)} for j in other],
                           'same_paper': [{'sup': uris[j], 'similarity': round(float(sim[i, j]), 3)} for j in order
                                          if sups[uris[j]]['paper'] == sups[u]['paper']],
                           'x': float(xy[i, 0]), 'y': float(xy[i, 1]),
                           'near': [{'sup': uris[j], 'similarity': round(float(sim[i, j]), 3)} for j in near]}
        out.add((URIRef(u), MKG.inGroup, URIRef(GROUP_NS + group_iri[lab[i]])))
    return out, data


def picture(data: dict, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colours = ['#1f4e79', '#c55a11', '#2e7d32', '#7b3f98', '#b8860b', '#00838f', '#ad1457', '#5d4037']
    fig, ax = plt.subplots(figsize=(13, 9))
    for c, name in data['clusters'].items():
        pts = [s for s in data['sups'].values() if s['cluster'] == int(c)]
        ax.scatter([p['x'] for p in pts], [p['y'] for p in pts], s=90, color=colours[int(c) % 8], label=name)
    papers = {}
    for s in data['sups'].values():
        papers.setdefault(s['paper'], []).append(s)
    for paper, pts in papers.items():                 # one label per paper; a paper with rival models says how many
        x, y = np.mean([p['x'] for p in pts]), np.mean([p['y'] for p in pts])
        ax.annotate(paper + (f' ({len(pts)} models)' if len(pts) > 1 else ''), (x, y), fontsize=9,
                    xytext=(7, 5), textcoords='offset points')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Sups by what the papers say (closer = more similar)', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(offline: bool):
    g = Graph().parse(os.path.join(KG_DIR, 'shared_graph.ttl'))
    out, data = build(g, offline)
    data['clusters'] = {str(k): v for k, v in data['clusters'].items()}
    json.dump(data, open(os.path.join(KG_DIR, 'sup_map.json'), 'w'), indent=1, ensure_ascii=False)
    out.serialize(os.path.join(KG_DIR, 'sup_map.ttl'), format='turtle')
    picture(data, os.path.join(KG_DIR, 'sup_map.png'))
    print(llm.usage_line())
    for c, name in data['clusters'].items():
        print(f'\n{name}')
        for s in data['sups'].values():
            if s['cluster'] == int(c):
                print(f"   {s['label'][:70]:70}  nearest: {data['sups'][s['near'][0]['sup']]['paper']} ({s['near'][0]['similarity']})")
    print(f"\n-> kg/sup_map.json, kg/sup_map.ttl, kg/sup_map.png")


if __name__ == '__main__':
    main('--offline' in sys.argv)
