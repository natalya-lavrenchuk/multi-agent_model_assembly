"""make_figures.py -- figures for the meeting, generated from the schema and the Sup graph files.

    python -m stages.make_figures papers/kontoravdi_2010 figures/
Needs graphviz ('dot') on the PATH.
"""
import json, os, subprocess, sys
from rdflib import Graph, Namespace, RDF

MKG = Namespace('http://example.org/mkg#')
C = {'Sup': '#3b3b6d', 'Law': '#2f6db5', 'Variable': '#2e8b57', 'Parameter': '#d9822b',
     'Metadata': '#8a8a8a', 'flag': '#c0392b', 'code': '#2f6db5', 'llm': '#8e44ad', 'later': '#9e9e9e'}
FONT = 'fontname="Helvetica"'


def render(dot_src, path):
    with open(path + '.dot', 'w') as f:
        f.write(dot_src)
    for ext in ('png', 'svg'):
        subprocess.run(['dot', f'-T{ext}', '-Gdpi=200' if ext == 'png' else '-Gdpi=72',
                        path + '.dot', '-o', f'{path}.{ext}'], check=True)


def ontology(out):
    box = lambda name, col, rows: (f'{name} [shape=plain label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="6">'
                                   f'<tr><td bgcolor="{col}"><font color="white"><b>{name}</b></font></td></tr>'
                                   f'<tr><td align="left" balign="left">{"<br/>".join(rows)}</td></tr></table>>];')
    src = f'''digraph G {{ rankdir=LR; nodesep=0.5; ranksep=1.1; node [{FONT} fontsize=12]; edge [{FONT} fontsize=11 color="#555555"];
    {box('Sup', C['Sup'], ['one published model', '<i>e.g. Kontoravdi 2005</i>'])}
    {box('Law', C['Law'], ['one printed equation', 'formula (SymPy string)', 'equationNumber'])}
    {box('Variable', C['Variable'], ['symbol, formulaName', 'unit*, dimensions*', 'description*'])}
    {box('Parameter', C['Parameter'], ['symbol, formulaName', 'value*, unit*', 'fixed (constants)'])}
    {box('Metadata', C['Metadata'], ['source: DOI, page, quote', 'form (ODE / algebraic; DAE on Sup)', 'conditions / assumptions', 'status', 'roles*, law type*, index*'])}
    Sup -> Law [label="includes"];
    Sup -> Metadata [label="hasMetadata"];
    Law -> Variable [label="computes (exactly 1)" color="{C['Law']}" penwidth=2];
    Law -> Variable [label="uses"];
    Law -> Parameter [label="hasParameter"];
    Law -> Metadata [label="hasMetadata"];
    Law -> Law [label="sameAs (proposed)" style=dashed];
    note [shape=note {FONT} fontsize=10 color="#999999" label="5 classes, no others\\l* = field defined, not yet filled by the extractor\\lone Variable node per symbol per Sup, shared by all its laws\\l"];
    }}'''
    render(src, f'{out}/ontology')


def pipeline(out):
    n = lambda id_, label, col, style='filled', shape='box': (f'{id_} [shape={shape} style="{style},rounded" fillcolor="{col}20" '
                                                               f'color="{col}" label="{label}"];')
    src = f'''digraph P {{ rankdir=LR; nodesep=0.35; ranksep=0.45; node [{FONT} fontsize=11 penwidth=1.5]; edge [{FONT} fontsize=10 color="#555555"];
    {n('pdf', 'paper PDF', '#555555', 'filled', 'note')}
    {n('ocr', 'PaddleOCR-VL\\n(Colab, GPU)', C['code'])}
    {n('pj', 'paddle.json\\n28 formulas, LaTeX', '#555555', 'filled', 'note')}
    {n('bj', 'paddle_blocks.json\\ntext, tables', '#555555', 'filled', 'note')}
    {n('conv', 'LaTeX -> SymPy\\nrule-based, no LLM', C['code'])}
    {n('struct', 'structure check\\ncomputes / uses\\n(elimination)', C['code'])}
    {n('tab', 'table evidence\\nParameter vs input', C['code'])}
    {n('build', 'build Sup graph\\n+ schema validation', C['code'])}
    {n('ttl', 'sup_graph.ttl', C['Sup'], 'filled', 'note')}
    {n('gold', 'reference graph\\n(hand-checked)', '#555555', 'filled', 'note')}
    {n('cmp', 'compare', C['code'])}
    {n('var', 'symbol definitions\\n& units (LLM picks\\na paper snippet)', C['later'], 'dashed')}
    {n('meta', 'law metadata:\\nassumptions, roles,\\nlaw type', C['later'], 'dashed')}
    {n('asm', 'Assembler:\\nmerge Sups', C['later'], 'dashed')}
    pdf -> ocr; ocr -> pj; ocr -> bj; pj -> conv -> struct -> build; bj -> tab -> build; build -> ttl;
    ttl -> cmp; gold -> cmp;
    bj -> var [style=dashed color="{C['later']}"]; var -> build [style=dashed color="{C['later']}"];
    bj -> meta [style=dashed color="{C['later']}"]; meta -> build [style=dashed color="{C['later']}"];
    ttl -> asm [style=dashed color="{C['later']}"];
    {{ rank=same; var; meta; }}
    legend [shape=plain {FONT} fontsize=10 label=<<table border="0" cellspacing="2"><tr><td bgcolor="{C['code']}" width="14"></td><td align="left">built, code only</td></tr><tr><td border="1" color="{C['later']}" style="dashed" width="14"></td><td align="left">postponed</td></tr></table>>];
    }}'''
    render(src, f'{out}/pipeline')


def sup_graph(paper_dir, out, tag='pipeline'):
    g = Graph().parse(f'{paper_dir}/sup_graph_{tag}.ttl')
    rep = json.load(open(f'{paper_dir}/sup_graph_{tag}_report.json'))
    flagged = set(rep['summary']['laws_flagged'])
    flag_syms = set()
    for f in rep['flags']:
        if f.get('check') in ('split name?', 'computes') or f.get('symbol') == 'd':
            flag_syms.update(s for s in str(f.get('symbol', '')).split(' ~ ') if s)
    flag_syms.add('d')
    sym = lambda x: str(g.value(x, MKG.symbol))
    laws = sorted(g.subjects(RDF.type, MKG.Law), key=lambda L: int(str(g.value(L, MKG.equationNumber))))
    groups = {'growth & death (eqs 1-7)': range(1, 8), 'metabolism (eqs 8-16)': range(8, 17), 'mAb synthesis (eqs 17-28)': range(17, 29)}
    lines = [f'digraph S {{ rankdir=LR; nodesep=0.08; ranksep=0.9; splines=true; node [{FONT} fontsize=10 style=filled penwidth=1.2]; edge [arrowsize=0.5 color="#9a9a9a"];']
    for gi, (label, rng) in enumerate(groups.items()):
        lines.append(f'subgraph cluster_{gi} {{ label="{label}"; {FONT}; fontsize=11; color="#cccccc";')
        for L in laws:
            n = int(str(g.value(L, MKG.equationNumber)))
            if n in rng:
                col = C['flag'] if n in flagged else C['Law']
                lines.append(f'"L{n}" [shape=box label="eq {n}" fillcolor="{col}" fontcolor=white color="{col}"];')
        lines.append('}')
    vs = set(g.subjects(RDF.type, MKG.Variable))
    for v in vs:
        s = sym(v)
        col = C['flag'] if s in flag_syms else C['Variable']
        lines.append(f'"V_{s}" [shape=ellipse label="{s}" fillcolor="{col}30" color="{col}"];')
    for L in laws:
        n = int(str(g.value(L, MKG.equationNumber)))
        for v in g.objects(L, MKG.computes):
            lines.append(f'"L{n}" -> "V_{sym(v)}" [color="{C["Law"]}" penwidth=1.6];')
        for v in g.objects(L, MKG.uses):
            lines.append(f'"V_{sym(v)}" -> "L{n}" [style=dashed];')
    npar = len(set(g.subjects(RDF.type, MKG.Parameter)))
    lines.append(f'legend [shape=note fillcolor=white color="#999999" fontsize=10 label="Kontoravdi Sup ({tag})\\l'
                 f'box = Law, ellipse = Variable\\lsolid blue: computes, dashed: uses\\lred: flagged by structural checks\\l'
                 f'{npar} Parameters not drawn (hasParameter edges)\\l"];')
    lines.append('}')
    render('\n'.join(lines), f'{out}/sup_graph_{tag}')


def example(paper_dir, out, eqs=(4, 5), tag='pipeline'):
    g = Graph().parse(f'{paper_dir}/sup_graph_{tag}.ttl')
    sym = lambda x: str(g.value(x, MKG.symbol))
    lines = [f'digraph E {{ rankdir=TB; nodesep=0.3; ranksep=0.6; node [{FONT} fontsize=11 style=filled penwidth=1.2]; edge [{FONT} fontsize=9 color="#777777"];',
             f'sup [shape=box label="Sup: Kontoravdi 2005" fillcolor="{C["Sup"]}" fontcolor=white];']
    for L in g.subjects(RDF.type, MKG.Law):
        n = int(str(g.value(L, MKG.equationNumber)))
        if n not in eqs:
            continue
        M = g.value(L, MKG.hasMetadata)
        lines.append(f'L{n} [shape=box label="Law eq {n}\\n{str(g.value(L, MKG.formula))[:60]}" fillcolor="{C["Law"]}" fontcolor=white];')
        lines.append(f'sup -> L{n} [label="includes"];')
        lines.append(f'M{n} [shape=note label="Metadata\\nform: {g.value(M, MKG.form)}\\npage {g.value(M, MKG.page)}, DOI\\nstatus: {g.value(M, MKG.status)}" fillcolor="{C["Metadata"]}25" color="{C["Metadata"]}"];')
        lines.append(f'L{n} -> M{n} [label="hasMetadata"];')
        for v in g.objects(L, MKG.computes):
            lines.append(f'"{sym(v)}" [shape=ellipse fillcolor="{C["Variable"]}30" color="{C["Variable"]}"]; L{n} -> "{sym(v)}" [label="computes" color="{C["Law"]}" penwidth=2];')
        for v in g.objects(L, MKG.uses):
            lines.append(f'"{sym(v)}" [shape=ellipse fillcolor="{C["Variable"]}30" color="{C["Variable"]}"]; L{n} -> "{sym(v)}" [label="uses" style=dashed];')
        for p in g.objects(L, MKG.hasParameter):
            lines.append(f'"{sym(p)}" [shape=diamond fillcolor="{C["Parameter"]}30" color="{C["Parameter"]}"]; L{n} -> "{sym(p)}" [label="hasParameter"];')
    lines.append('}')
    render('\n'.join(lines), f'{out}/example_eq{"_".join(map(str, eqs))}')


if __name__ == '__main__':
    paper_dir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    ontology(out); pipeline(out)
    sup_graph(paper_dir, out, 'pipeline'); sup_graph(paper_dir, out, 'reference')
    example(paper_dir, out, (4, 5))
    print('figures ->', out)
