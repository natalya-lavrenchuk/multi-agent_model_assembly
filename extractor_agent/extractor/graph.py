"""graph.py -- build one paper's Sup graph (Turtle), check it against the schema, compare
the pipeline graph with the reference graph.

    python -m extractor.graph papers/kontoravdi_2010

Reads papers/<id>/paper.json (sup id, doi, title) and writes, in the same folder:
    sup_graph_pipeline.ttl   + _report.json   from the reader's LaTeX (paddle.json)
    sup_graph_reference.ttl  + _report.json   from the hand-checked LaTeX (gold_laws.json), if present
    graph_comparison.json                     the two graphs compared law by law

v0 STRUCTURE (Oct 2026, for now): every law gets its OWN Variable and Parameter nodes.
mu in eq 4 and mu in eq 9 are kg:law/<paper>_eq4/var/mu and kg:law/<paper>_eq9/var/mu. Nothing is shared.
IRIs end in readable names (the Sup is kg:<sup id>, a law kg:law/<paper>_eq<n>), so a viewer that shows
the end of an IRI instead of rdfs:label still says which paper a node is from.
Equations are stored as LaTeX (mkg:latex); mkg:formula (SymPy) is left empty.

How each part is decided:
    computes      the one symbol on the left of '=' (inside the derivative for an ODE). If the left side
                  has several (d(V X_v)/dt), no computes edge, and the law is flagged.
    hasParameter  for symbols no law computes, Parameter or Variable comes from:
                    pipeline graph   classification.json (extractor/classify.py, LLM) if present
                    otherwise, and always for the reference graph: the table rule (code):
                    Parameter = a table row gives it a number
                  The answer (basis, quote, page, reason, status) is written on each node.
    uses          every other symbol
    Metadata      doi, page, quote (the LaTeX), form (ODE/PDE/algebraic), side condition, status
"""
from __future__ import annotations
import json
import os
import re
import sys

from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS
from rdflib.term import Node

from extractor.units import parse as parse_unit
from extractor import triage
from extractor.paper import load_model, load_snippets, models, table_value

MKG = Namespace('http://example.org/mkg#')
SCHEMA = os.path.join(os.path.dirname(__file__), '..', 'schema', 'kg_schema.ttl')
EDGES = ('computes', 'uses', 'hasParameter')

# =============================================================================
# Build
# =============================================================================

def build(paper_dir: str, cfg: dict, gold: bool = False, model: dict | None = None) -> tuple[Graph, list[dict], dict]:
    """One Sup. model: one of paper.models(cfg) (a paper with rival models has several); default the first."""
    model = model or models(cfg)[0]
    formulas, reads, source = load_model(paper_dir, gold, model=model)
    eqs = [f.get('eq') or i + 1 for i, f in enumerate(formulas)]
    if not gold:                                          # symbols are read from the equation the model uses
        reads = _model_reads(paper_dir, formulas, reads)

    # Symbols no law computes: Parameter or Variable (whole-paper decision, one per symbol)
    on_lhs = {s for r in reads for s in r['lhs']}
    rhs_only = sorted({s for r in reads for s in r['rhs']} - on_lhs)
    snippets = load_snippets(paper_dir)
    tab = {s: hit for s in rhs_only if (hit := table_value(s, snippets))}
    sym_file, llm_file = f'{paper_dir}/symbols.json', f'{paper_dir}/classification.json'
    described = json.load(open(sym_file))['symbols'] if not gold and os.path.exists(sym_file) else {}
    law_meta = json.load(open(f'{paper_dir}/laws.json'))['laws'] \
        if not gold and os.path.exists(f'{paper_dir}/laws.json') else {}
    fixes = json.load(open(f'{paper_dir}/fixes.json'))['fixes'] if not gold and os.path.exists(f'{paper_dir}/fixes.json') else {}
    meta = json.load(open(f'{paper_dir}/sup_meta.json'))['sups'].get(model['id'], {}) \
        if not gold and os.path.exists(f'{paper_dir}/sup_meta.json') else {}
    misreads = {} if gold else _misreads(paper_dir)
    _flag_misreads(paper_dir, law_meta)
    _apply_corrections(paper_dir, described, law_meta, {model['id']: meta}, quiet=gold)   # a human's corrections win (the hand-checked reference graph has no laws.json to correct)
    if described:                                         # symbols.py: class, basis, quote, reason per symbol
        decided, how = {s: {'class': d['class'], 'basis': d.get('class_basis'), 'quote': d.get('quote'),
                            'page': d.get('page'), 'reason': d.get('class_reason'), 'status': triage.status(d)}
                        for s, d in described.items() if s in rhs_only}, 'symbols.json (LLM)'
        for s in set(rhs_only) - set(decided):
            decided[s] = _table_rule(tab.get(s)) | {'status': 'pending', 'reason': 'not in symbols.json: rerun symbols'}
    elif not gold and os.path.exists(llm_file):
        decided, how = json.load(open(llm_file))['symbols'], 'classification.json (LLM)'
        for s, d in decided.items():                      # v0: Parameter or Variable only; stated = accepted
            d['status'] = 'accepted' if d['basis'] == 'stated' else 'pending'
        for s in set(rhs_only) - set(decided):            # read after classify ran (lister changed): rerun classify
            decided[s] = _table_rule(tab.get(s)) | {'status': 'pending',
                                                     'reason': 'not in classification.json: rerun classify'}
    else:
        decided, how = {s: _table_rule(tab.get(s)) for s in rhs_only}, 'table rule (code)'

    computes = dict(zip(eqs, _computes(reads)))
    flags = _flags(eqs, reads, rhs_only, tab, decided, computes)
    flagged = {f['law'] for f in flags if 'law' in f and f['check'] != 'review'}
    flagged |= {n for f in flags if f['check'] == 'split name?' for n in f['laws']}

    g = Graph()
    KG = Namespace(f"http://example.org/kg/{model['id']}/")
    g.bind('mkg', MKG); g.bind('kg', KG)

    sup, sm = KG[model['id']], KG[f"{model['id']}_meta"]   # IRIs end in names a person can read: xing_2010, xing_2010_meta
    forms = {r['form'] for r in reads}
    short = os.path.basename(os.path.normpath(paper_dir))    # 'xing_2010': which paper a node is from, in its label
    free = bool(model.get('free'))                        # free-floating laws: no Sup, context on each law
    title = cfg['title'] + (f" ({model['name']})" if len(models(cfg)) > 1 else '')
    from extractor.supmeta import facts_of               # here, not at the top: supmeta.py imports graph.py
    facts = facts_of(meta)
    if not free:
        _sup_metadata(g, sup, sm, title, cfg, forms, source, meta, model, facts)
    for n, f, r in zip(eqs, formulas, reads):
        L, M = KG[f'law/{short}_eq{n}'], KG[f'law/{short}_eq{n}_meta']
        if not free:
            g.add((sup, MKG.includes, L))
        name = model.get('law_names', {}).get(str(n))
        _add(g, L, RDF.type, MKG.Law, RDFS.label, _en(f'{short} eq {n}' + (f' ({name})' if name else '')), MKG.equationNumber, str(n),
             MKG.latex, r['latex'], MKG.hasMetadata, M)
        if free:                                          # no Sup to carry the paper's context: it goes on each law
            _add(g, M, MKG.freeLaw, True, MKG.contextDescription, meta.get('description'))
            _facts(g, M, facts)
        if r.get('updates'):
            _add(g, M, MKG.updates, r['updates'])
        _add(g, M, RDF.type, MKG.Metadata, MKG.doi, cfg['doi'], MKG.page, str(f.get('page', 0) + 1),
             MKG.quote, r['latex'], MKG.form, r['form'], MKG.status, 'flagged' if n in flagged else 'pending')
        if r['condition']:
            a = BNode()
            g.add((M, MKG.hasAssumption, a))
            _add(g, a, MKG.text, r['condition'], MKG.reasoned, False, MKG.quote, r['condition'])
        for a in [a for a in model.get('model_assumptions', []) if str(a.get('eq')) == str(n)]:
            b = BNode()                                   # an assumption sup.py found in this law's own sentence
            g.add((M, MKG.hasAssumption, b))
            _add(g, b, MKG.text, a['text'], MKG.severity, a.get('severity'), MKG.reasoned, a.get('reasoned'),
                 MKG.quote, a.get('quote'))
        lm = law_meta.get(str(n), {})                     # laws.py: description, law type, assumptions, roles
        _add(g, M, MKG.possibleMisprint, lm.get('issue'))   # the LLM's misprint opinion stays visible after a person accepts it
        fx = fixes.get(str(n), {})                        # fixes.py: a corrected form, next to the printed one, never instead
        mr = misreads.get(str(n))
        me = _model_equation(lm, fx, mr) if (lm.get('issue') or mr or fx) else {'latex': None, 'corrected': None, 'outcomes': [], 'reason': None}
        if me['latex']:
            g.set((L, MKG.latex, Literal(me['latex'])))
        if 'reading corrected' in me['outcomes']:
            _add(g, M, MKG.ocrFixNote, (mr or {}).get('note') or (mr or {}).get('reason') or 'reading corrected by the reviewer against the PDF')
        if me['corrected']:                               # the paper prints it wrong: keep both
            _add(g, L, MKG.latexCorrected, me['corrected'])
            _add(g, M, MKG.correctionReason, me['reason'], MKG.correctionQuote, fx.get('quote') if me['corrected'] == fx.get('corrected_latex') else None,
                 MKG.correctionStatus, 'accepted' if lm.get('status') == 'accepted' else 'proposed')
        if (lm.get('issue') or mr) and lm.get('status') == 'accepted':
            for o in me['outcomes']:
                _add(g, M, MKG.reviewOutcome, o)
        _add(g, M, MKG.description, lm.get('description'),
             MKG.basis, lm.get('description_basis'))
        for x in lm.get('assumptions', []):
            a = BNode()
            g.add((M, MKG.hasAssumption, a))
            _add(g, a, MKG.text, x['text'], MKG.severity, x['severity'], MKG.reasoned, x['reasoned'], MKG.quote, x['quote'])

        names = _local_names(r['lhs'] + r['rhs'])
        computed = computes[n]
        for s in r['lhs'] + r['rhs']:
            if s == computed:
                edge, cls, kind = MKG.computes, MKG.Variable, 'var'
            elif decided.get(s, {}).get('class') == 'parameter':
                edge, cls, kind = MKG.hasParameter, MKG.Parameter, 'par'
            else:
                edge, cls, kind = MKG.uses, MKG.Variable, 'var'
            x = KG[f'law/{short}_eq{n}/{kind}/{names[s]}']
            if (x, RDF.type, cls) not in g:
                _add(g, x, RDF.type, cls, MKG.symbol, s, RDFS.label, _en(f'{s} ({short} eq {n})'))
                d = decided.get(s, {})                    # the classification, on every node of the symbol
                for p in ('basis', 'quote', 'page', 'reason', 'status'):
                    if d.get(p) is not None:
                        _add(g, x, MKG[p], str(d[p]) if p == 'page' else d[p])
                ds = described.get(s, {})                 # symbols.py: what it is, its unit and value
                _add(g, x, MKG.description, ds.get('description'), MKG.unitString, ds.get('unit'))
                pu = parse_unit(ds.get('unit'))                   # units.py: the printed unit as a pint unit
                if pu:
                    _add(g, x, MKG.unit, pu['unit'] or 'dimensionless', MKG.dimension, pu['dimension'],
                         MKG.siFactor, pu['si_factor'], MKG.unitNote, pu['note'])
                own = ds.get('values_by_model') or {}    # rival models / free laws fitted one by one
                val = own.get(model.get('law_names', {}).get(str(n)) if free else model.get('name'), ds.get('value'))
                num = ds.get('value_number') if val == ds.get('value') and ds.get('value_number') is not None else _number(val)
                if ds.get('same_as'):                                                  # S and s: one quantity
                    _add(g, x, MKG.sameQuantityAs, ds['same_as'])
                if kind == 'par' and ds.get('uncertainty') and not ds.get('values_by_model'):   # per-model values keep their ± in the value text
                    _add(g, x, MKG.valueUncertainty, ds['uncertainty'])
                if kind == 'par' and ds.get('estimation') not in (None, 'not said'):   # Layer 3: what to estimate
                    _add(g, x, MKG.estimation, ds['estimation'], MKG.estimationQuote, ds.get('estimation_quote'))
                if kind == 'var' and ds.get('measured') == 'measured':                 # Layer 3: outputs with data
                    _add(g, x, MKG.measured, True, MKG.measuredQuote, ds.get('measured_quote'))
                if kind == 'var' and ds.get('initial_value'):                          # to reproduce the run
                    _add(g, x, MKG.initialValue, ds['initial_value'], MKG.initialQuote, ds.get('initial_quote'))
                if kind == 'par' and ds.get('values_by_case'):   # one value per data set: kept as text, no single value
                    _add(g, x, MKG.valuesByCase, json.dumps(ds['values_by_case'], ensure_ascii=False))
                if kind == 'par' and (val != ds.get('value') or ds.get('value_set') != 'unclear') \
                        and ';' not in str(val) and num is not None:   # several candidates: wait for a person
                    _add(g, x, MKG.value, num)
            g.add((L, edge, x))
        for role in (lm.get('roles', []) if r['form'] not in ('ODE', 'PDE') and not r.get('value') else []):   # slot -> node; algebraic laws only
            if role['symbol'] not in names:               # laws.json older than the reading: skip, rerun laws
                continue
            kind = next((k for k in ('var', 'par') if (KG[f'law/{short}_eq{n}/{k}/{names[role["symbol"]]}'], None, None) in g), None)
            if kind:
                b = BNode()
                g.add((M, MKG.hasRole, b))
                _add(g, b, MKG.slot, role['role'], MKG.boundTo, KG[f'law/{short}_eq{n}/{kind}/{names[role["symbol"]]}'])

    count = lambda cls: len(set(g.subjects(RDF.type, cls)))
    edges = lambda p: len(list(g.triples((None, p, None))))
    summary = {
        'structure': 'v0 distinct nodes (one Variable/Parameter node per symbol per law)',
        'source': source, 'laws': len(formulas),
        'variable_nodes': count(MKG.Variable), 'parameter_nodes': count(MKG.Parameter),
        'distinct_symbols': len({s for r in reads for s in r['lhs'] + r['rhs']}),
        'computes_edges': edges(MKG.computes), 'uses_edges': edges(MKG.uses),
        'hasParameter_edges': edges(MKG.hasParameter),
        'laws_by_form': {fm: sum(r['form'] == fm for r in reads) for fm in sorted(forms)},
        'laws_flagged': sorted(flagged, key=_eq_key),
        'classification': how,
        'parameter_symbols': sorted(s for s in rhs_only if decided[s]['class'] == 'parameter'),
        'for_review': sorted(s for s in rhs_only if decided[s]['status'] == 'pending'),
        'triples': len(g),
    }
    return g, flags, summary


def _en(text: str) -> Literal:
    """A label with a language tag: GraphDB's visual graph shows labels in its preferred language (en) and
    falls back to the last part of the IRI ('sup', 'eq1') for labels without one."""
    return Literal(text, lang='en')


def _facts(g, node, facts):
    """supmeta.py's free-form facts: one blank node each, topic chosen by the LLM, quote when the paper says it."""
    for f in facts:
        b = BNode()
        g.add((node, MKG.hasFact, b))
        _add(g, b, MKG.topic, f.get('topic'), MKG.text, f.get('text'), MKG.quote, f.get('quote'), MKG.basis, f.get('basis'))


def _sup_metadata(g, sup, sm, title, cfg, forms, source, meta, model, facts):
    _add(g, sup, RDF.type, MKG.Sup, RDFS.label, _en(f"{model['id']} · {title}"), MKG.hasMetadata, sm)
    _add(g, sm, RDF.type, MKG.Metadata, MKG.doi, cfg['doi'],
         MKG.form, 'DAE' if {'ODE', 'algebraic'} <= forms else '/'.join(sorted(forms)),
         MKG.status, 'pending', RDFS.comment, f'v0 distinct nodes; built from {source} by extractor/graph.py',
         MKG.description, meta.get('description'), MKG.completeModel, model.get('complete_model'))
    _facts(g, sm, facts)
    for a in [a for a in model.get('model_assumptions', []) if not a.get('eq')]:   # model-wide only, approved by a human
        b = BNode()
        g.add((sm, MKG.hasAssumption, b))
        _add(g, b, MKG.text, a['text'], MKG.severity, a.get('severity'), MKG.reasoned, a.get('reasoned'), MKG.quote, a.get('quote'))


def _model_reads(paper_dir: str, formulas: list[dict], reads: list[dict]) -> list[dict]:
    from extractor.paper import clean, prose_names, read_equations
    _, laws = corrected(paper_dir)
    p = f'{paper_dir}/fixes.json'
    fixes = json.load(open(p))['fixes'] if os.path.exists(p) else {}
    mrs, names, out = _misreads(paper_dir), None, list(reads)
    for i, f in enumerate(formulas):
        n = str(f.get('eq'))
        lm, fx, mr = laws.get(n, {}), fixes.get(n, {}), mrs.get(n)
        if not (lm.get('issue') or mr):
            continue
        me = _model_equation(lm, fx, mr)                 # a reading (OCR or corrected against the PDF), never a misprint correction:
        use = me['latex'] if {'reading corrected', 'kept as read'} & set(me['outcomes']) else None   # links stay those of the printed law
        if use and clean(use) != clean(f['latex']):
            names = names if names is not None else prose_names(paper_dir)
            out[i] = read_equations([{**f, 'latex': f'$$ {clean(use)} $$'}], names)[0]
    return out


def corrected(paper_dir: str) -> tuple[dict, dict]:
    """symbols.json and laws.json with the reviewer's corrections applied: what the later LLM steps
    (laws.py reads the symbols, supmeta.py reads both) should see, so a fixed description or class
    is passed on instead of the LLM's first answer."""
    load = lambda f, k: json.load(open(f'{paper_dir}/{f}'))[k] if os.path.exists(f'{paper_dir}/{f}') else {}
    syms, laws = load('symbols.json', 'symbols'), load('laws.json', 'laws')
    _flag_misreads(paper_dir, laws)
    _apply_corrections(paper_dir, syms, laws, quiet=True)
    return syms, laws



def _model_equation(lm: dict, fx: dict, mr: dict | None = None) -> dict:
    """What the reviewer's answer means for a flagged law.
    fx: fixes.py's proposed correction (a misprint in the paper). mr: an OCR misread found in review
    (printed_latex = what the OCR read, corrected_latex = what the PDF shows).
    The reviewer picks the equation the model uses (choice: 'fix' the corrected version, 'paper' the
    paper's version, 'ocr' the OCR reading, 'own' their LaTeX). Unreviewed, a misread keeps the OCR reading.
    Returns latex (what mkg:latex should hold, None = as stored), corrected (mkg:latexCorrected),
    outcomes (for mkg:reviewOutcome) and reason."""
    raw, reading = (mr['printed_latex'], mr['corrected_latex']) if mr else (None, None)
    proposed = fx.get('corrected_latex') if fx.get('action') == 'correct' else None
    choice = lm.get('choice')
    if choice is None and not mr:                        # answers saved by the earlier review page
        if lm.get('misprint_kind') == 'ocr' and lm.get('latex_fix'):
            return {'latex': lm['latex_fix'], 'corrected': None, 'outcomes': ['reading corrected'], 'reason': None}
        choice = 'paper' if lm.get('misprint_kind') == 'fine' or lm.get('use_fix') is False else 'fix'
    choice = choice or 'ocr'
    misread = ['reading corrected'] if mr else []
    if choice == 'ocr' and mr:
        return {'latex': raw, 'corrected': None, 'outcomes': ['kept as read'], 'reason': None}
    if choice == 'fix' and proposed:
        if lm.get('already_in_paper'):
            return {'latex': proposed, 'corrected': None, 'outcomes': ['reading corrected'], 'reason': fx.get('reason')}
        return {'latex': reading, 'corrected': proposed, 'outcomes': misread + ['paper misprint'], 'reason': fx.get('reason')}
    if choice == 'own' and (lm.get('own_latex') or '').strip():
        if lm.get('paper_prints_it') in (True, 'yes'):
            return {'latex': lm['own_latex'], 'corrected': None, 'outcomes': ['reading corrected'], 'reason': None}
        return {'latex': reading, 'corrected': lm['own_latex'], 'outcomes': misread + ['paper misprint'], 'reason': 'written by the reviewer'}
    return {'latex': reading, 'corrected': None, 'outcomes': misread or ['false alarm'], 'reason': None}


def _misreads(paper_dir: str) -> dict:
    p = f'{paper_dir}/fixes.json'
    return json.load(open(p)).get('misreads', {}) if os.path.exists(p) else {}


def _flag_misreads(paper_dir: str, laws: dict):
    """An OCR misread found in review is a review item of its law, like a possible misprint."""
    for n, m in _misreads(paper_dir).items():
        if n in laws:
            laws[n]['review'] = (laws[n].get('review') or []) + ['possible OCR misread']
            laws[n]['status'] = 'pending'                # until the reviewer resolves it (corrections.json)


def _apply_corrections(paper_dir, described, law_meta, sup_meta=None, quiet=False):
    """corrections.json (exported from review.html): edited fields replace the LLM's, with basis
    'human'; 'accepted' sets status accepted. Corrections for symbols or laws no longer in the
    model are reported, not applied."""
    path = f'{paper_dir}/corrections.json'
    if not os.path.exists(path):
        return
    corr = json.load(open(path))
    for kind, table in (('symbols', described), ('laws', law_meta), ('sup', sup_meta or {})):
        for key, change in corr.get(kind, {}).items():
            if key not in table:
                if kind != 'sup' and not quiet:
                    print(f'  correction for {kind[:-1]} {key} not applied: not in {kind}.json any more')
                continue
            d = table[key]
            for field, v in change.items():
                if field in ('accepted', 'approved'):
                    d['status'] = 'accepted' if v else d.get('status')
                elif kind == 'sup' and field == 'facts':      # 'topic: text' per line; unchanged lines keep their quote
                    from extractor.supmeta import facts_of
                    old = {(f['topic'], f['text']): f for f in facts_of(d)}
                    new = []
                    for line in (v.splitlines() if isinstance(v, str) else []):
                        if not line.strip():
                            continue
                        topic, _, text = line.partition(':') if ':' in line else ('note', '', line)
                        key = (topic.strip().lower(), text.strip())
                        new.append(old.get(key) or {'topic': key[0], 'text': key[1], 'quote': None, 'basis': 'human'})
                    d['facts'] = new
                elif kind == 'sup' and field == 'equations':
                    continue                              # applied to paper.json by sup.py --approve
                else:
                    d[field] = v
                    basis = {'description': 'description_basis', 'unit': 'unit_basis', 'class': 'class_basis'}.get(field)
                    if basis:
                        d[basis] = 'human'
                    if field == 'class':
                        d['class_reason'] = 'corrected by the reviewer'


def _eq_key(n: str):
    """Sort '2' < '10' < 'A-3' < 'C-11': numbers first, then prefixed ids."""
    n = str(n)                                        # also '6.2', 'p5-3', 'S-p2-1': natural order, never int vs str
    prefix = re.match(r'[A-Za-z-]*', n).group(0)
    return (prefix != '', prefix, [(0, int(x), '') if x.isdigit() else (1, 0, x) for x in re.findall(r'\d+|\D+', n[len(prefix):])])


def _number(v) -> float | None:
    """'6 \\times 10^{-2} h^{-1}' / '1.76 mM' / '2.6e8' -> the number, or None."""
    if not v:
        return None
    v = re.sub(r'\s*(?:\\pm|±|\+/?-)\s*\d[\d.]*', '', str(v))   # '7.2 ± 0.4 × 10^{-3}': the error goes
    m = re.match(r'\s*\$?\s*(-?\d+(?:\.\d+)?)\s*(?:(?:\\times|×|x)\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?|e(-?\d+))?', str(v))
    if not m:
        return None
    return float(f'{m.group(1)}e{int(m.group(2) or m.group(3) or 0)}')   # exact: 1.4e-1, not 1.4 * 10**-1


def _table_rule(hit) -> dict:
    """The code-only classification: Parameter if a table row gives the symbol a number."""
    if hit:
        return {'class': 'parameter', 'basis': 'stated', 'quote': hit[0], 'page': None,
                'reason': f'code rule: never computed, numeric value in a table row (match: {hit[1]})', 'status': 'accepted'}
    return {'class': 'variable', 'basis': 'inferred', 'quote': None, 'page': None,
            'reason': 'code rule: never computed, no table value; kept as Variable', 'status': 'pending'}


def _computes(reads) -> list[str | None]:
    """What each law computes. One symbol on the left: that one. Several, as in d(VX)/dt or
    V dS/dt + S dV/dt: the one no other law of this Sup computes (V comes from dV/dt = F).
    None if that still leaves zero or several."""
    single = [r['lhs'][0] for r in reads if len(r['lhs']) == 1]
    out = []
    for r in reads:
        if len(r['lhs']) == 1:
            out.append(r['lhs'][0])
            continue
        left = [x for x in r['lhs'] if x not in single]
        if len(left) != 1 and r['form'] in ('ODE', 'PDE') and r.get('states'):
            # dC/dt + ((1-eps)/eps) dq/dt + u dC/dz = ...: a balance computes the state written first
            # (q has its own law); the population balance dn/dt + d(Gn)/dx = ... computes n
            left = [r['states'][0]]
        out.append(left[0] if len(left) == 1 else None)
    return out


def _flags(eqs, reads, rhs_only, tab, decided, computes) -> list[dict]:
    flags = []
    for n, r in zip(eqs, reads):
        if computes[n] is None:
            flags.append({'law': n, 'check': 'computes undecided',
                          'detail': f"left side has {len(r['lhs'])} symbols {r['lhs']}: which one this law computes "
                                    f"needs another law, so no computes edge"})
        flags += [{'law': n, 'check': 'reading', 'detail': note} for note in r['notes']]
    used = {s for r in reads for s in r['rhs']}
    for s in sorted({r['lhs'][0] for r in reads if r['form'] == 'algebraic' and len(r['lhs']) == 1} - used):
        flags.append({'symbol': s, 'check': 'computed, never used',
                      'laws': [n for n, r in zip(eqs, reads) if r['lhs'] == [s]],
                      'detail': 'an algebraic law computes it but no law uses it: a misread or misprinted symbol '
                                'elsewhere, or the model output (e.g. rate laws printed without balances)'})
    for s in rhs_only:
        laws = [n for n, r in zip(eqs, reads) if s in r['rhs']]
        if s in tab and tab[s][1] in ('partial', 'confusable'):
            flags.append({'symbol': s, 'check': 'split name?',
                          'detail': f'table names a longer or confusable symbol ({tab[s][0]}): the reader may have '
                                    f'split one symbol into two', 'laws': laws})
        if decided[s]['status'] == 'pending':
            flags.append({'symbol': s, 'check': 'review', 'laws': laws,
                          'detail': f"{decided[s]['class']} ({decided[s]['basis']}): {decided[s]['reason']}"})
    return flags


def _local_names(syms: list[str]) -> dict[str, str]:
    """LaTeX symbol -> IRI-safe name, unique within one law: '[GLC]_{in}' -> 'C_GLC_in', '\\mu_{d,max}' -> 'mu_d_max',
    'DOT^{*}' -> 'DOT_star' (not DOT)."""
    out = {}
    for s in syms:
        if s in out:
            continue
        base = re.sub(r'[^A-Za-z0-9]+', '_', re.sub(r'\[([^\]]*)\]', r'C_\1', s).replace('\\', '').replace('*', '_star').replace("'", '_prime')).strip('_') or 'sym'
        name, k = base, 2
        while name in out.values():
            name, k = f'{base}_{k}', k + 1
        out[s] = name
    return out


def _add(g: Graph, subject, *pairs):
    """_add(g, s, p1, o1, p2, o2, ...): plain Python values become Literals."""
    for p, o in zip(pairs[::2], pairs[1::2]):
        if o is not None:                                  # a field the paper does not give (no DOI printed)
            g.add((subject, p, o if isinstance(o, Node) else Literal(o)))

# =============================================================================
# Check against the schema
# =============================================================================

def validate(g: Graph) -> list[str]:
    """Every mkg: term is defined in the schema; every edge connects the right classes; each Law
    computes exactly one Variable; v0 only: every Variable/Parameter node belongs to one Law."""
    defined = {str(s) for s in Graph().parse(SCHEMA).subjects()}
    problems = [f'undefined term {t}' for s, p, o in g for t in (p, o)
                if str(t).startswith(str(MKG)) and str(t) not in defined]
    ranges = {MKG.includes: MKG.Law, MKG.computes: MKG.Variable, MKG.uses: MKG.Variable,
              MKG.hasParameter: MKG.Parameter, MKG.hasMetadata: MKG.Metadata}
    for p, cls in ranges.items():
        problems += [f'{p.split("#")[1]} points at a non-{cls.split("#")[1]}: {o}'
                     for o in g.objects(None, p) if (o, RDF.type, cls) not in g]
    for L in sorted(g.subjects(RDF.type, MKG.Law), key=lambda x: _eq_key(str(g.value(x, MKG.equationNumber)))):
        k = len(list(g.objects(L, MKG.computes)))
        if k != 1:
            problems.append(f'{str(L).split("/kg/")[-1]}: computes {k} variables (schema: exactly 1)')
    for cls in (MKG.Variable, MKG.Parameter):
        for x in g.subjects(RDF.type, cls):
            owners = {s for p in (MKG.computes, MKG.uses, MKG.hasParameter) for s in g.subjects(p, x)}
            if len(owners) != 1:
                problems.append(f'{x} is attached to {len(owners)} laws (v0: exactly 1)')
    return problems

# =============================================================================
# Compare pipeline vs reference (read back from the Turtle files)
# =============================================================================

def compare(paper_dir: str) -> dict:
    """Per law, the symbols on each edge type. 'normalised' counts X_{\\nu} and X_{v} as the same
    symbol (Paddle reads the printed subscript v as the Greek nu)."""
    P = _per_law(Graph().parse(f'{paper_dir}/sup_graph_pipeline.ttl'))
    R = _per_law(Graph().parse(f'{paper_dir}/sup_graph_reference.ttl'))
    norm = lambda syms: sorted(s.replace('\\nu', 'v') for s in syms)
    empty = {e: [] for e in EDGES} | {'latex': None}
    same_read = same_norm = same_latex = 0
    differ = []
    for n in sorted(R):
        p, r = P.get(n, empty), R[n]
        same_read += all(p[e] == r[e] for e in EDGES)
        same_latex += p['latex'] == r['latex']
        if all(norm(p[e]) == norm(r[e]) for e in EDGES):
            same_norm += 1
            continue
        diff = {e: {'only_pipeline': sorted(set(norm(p[e])) - set(norm(r[e]))),
                    'only_reference': sorted(set(norm(r[e])) - set(norm(p[e])))}
                for e in EDGES if norm(p[e]) != norm(r[e])}
        differ.append({'eq': n, 'differences': diff, 'latex_pipeline': p['latex'], 'latex_reference': r['latex']})
    return {'laws': len(R), 'identical_edges_as_read': same_read, 'identical_edges_normalised': same_norm,
            'identical_latex_string': same_latex, 'laws_that_differ': differ}


def _per_law(g: Graph) -> dict:
    out = {}
    for L in g.subjects(RDF.type, MKG.Law):
        n = int(str(g.value(L, MKG.equationNumber)))
        out[n] = {'latex': str(g.value(L, MKG.latex))}
        for e in EDGES:
            out[n][e] = sorted(str(g.value(x, MKG.symbol)) for x in g.objects(L, MKG[e]))
    return out

# =============================================================================
# Run
# =============================================================================

def main(paper_dir: str):
    """All Sups of the paper into one file per variant. The reference graph (gold_laws.json) exists for
    single-model papers only."""
    cfg = json.load(open(f'{paper_dir}/paper.json'))
    sups = models(cfg)
    gold_ok = os.path.exists(f'{paper_dir}/gold_laws.json') and len(sups) == 1
    variants = [('pipeline', False)] + ([('reference', True)] if gold_ok else [])
    for tag, gold in variants:
        whole, reports = Graph(), []
        for m in sups:
            g, flags, summary = build(paper_dir, cfg, gold, m)
            problems = validate(g)
            whole += g
            reports.append({'sup': m['id'], 'summary': summary, 'flags': flags, 'schema_problems': problems})
            print(f"{tag:9}  {m['id']}: {summary['laws']} laws, {summary['variable_nodes']} Variable and "
                  f"{summary['parameter_nodes']} Parameter nodes, {len(flags)} flags, {len(problems)} schema problems")
            for p in problems:
                print('           ', p)
        for prefix, ns in g.namespaces():
            whole.bind(prefix, ns)
        whole.serialize(f'{paper_dir}/sup_graph_{tag}.ttl', format='turtle')
        report = reports[0] if len(reports) == 1 else {'sups': reports}
        json.dump(report, open(f'{paper_dir}/sup_graph_{tag}_report.json', 'w'), indent=1, ensure_ascii=False)
        print(f'           -> sup_graph_{tag}.ttl')
    if len(variants) == 2:
        c = compare(paper_dir)
        json.dump(c, open(f'{paper_dir}/graph_comparison.json', 'w'), indent=1, ensure_ascii=False)
        print(f"compare    {c['identical_edges_normalised']}/{c['laws']} laws identical "
              f"(as read: {c['identical_edges_as_read']}); differ: {[d['eq'] for d in c['laws_that_differ']]}"
              f" -> graph_comparison.json")


if __name__ == '__main__':
    main(sys.argv[1])
