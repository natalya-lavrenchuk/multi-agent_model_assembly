"""supmeta.py -- what the paper says about each Sup, written last, from the whole paper.

    python -m extractor.supmeta papers/kontoravdi_2010              # run (answer cached)
    python -m extractor.supmeta papers/kontoravdi_2010 --dry-run

Run after symbols.py and laws.py. Writes papers/<id>/sup_meta.json (one entry per Sup of the paper),
which review.html shows and graph.py writes on the Sup's Metadata.

No fixed fields. Linking Sups later (by search, by an LLM, by a person) needs as much as the paper gives
about the model, and a fixed list of keys (system, unit operation, operating mode ...) keeps only what the
list foresaw. So the LLM reads the whole paper and returns:
    description   a paragraph: what is modelled, how, for what, on which data
    facts         as many short statements about this model as the paper supports, each with a topic the
                  LLM chooses in plain words ('organism', 'scale', 'feeding strategy', 'parameter
                  estimation', 'validation', 'limitation' ...) and the paper's exact words as quote
By code, no LLM: states (what the ODEs compute) and inputs (variables no law computes).
Code checks every quote against the paper: found -> basis 'stated', else 'inferred' (shown for information).
Older sup_meta.json files with the fixed fields are read as facts (facts_of), so nothing breaks before a rerun.
"""
from __future__ import annotations
import json
import os
import sys

from extractor import llm
from extractor.classify import squash
from extractor.graph import corrected
from extractor.paper import load_model, load_snippets, models, paper_text

PROMPT = """You record what a published paper says about one mechanistic process model (a Sup), so that the
model can later be found, compared and linked with models from other papers and other fields of chemical
engineering (reactors, separations, particle processes, bioprocesses, transport, thermodynamics ...). Nobody
knows yet which details will matter for that, so record as much as the paper supports, not a fixed form.

You get the whole paper (main text, tables, figure captions), the model's name, its laws (each with its form
and a one-line description), its states and inputs, and its model-wide assumptions.

description  One paragraph (4-8 sentences), plain terms, no equation numbers: what system is modelled, what
             the model predicts, how it is built (main mechanisms), how the process is run, what data it was
             built and tested on, and what the authors use it for.

facts        Short, self-contained statements about THIS model and the process and data behind it, one idea
             each, with numbers and units where the paper gives them. Questions to cover, whatever the field,
             as far as the paper answers them:
               What physical, chemical or biological system is modelled, with which materials or species?
               In what equipment, at what scale, and how is it operated (conditions, inputs, regime)?
               What kind of model is it: dynamic or steady state, lumped or spatially distributed, and how
               detailed (in the paper's own terms: unstructured, macro-kinetic, population balance, ...)?
               How big is it: how many states, equations and parameters, and how many were estimated?
               What phenomena and mechanisms does the model represent, and what does it predict? Name
               kinetic and transport forms by their standard names when the paper does (Monod, Langmuir,
               Arrhenius, non-competitive inhibition, first-order lag, ...).
               What experiments or data was it built on: what was measured, how, how often, how much?
               How were its parameters obtained, with what method or software, and how certain are they?
               How well does it fit or predict, how was it validated, and where does it fail?
               Over what conditions is it claimed or shown to hold?
               Which earlier models or works does it build on or compare with, and with what result? Name
               each by its first author and year as listed in the paper's references (e.g. "Xu et al.
               1999"), so it can be matched to other papers.
               What is it meant to be used for? Are the code or data available?
             And anything else the paper says that someone reusing or comparing this model would want.
             For each fact:
             topic  one or two plain words you choose for what the fact is about, in the paper's field. Keep
                    topics broad and reuse them: several facts share a topic ("feeding" for the feed
                    concentration, the feed schedule and the pulses; not three topics). Usually 8-15 topics.
                    Broad topics do not mean fewer facts: keep each fact to one idea (one sentence the paper
                    supports, so it can be quoted) and keep the details (instruments, methods, numbers,
                    results). Several facts under one topic is normal.
             text   the fact in one sentence.
             quote  the paper's exact words that state it (a phrase or sentence, copied exactly), or null if you
                    infer it from the paper without it being said.
             Refer to "the paper", never to "the text provided". Do not restate equations or list symbols, do not repeat the model-wide assumptions, and do not
             add knowledge the paper does not support. Do not list individual parameter values (they are
             recorded with each parameter); a summary of them is fine ("12 of 16 parameters within 10%").
             Record that something is missing only when it matters for reusing the model ("no independent
             validation data"), not as a placeholder ("no limitation reported"). If the paper has several
             models, give the facts that hold for this one (shared context is fine). Typically 25-45 facts;
             fewer for a short paper."""

FACT = {'type': 'object', 'additionalProperties': False, 'required': ['topic', 'text', 'quote'],
        'properties': {'topic': {'type': 'string'}, 'text': {'type': 'string'}, 'quote': {'type': ['string', 'null']}}}
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['description', 'facts'],
          'properties': {'description': {'type': 'string'}, 'facts': {'type': 'array', 'items': FACT}}}


def describe(paper_dir: str, refresh: bool = False) -> dict:
    cfg = json.load(open(f'{paper_dir}/paper.json'))
    for f in ('symbols.json', 'laws.json'):
        if not os.path.exists(f'{paper_dir}/{f}') and not llm.DRY_RUN:
            raise SystemExit(f'run {f[:-5]}.py first')
    syms, laws = corrected(paper_dir)                  # with the reviewer's corrections
    text = paper_text(paper_dir, references=True)       # the reference list: earlier models by author and year
    found = [squash(x['text']) for x in load_snippets(paper_dir)] + [squash(text)]

    out = {}
    for m in models(cfg):
        formulas, reads, _ = load_model(paper_dir, model=m)
        eqs = [str(f.get('eq')) for f in formulas]
        states, inputs = _states_inputs(eqs, reads, syms)
        lines = [f"({n}) [{r['form']}] {laws.get(n, {}).get('description', '')}" for n, r in zip(eqs, reads)]
        msg = (f"Paper:\n\n{text}\n\n=====\nModel: {m.get('name')}"
               + (' (rate expressions the paper fits on their own, no model around them)' if m.get('free') else '')
               + '\nLaws:\n' + '\n'.join(lines)
               + '\n\nStates (computed by the ODEs):\n' + ('\n'.join(f'- {s}: {d}' for s, d in states) or '- none')
               + '\n\nInputs (variables no law computes):\n' + ('\n'.join(f'- {s}: {d}' for s, d in inputs) or '- none')
               + '\n\nModel-wide assumptions (already recorded):\n'
               + ('\n'.join(f"- {a['text']}" for a in m.get('model_assumptions', [])) or '- none recorded'))
        a = llm.cached_ask(paper_dir, PROMPT, msg, SCHEMA, 'supmeta', 'medium', label=f"sup {m['id']}", refresh=refresh)
        if a is None:
            continue
        facts = []
        for x in a['facts']:
            q = x['quote']
            ok = bool(q) and len(squash(q)) >= 6 and any(squash(q) in t for t in found)
            facts.append({'topic': x['topic'].strip().lower(), 'text': x['text'],
                          'quote': q if ok else None, 'basis': 'stated' if ok else 'inferred'})
        inferred = sum(f['basis'] == 'inferred' for f in facts)
        out[m['id']] = {'name': m.get('name'), 'equations': eqs, 'description': a['description'], 'facts': facts,
                        'states': [s for s, _ in states], 'inputs': [s for s, _ in inputs], 'status': 'pending',
                        'review': [f'{inferred} of {len(facts)} facts inferred (no quote found)'] if inferred else []}
    return {'model': llm.MODEL, 'sups': out}


def facts_of(meta: dict) -> list[dict]:
    """The Sup's facts. An older sup_meta.json (fixed fields) is read as facts, so nothing is lost before a rerun."""
    if 'facts' in meta:
        return meta['facts']
    out = []
    for f, topic in (('system', 'system'), ('unit_operation', 'equipment'), ('operating_mode', 'operating mode'),
                     ('fitted_on', 'data set')):
        x = meta.get(f)
        if isinstance(x, dict) and x.get('value') and x['value'] != 'not stated':
            out.append({'topic': topic, 'text': x['value'], 'quote': x.get('quote'), 'basis': x.get('basis') or 'inferred'})
    out += [{'topic': 'phenomenon', 'text': p, 'quote': None, 'basis': 'inferred'} for p in meta.get('phenomena', [])]
    return out


def _states_inputs(eqs, reads, syms):
    """States: what the ODEs compute. Inputs: symbols no law computes that are variables (not parameters)."""
    computed = {s for r in reads for s in r['lhs']}
    states = list(dict.fromkeys(s for r in reads if r['form'] in ('ODE', 'PDE') for s in r['lhs']))
    inputs = list(dict.fromkeys(s for r in reads for s in r['rhs']
                                if s not in computed and syms.get(s, {}).get('class') == 'variable'))
    d = lambda s: syms.get(s, {}).get('description', '?')
    return [(s, d(s)) for s in states], [(s, d(s)) for s in inputs]


def main(paper_dir: str, refresh: bool):
    result = describe(paper_dir, refresh)
    print(llm.usage_line())
    if llm.DRY_RUN:
        return
    json.dump(result, open(f'{paper_dir}/sup_meta.json', 'w'), indent=1, ensure_ascii=False)
    for sid, e in result['sups'].items():
        topics = sorted({f['topic'] for f in e['facts']})
        print(f"{sid}: {len(e['facts'])} facts ({sum(f['basis'] == 'stated' for f in e['facts'])} quoted), "
              f"{len(topics)} topics: {', '.join(topics)}")
    print(f'-> {paper_dir}/sup_meta.json')


if __name__ == '__main__':
    llm.DRY_RUN = '--dry-run' in sys.argv
    main(sys.argv[1], '--refresh' in sys.argv)
