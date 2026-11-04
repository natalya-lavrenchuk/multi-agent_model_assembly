"""agents.py -- the multi-agent loop: find the rate law that explains the data, from the whole knowledge graph.

    python agents.py runs/andrews_seed1.json [--rounds 3] [--per-round 3] [--dry-run] [--confirm]
    python agents.py --graph                       # the loop as a Mermaid diagram

The balance equations are given (Asprey & Macchietto's biomass and substrate balances); the growth rate law in
them is unknown. Three agents (LLM calls with a fixed JSON answer format) and the code tools between them:

  Hypothesis agent   reads the process description, the model, every rate law the graph holds
                     (kg_query.find_rate_laws) and the feedback so far; chooses up to N new laws to try, each
                     with a reason.
  Modeler agent      for each chosen law: the law is transcribed (translate.py), then this agent maps the law's
                     variables onto the model's (S -> x2, X -> x1). Code builds the candidate and runs Layer 1
                     (candidates.swap); if the checks fail, the agent sees the problems and gets one retry.
  [code]             every candidate that passes is fitted to all experiments (fit.py: parmest, chi-square,
                     confidence intervals, correlations, Akaike weights, where the misfit is).
  Statistical agent  reads the fit reports and decides: accept one candidate, search again (with feedback for
                     the hypothesis agent: what kind of misfit, where), or stop as inconclusive.

The loop is a LangGraph state graph (make_graph): hypothesis -> build_and_fit -> statistician, and back to
hypothesis while the statistical agent asks for a new search, until a candidate is accepted, the agent stops, or
the round limit.

--confirm adds a confirmation round after an acceptance: the hypothesis agent picks the strongest rivals of the
accepted law (same mechanism in another form, or another mechanism with similar curves), they are built and fitted,
and the statistical agent gives a verdict: confirmed, switch, or indistinguishable (all adequate candidates named,
with an experiment that would separate them). Code computes the same verdict by a fixed rule (Akaike weight >= 0.9
among the adequate candidates) and the log records whether the agent agrees. Written to runs/agents_<data>_confirm.json. Install once: pip install "langgraph==0.6.11" (the last version for Python 3.9). Everything is written to runs/agents_<data file>.json
for scoring: decisions with reasons, candidates, fits, LLM calls and tokens, time.
The agents never see the truth file.
"""
from __future__ import annotations
import os
import sys
# macOS: the fits run in forked processes (fit.py). After an HTTPS call (urllib loads Apple's system frameworks),
# fork() can abort unless this is set before Python starts, so the script restarts itself once with it set.
if __name__ == '__main__' and sys.platform == 'darwin' and os.environ.get('OBJC_DISABLE_INITIALIZE_FORK_SAFETY') != 'YES':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.execv(sys.executable, [sys.executable] + sys.argv)
import json
import os
import sys
import time
from typing import TypedDict

import candidates
import fit
import kg_query
import problem
import translate
from extractor import llm
from langgraph.graph import END, START, StateGraph

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')

# Every LLM call of a run, with its tokens, also when the answer came from the cache (the tokens it cost when it
# was made). So a run's cost is known whatever was cached: cost per run is one of the thesis metrics.
CALLS: list = []
_cached_ask = llm.cached_ask


def _logged_ask(paper_dir, system, user, schema, name, effort='low', label='', refresh=False, max_tokens=None,
                keep_effort=False):
    import hashlib
    eff = effort if keep_effort else (llm.EFFORT or effort)
    key = hashlib.sha256(json.dumps([llm.MODEL, eff, system, user, schema], ensure_ascii=False).encode()).hexdigest()[:16]
    path = f'{paper_dir}/llm_cache/{name}_{key}.json'
    was_cached = os.path.exists(path) and not refresh
    ans = _cached_ask(paper_dir, system, user, schema, name, effort, label=label, refresh=refresh,
                      max_tokens=max_tokens, keep_effort=keep_effort)
    usage = {}
    if os.path.exists(path):
        try:
            usage = json.load(open(path)).get('usage', {})
        except (OSError, ValueError):
            pass
    CALLS.append({'name': name, 'label': label, 'cached': was_cached, 'dry_run': ans is None,
                  'prompt_tokens': usage.get('prompt_tokens', 0), 'completion_tokens': usage.get('completion_tokens', 0),
                  'estimate_tokens': (len(system) + len(user)) // 4})
    return ans


llm.cached_ask = _logged_ask             # translate.py calls llm.cached_ask too, so its calls are counted as well
BASE_SUP = 'asprey_2000_monod_kinetics'   # set per run from the problem record (problem.py): balances kept
RATE = 'r'                                # set per run: the model symbol the searched laws must supply
PROBLEM: dict = {}

# ---------------------------------------------------------------------------------------------------- prompts
HYPOTHESIS_T = """You are the hypothesis agent in a team that builds mechanistic process models from the literature.
The model's balance equations are given. One term is unknown: <<UNKNOWN_TEXT>>. Your job is to choose
candidate rate laws for <<UNKNOWN>> from a knowledge graph of published models, so the team can fit them to the data.
- Choose laws whose mechanism could explain the process and the feedback: <<HINT:mechanisms>>. Prefer laws with different mechanisms over near-duplicates of each
  other or of laws already tried.
- Only choose laws whose variables the model has<<HINT:variables>>
- Look at the summary of the measurements (<<HINT:data>>) and use the statistical agent's feedback: it says how the
  earlier candidates failed.
- Give each choice a short reason. Choose at most the number asked for; fewer is fine."""

MODELER_T = """You are the modeler agent. A published rate law is being put into a process model. Map each variable of
the law onto the model symbol that is the same physical quantity. Parameters of the law are not mapped: they
become new parameters of the model. The quantity the law computes must map to the model's rate symbol.
- Map by meaning, using the descriptions and units, not by letter: <<HINT:modeler_example>>.
- Several law symbols may map to the same model symbol if they are the same quantity printed two ways.
- If a law variable has no counterpart in the model, map it to "none" and say why in the note.
If the code checks failed on an earlier mapping, you are shown the problems: fix the mapping if you can."""

STATISTICIAN_T = """You are the statistical agent. You judge candidate models fitted to the experimental data by weighted
least squares. For each candidate you get:
- chi2 and its 95% limit: above the limit means lack of fit, the model is systematically wrong, not just noisy.
- per-parameter estimate, 95% interval and t-value: t below 1 means the data cannot pin that parameter down.
- strongly correlated parameter pairs (|correlation| > 0.95): this data separates the two only poorly. Report
  it as a caveat (and a reason for a more informative experiment), not as a reason to reject a model whose
  t-values are all above 1.
- Akaike weight: relative probability among the candidates fitted on the same data.
- for misfitting models: where the misfit is (experiment, output, over- or under-prediction, worst time).
Rules: accept only a candidate that fits within the noise and has every parameter estimated precisely (every
t-value above 1). If several
qualify, prefer the higher Akaike weight; if they are practically tied, prefer fewer parameters.
If none fits within the noise but one is MARGINAL (just outside the 95% limit, inside the 99% limit), has every
t-value above 1 and is clearly the best of all fitted candidates (Akaike weight at least 0.9), you may accept it
as marginal (decision accept_marginal): the best available model, with a slight lack of fit that may be noise.
Say so in the feedback. If none qualifies,
ask for a new search and tell the hypothesis agent what the misfits suggest about the missing mechanism (be
specific: which experiment, <<HINT:misfit>>, over- or under-prediction). Stop as inconclusive only if the
data cannot distinguish the remaining plausible candidates."""

CHALLENGER_T = """You are the hypothesis agent in a team that builds mechanistic process models from the literature.
A candidate rate law has just been accepted because it fits the data within the noise. Before the team reports it,
test whether the data really single it out. From the knowledge graph, choose the strongest rivals of the accepted
law that have not been tried yet:
- laws with the same kind of mechanism in a different functional form (for example <<HINT:challenger_example>>), since these are the hardest to tell apart;
- a law with a different mechanism that could produce similar curves;
- only laws whose variables the model has; not laws already tried; no exact duplicates of the accepted law.
Give each choice a short reason. Choose at most the number asked for; fewer is fine; none if no real rival exists."""

VERDICT = """You are the statistical agent, confirming a decision. A candidate was accepted; then its strongest rivals
were fitted to the same data. You get every fitted candidate's report and a table, computed by code, of the
adequate candidates (fit within the noise, every parameter precise, no fitting warnings) with their Akaike weights
among the adequate ones; if none fits within the noise, the table lists the marginal ones instead (inside the 99%
limit), marked so, and the verdict then applies to them (say in the reasoning that the fit is marginal). Decide:
- confirmed: the accepted candidate is the only adequate one, or it has Akaike weight of at least 0.9 among them;
- switch: another candidate is adequate and clearly better (weight at least 0.9); name it as final;
- indistinguishable: two or more candidates are adequate and none reaches 0.9. Name all of them, say what they
  share and how they differ, and describe an experiment that would separate them (which input, which range, which
  measurement), based on where their mechanisms differ.
'final' is the candidate label to report (for indistinguishable, the one with the highest weight)."""

HYPOTHESIS = MODELER = STATISTICIAN = CHALLENGER = ''    # rendered from the templates for each run (use_problem)


def use_problem(p: dict) -> None:
    """Set the run's problem: the prompts' domain words, the unknown symbol and the base model."""
    global HYPOTHESIS, MODELER, STATISTICIAN, CHALLENGER, RATE, BASE_SUP, PROBLEM
    HYPOTHESIS, MODELER = problem.render(HYPOTHESIS_T, p), problem.render(MODELER_T, p)
    STATISTICIAN, CHALLENGER = problem.render(STATISTICIAN_T, p), problem.render(CHALLENGER_T, p)
    RATE, BASE_SUP, PROBLEM = p['unknown'], p['base_model'], p


def label(s: str | None) -> str:
    """'C17 (muloiwa_2020 eq 11)' -> 'C17': agents sometimes add the law's name to a candidate label."""
    import re
    m = re.search(r'C(\d+)', s or '')
    return f'C{m.group(1)}' if m else (s or '')


S_HYP = {'type': 'object', 'additionalProperties': False, 'required': ['candidates', 'note'],
         'properties': {'candidates': {'type': 'array', 'items': {
             'type': 'object', 'additionalProperties': False, 'required': ['id', 'reason'],
             'properties': {'id': {'type': 'integer'}, 'reason': {'type': 'string'}}}},
             'note': {'type': 'string'}}}
S_MAP = {'type': 'object', 'additionalProperties': False, 'required': ['mapping', 'note'],
         'properties': {'mapping': {'type': 'array', 'items': {
             'type': 'object', 'additionalProperties': False, 'required': ['law_symbol', 'model_symbol'],
             'properties': {'law_symbol': {'type': 'string'}, 'model_symbol': {'type': 'string'}}}},
             'note': {'type': 'string'}}}
S_STAT = {'type': 'object', 'additionalProperties': False, 'required': ['decision', 'chosen', 'feedback', 'reasoning'],
          'properties': {'decision': {'type': 'string', 'enum': ['accept', 'accept_marginal', 'search_again',
                                                                  'stop_inconclusive']},
                         'chosen': {'type': 'string'}, 'feedback': {'type': 'string'},
                         'reasoning': {'type': 'string'}}}

S_VERDICT = {'type': 'object', 'additionalProperties': False,
             'required': ['verdict', 'final', 'adequate', 'reasoning', 'next_experiment'],
             'properties': {'verdict': {'type': 'string', 'enum': ['confirmed', 'switch', 'indistinguishable']},
                            'final': {'type': 'string'}, 'adequate': {'type': 'array', 'items': {'type': 'string'}},
                            'reasoning': {'type': 'string'}, 'next_experiment': {'type': 'string'}}}
CONFIRM_WEIGHT = 0.9                                   # Akaike weight that counts as "clearly better"


# ---------------------------------------------------------------------------------------------- context texts
def model_text(base: dict) -> str:
    inv = {v: k for k, v in base['names'].items()}
    lines = ['Balance equations (given):']
    for L in base['laws']:
        if L.get('target') != RATE and 'rhs' in L:
            lines.append(f"  d{L['target']}/dt = {L['rhs_text']}")
    lines.append('Model symbols:')
    for p in sorted(set(base['names'].values()) - set(base['parameters'])):
        S = base['symbols'][inv[p]]
        what = 'unknown, to be supplied by a rate law' if p == RATE else ''
        lines.append(f"  {p}: {S.get('description', '')} [{S.get('unit') or '?'}] {what}")
    lines.append('  t: time [h]')
    return '\n'.join(lines)


def laws_text(rates: list) -> str:
    return '\n'.join(f"[{i}] {r['source']} eq {r['eq']}: computes {r['computes']} ({r['computes_description']}); "
                     f"roles: {', '.join(sorted(r['roles'])) or '-'}\n     {r['latex']}" for i, r in enumerate(rates))


def tried_text(log: dict) -> str:
    if not log['candidates']:
        return 'Nothing tried yet.'
    out = []
    for c in log['candidates']:
        verdict = ('could not be built: ' + '; '.join(c['problems'])) if not c['built'] else \
                  ('fit failed' if not c['fit'].get('ok') else
                   f"chi2 {c['fit']['chi2']:.0f} (limit {c['fit']['chi2_limit']:.0f}), "
                   f"{'fits' if c['fit']['fits_within_noise'] else 'lack of fit'}")
        out.append(f"  [{c['rate_id']}] {c['law']}: {verdict}")
    return 'Already tried:\n' + '\n'.join(out)


def data_summary(data: dict) -> str:
    """What the measurements show, in a few numbers per experiment (code, no fitting): the inputs, and where
    biomass and substrate start, peak and end. Lets the hypothesis agent look at the data before choosing, as a
    modeller would (decision 3 Nov, option B: before, round 1 saw no data and was the same for every data set)."""
    lines = ['Measurements (summary; every hour, g/L):']
    for e in data['experiments']:
        u1 = ', '.join(f"{v} from {a} h" for a, v in e['inputs']['u1'])
        u2 = ', '.join(f"{v} from {a} h" for a, v in e['inputs']['u2'])
        x1, x2, t = e['y']['x1'], e['y']['x2'], e['samples']
        i1, i2 = max(range(len(x1)), key=x1.__getitem__), max(range(len(x2)), key=x2.__getitem__)
        lines.append(f"  {e['name']} ({e['tf']} h): u1 (1/h) {u1}; u2 (g/L) {u2}. Biomass x1 {x1[0]:.2f} -> {x1[-1]:.2f}"
                     f" (max {x1[i1]:.2f} at {t[i1]} h); substrate x2 {x2[0]:.2f} -> {x2[-1]:.2f} (max {x2[i2]:.2f}"
                     f" at {t[i2]} h).")
    return '\n'.join(lines)


# --------------------------------------------------------------------------------------------------- the steps
def hypothesis(data, base, rates, log, per_round) -> list:
    user = (f"Process: {data['description']}\n\n{data_summary(data)}\n\n{model_text(base)}\n\nRate laws in the knowledge graph:\n"
            f"{laws_text(rates)}\n\n{tried_text(log)}\n\nFeedback from the statistical agent: "
            f"{log['feedback'] or 'none yet'}\n\nChoose at most {per_round} new laws to try.")
    ans = llm.cached_ask(CACHE, HYPOTHESIS, user, S_HYP, 'hypothesis', label='hypothesis agent')
    if ans is None:
        return []
    done = {c['rate_id'] for c in log['candidates']}
    picks = [c for c in ans['candidates'] if 0 <= c['id'] < len(rates) and c['id'] not in done][:per_round]
    log['steps'].append({'agent': 'hypothesis', 'round': log['round'], 'answer': ans})
    return picks


def model_symbols_text(base: dict) -> str:
    inv = {v: k for k, v in base['names'].items()}
    return '\n'.join(f"  {p}: {base['symbols'][inv[p]].get('description', '')} [{base['symbols'][inv[p]].get('unit') or '?'}]"
                     for p in sorted(set(base['names'].values()) - set(base['parameters'])))


def modeler(base: dict, rate: dict, log: dict) -> dict:
    law_tr = translate.translate_model(kg_query.get_law(rate['iri']))
    L = law_tr['laws'][0]
    if 'rhs' not in L or L['problems']:
        return {'built': False, 'problems': [f'transcription: {p}' for p in L['problems']] or ['no transcription']}
    inv = {v: k for k, v in law_tr['names'].items()}
    params = set(law_tr['parameters'])
    law_vars = sorted({str(s) for s in L['rhs'].free_symbols} - params - {'t'} | {L['target']})
    rows = '\n'.join(f"  {v}: {law_tr['symbols'][inv[v]].get('description', '')} [{law_tr['symbols'][inv[v]].get('unit') or '?'}]"
                     + ('   <- the quantity the law computes' if v == L['target'] else '') for v in law_vars)
    prs = ', '.join(sorted(params)) or '-'
    problems, cand = [], None
    for attempt in (1, 2):
        user = (f"Law ({rate['source']} eq {rate['eq']}):\n  {rate['latex']}\n  as transcribed: "
                f"{L['target']} = {L['rhs_text']}\nLaw variables to map:\n{rows}\nLaw parameters (not mapped): {prs}\n\n"
                f"Model symbols (the rate symbol is {RATE}):\n{model_symbols_text(base)}"
                + (f"\n\nThe previous mapping failed the code checks:\n" + '\n'.join(f'  - {p}' for p in problems)
                   if problems else ''))
        ans = llm.cached_ask(CACHE, MODELER, user, S_MAP, 'modeler', label=f"modeler agent ({rate['source']} eq {rate['eq']})")
        if ans is None:
            return {'built': False, 'problems': ['dry run']}
        mapping = {candidates.resolve_symbol(law_tr, m['law_symbol']): m['model_symbol'] for m in ans['mapping']
                   if m['model_symbol'] != 'none'}
        log['steps'].append({'agent': 'modeler', 'round': log['round'], 'law': rate['iri'], 'attempt': attempt,
                             'answer': ans})
        cand = candidates.swap(base, rate['iri'], mapping, law_tr)
        if cand['ok']:
            return {'built': True, 'cand': cand, 'problems': [], 'attempts': attempt}
        problems = cand['problems']
    return {'built': False, 'problems': problems, 'attempts': 2}


def statistician(data: dict, log: dict) -> dict:
    fitted = [c for c in log['candidates'] if c['built']]
    blocks = []
    for c in fitted:
        f = c['fit']
        blocks.append(f"Candidate C{c['rate_id']} ({c['law']}, rate law r = {c['rate_law']}):\n" +
                      (fit.report(f) if f.get('ok') else f"  fit failed: {f.get('error')}"))
    not_built = [f"  [{c['rate_id']}] {c['law']}: {'; '.join(c['problems'])}" for c in log['candidates'] if not c['built']]
    user = (f"Process: {data['description']}\n\nFitted candidates (all on the same data, "
            f"{fitted[0]['fit'].get('n', '?') if fitted else '?'} readings):\n\n" + '\n\n'.join(blocks) +
            (f"\n\nCandidates that could not be built:\n" + '\n'.join(not_built) if not_built else '') +
            f"\n\nRound {log['round']} of {log['max_rounds']}. Decide. If you accept, 'chosen' is the candidate "
            f"label, e.g. C{fitted[0]['rate_id'] if fitted else 0}.")
    ans = llm.cached_ask(CACHE, STATISTICIAN, user, S_STAT, 'statistician', label='statistical agent')
    log['steps'].append({'agent': 'statistician', 'round': log['round'], 'answer': ans})
    return ans or {'decision': 'stop_inconclusive', 'chosen': '', 'feedback': '', 'reasoning': 'dry run'}


def adequacy(log: dict) -> list:
    """Code, no LLM: the candidates that are adequate (fit within the noise, every t > 1, no parameter that ran
    away), with their Akaike weights among the adequate ones only, best first. If none is, the MARGINAL ones
    (outside the 95% chi-square limit, inside the 99% limit; each flagged 'marginal': True). The same law printed in two papers
    (Andrews in Zentou eq 5 and Muloiwa eq 6) gives the same fit: such candidates count as one model, with
    'same_as' listing the others, so the weight is not split between copies."""
    import math
    out = _adequate(log, marginal=False)
    if not out:                           # none within the noise: the marginal ones (95-99%), flagged as such
        out = _adequate(log, marginal=True)
    if out:
        best = min(o['aic'] for o in out)
        w = [math.exp(-(o['aic'] - best) / 2) for o in out]
        for o, wi in zip(out, w):
            o['weight'] = wi / sum(w)
    return sorted(out, key=lambda o: -o['weight'])


def _adequate(log: dict, marginal: bool) -> list:
    out = []
    for c in log['candidates']:
        f = c.get('fit') or {}
        ok_fit = f.get('fits_within_noise') if not marginal else f.get('fit_class') == 'marginal'
        if c['built'] and f.get('ok') and ok_fit \
                and not any('ran to' in w for w in f.get('warnings', [])) \
                and all(t > 1 for t in f['t_values'].values()):
            o = {'label': f"C{c['rate_id']}", 'law': c['law'], 'aic': f['aic'], 'chi2': f['chi2'], 'p': f['p'],
                 'same_as': [], 'marginal': marginal}
            twin = next((x for x in out if x['p'] == o['p'] and abs(x['chi2'] - o['chi2']) <= 1e-3 * max(1, x['chi2'])), None)
            if twin:
                twin['same_as'].append(f"{o['label']} ({o['law']})")
            else:
                out.append(o)
    return out


def code_verdict(adequate: list, accepted: str) -> dict:
    """The same decision by the fixed rule, to check the agent against."""
    if not adequate:
        return {'verdict': 'none adequate', 'final': ''}
    top = adequate[0]
    if top['weight'] >= CONFIRM_WEIGHT:
        return {'verdict': 'confirmed' if top['label'] == accepted else 'switch', 'final': top['label']}
    return {'verdict': 'indistinguishable', 'final': top['label']}


def challenger(data, base, rates, log, per_round) -> list:
    acc = log['decision']['chosen']
    acc_c = next((c for c in log['candidates'] if f"C{c['rate_id']}" == acc), None)
    acc_text = (f"{acc} = {acc_c['law']}, rate law r = {acc_c.get('rate_law')}" if acc_c else acc)
    user = (f"Process: {data['description']}\n\n{data_summary(data)}\n\n{model_text(base)}\n\nAccepted: {acc_text}\n\n"
            f"Rate laws in the knowledge graph:\n{laws_text(rates)}\n\n{tried_text(log)}\n\n"
            f"Choose at most {per_round} rivals to test.")
    ans = llm.cached_ask(CACHE, CHALLENGER, user, S_HYP, 'challenger', label='hypothesis agent (confirmation)')
    if ans is None:
        return []
    done = {c['rate_id'] for c in log['candidates']}
    picks = [c for c in ans['candidates'] if 0 <= c['id'] < len(rates) and c['id'] not in done][:per_round]
    log['steps'].append({'agent': 'challenger', 'round': log['round'], 'answer': ans})
    return picks


def verdict(data: dict, log: dict) -> dict:
    adequate = adequacy(log)
    table = '\n'.join(f"  {o['label']} ({o['law']}){' MARGINAL' if o.get('marginal') else ''}: chi2 {o['chi2']:.1f}, {o['p']} parameters, AIC {o['aic']:.1f}, "
                       f"Akaike weight among adequate {o['weight']:.3f}"
                       + (f"; same fit as {', '.join(o['same_as'])} (the same law printed elsewhere)" if o['same_as'] else '')
                       for o in adequate) or '  none'
    blocks = [f"Candidate C{c['rate_id']} ({c['law']}, rate law r = {c['rate_law']}):\n" +
              (fit.report(c['fit']) if c['fit'].get('ok') else f"  fit failed: {c['fit'].get('error')}")
              for c in log['candidates'] if c['built']]
    user = (f"Process: {data['description']}\n\nAccepted before confirmation: {log['decision']['chosen']}\n\n"
            f"Adequate candidates (code):\n{table}\n\nAll fitted candidates:\n\n" + '\n\n'.join(blocks))
    ans = llm.cached_ask(CACHE, VERDICT, user, S_VERDICT, 'verdict', label='statistical agent (confirmation)')
    ans = ans or {'verdict': 'confirmed', 'final': log['decision']['chosen'], 'adequate': [], 'reasoning': 'dry run',
                  'next_experiment': ''}
    log['steps'].append({'agent': 'verdict', 'round': log['round'], 'answer': ans})
    ans = {**ans, 'final': label(ans['final']), 'adequate': [label(a) for a in ans['adequate']]}
    rule = code_verdict(adequate, log['decision']['chosen'])
    log['confirmation'] = {'agent': ans, 'rule': rule, 'adequate': adequate,
                           'agent_agrees_with_rule': ans['verdict'] == rule['verdict'] and ans['final'] == rule['final']}
    return ans


# ------------------------------------------------------------------------------------------------------ loop
class State(TypedDict):
    """What flows between the nodes of the graph. 'log' is the record of the whole run."""
    data: dict
    base: dict
    rates: list
    per_round: int
    picks: list
    confirm: bool
    log: dict


def node_hypothesis(s: State) -> dict:
    log = s['log']
    log['round'] += 1
    return {'picks': hypothesis(s['data'], s['base'], s['rates'], log, s['per_round']), 'log': log}


def node_build_and_fit(s: State) -> dict:
    """Modeler agent for each pick, then code: build, Layer 1, fit."""
    log, rates = s['log'], s['rates']
    for p in s['picks']:
        rate = rates[p['id']]
        m = modeler(s['base'], rate, log)
        entry = {'round': log['round'], 'rate_id': p['id'], 'law': f"{rate['source']} eq {rate['eq']}",
                 'iri': rate['iri'], 'reason': p['reason'], 'built': m['built'], 'problems': m['problems']}
        if m['built']:
            c = m['cand']
            entry['rate_law'] = c['laws'][-1]['rhs_text']
            entry['mapping'] = c['mapping']
            print(f"  round {log['round']}: [{p['id']}] fitting (limit {fit.TIME_LIMIT:.0f} s)...", flush=True)
            entry['fit'] = fit.fit(c, s['data'])
            if entry['fit'].get('cached'):
                print('    (fit cached)')
        log['candidates'].append(entry)
        print(f"  round {log['round']}: [{p['id']}] {entry['law']}: " +
              (f"chi2 {entry['fit'].get('chi2', float('nan')):.0f}" if m['built'] and entry['fit'].get('ok')
               else (('fit failed: ' + entry['fit'].get('error', '')[:70]) if m['built'] else 'not built: ' + '; '.join(m['problems']))))
    fit.compare([c['fit'] for c in log['candidates'] if c['built']])
    return {'log': log}


def node_statistician(s: State) -> dict:
    log = s['log']
    dec = statistician(s['data'], log)
    dec = {**dec, 'chosen': label(dec['chosen'])}
    print(f"  statistical agent: {dec['decision']} {dec['chosen']}  {dec['feedback'][:150]}")
    log['feedback'] = dec['feedback']
    log['last_decision'] = dec
    if dec['decision'] in ('accept', 'accept_marginal', 'stop_inconclusive'):
        log['decision'] = dec
    return {'log': log}


def node_challenger(s: State) -> dict:
    log = s['log']
    log['round'] += 1
    log['phase'] = 'confirm'
    return {'picks': challenger(s['data'], s['base'], s['rates'], log, s['per_round']), 'log': log}


def node_verdict(s: State) -> dict:
    log = s['log']
    v = verdict(s['data'], log)
    r = log['confirmation']['rule']
    print(f"  confirmation: {v['verdict']} -> {v['final']}  (rule: {r['verdict']} -> {r['final']})  "
          f"{v.get('next_experiment', '')[:150]}")
    return {'log': log}


def after_challenger(s: State) -> str:
    return 'build_and_fit' if s['picks'] else 'verdict'


def after_build(s: State) -> str:
    return 'verdict' if s['log'].get('phase') == 'confirm' else 'statistician'


def after_hypothesis(s: State) -> str:
    return 'build_and_fit' if s['picks'] else END                     # nothing new to try: stop


def after_statistician(s: State) -> str:
    log = s['log']
    if log['decision'] is not None and log['decision']['decision'] in ('accept', 'accept_marginal') and s.get('confirm'):
        return 'challenger'                                              # test the accepted law against its rivals
    if log['decision'] is not None or log['round'] >= log['max_rounds']:
        return END
    return 'hypothesis'                                                  # search again with the feedback


def make_graph():
    """hypothesis -> build_and_fit -> statistician -> (accept / stop / round limit: END; else back to hypothesis).
    With confirmation: an accept goes to challenger -> build_and_fit -> verdict -> END."""
    g = StateGraph(State)
    g.add_node('hypothesis', node_hypothesis)
    g.add_node('build_and_fit', node_build_and_fit)
    g.add_node('statistician', node_statistician)
    g.add_node('challenger', node_challenger)
    g.add_node('verdict', node_verdict)
    g.add_edge(START, 'hypothesis')
    g.add_conditional_edges('hypothesis', after_hypothesis, ['build_and_fit', END])
    g.add_conditional_edges('build_and_fit', after_build, ['statistician', 'verdict'])
    g.add_conditional_edges('statistician', after_statistician, ['hypothesis', 'challenger', END])
    g.add_conditional_edges('challenger', after_challenger, ['build_and_fit', 'verdict'])
    g.add_edge('verdict', END)
    return g.compile()


def run(data: dict, max_rounds: int = 3, per_round: int = 3, tag: str = 'agents', confirm: bool = False) -> dict:
    t0, usage0, calls0 = time.time(), dict(llm.USAGE), len(CALLS)
    log = {'tag': tag, 'mode': 'multi', 'seed': data.get('seed'), 'max_rounds': max_rounds, 'round': 0, 'feedback': '',
           'candidates': [], 'steps': [], 'decision': None, 'confirm': confirm}
    use_problem(problem.of(data))
    log['problem'] = PROBLEM
    state = {'data': data, 'base': translate.translate(BASE_SUP), 'rates': kg_query.find_rate_laws(PROBLEM['unknown_dimension']),
             'per_round': per_round, 'picks': [], 'confirm': confirm, 'log': log}
    log = make_graph().invoke(state, {'recursion_limit': 4 * max_rounds + 10})['log']
    c = log.get('confirmation')
    log['final'] = ({'model': c['agent']['final'], 'status': c['agent']['verdict']} if c else
                    {'model': (log['decision'] or {}).get('chosen', ''), 'status': (log['decision'] or {}).get('decision', 'none')})
    log['final']['marginal'] = bool((log['decision'] or {}).get('decision') == 'accept_marginal' or
                                    (c and c['adequate'] and c['adequate'][0].get('marginal')))
    log['seconds'] = round(time.time() - t0, 1)
    log['llm'] = {k: llm.USAGE[k] - usage0.get(k, 0) for k in llm.USAGE}
    calls = CALLS[calls0:]
    log['llm_calls'] = calls
    log['llm_total'] = {'calls': len(calls), 'agent_calls': sum(1 for c in calls if c['name'] != 'translate'),
                        'prompt_tokens': sum(c['prompt_tokens'] for c in calls),
                        'completion_tokens': sum(c['completion_tokens'] for c in calls)}
    return log


def save(log: dict) -> str:
    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
    path = os.path.join(HERE, 'runs', f"{log['tag']}.json")
    clean = json.loads(json.dumps(log, default=str))                 # SymPy objects -> text
    json.dump(clean, open(path, 'w'), indent=1, ensure_ascii=False)
    return path


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--graph' in args:                                            # the loop as a Mermaid diagram (thesis figure)
        print(make_graph().get_graph().draw_mermaid())
        sys.exit()
    llm.DRY_RUN = '--dry-run' in args
    rounds = int(args[args.index('--rounds') + 1]) if '--rounds' in args else 3
    per = int(args[args.index('--per-round') + 1]) if '--per-round' in args else 3
    confirm = '--confirm' in args
    d = json.load(open(args[0]))
    tag = 'agents_' + os.path.basename(args[0]).replace('.json', '') + ('_confirm' if confirm else '')
    out = run(d, rounds, per, tag=tag, confirm=confirm)
    out['data_file'] = os.path.basename(args[0])
    print('decision:', json.dumps(out['decision'], ensure_ascii=False))
    if out.get('confirmation'):
        c = out['confirmation']
        print('confirmation:', c['agent']['verdict'], '->', c['agent']['final'], '| adequate:',
              ', '.join(f"{o['label']} {o['law']} (weight {o['weight']:.2f})" for o in c['adequate']) or 'none',
              '| agent agrees with the rule:', c['agent_agrees_with_rule'])
        if c['agent'].get('next_experiment'):
            print('experiment to separate them:', c['agent']['next_experiment'])
    print('final:', out['final'])
    print('wrote', save(out))
    print(llm.usage_line())
