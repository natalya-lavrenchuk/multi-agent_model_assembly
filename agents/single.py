"""single.py -- baseline B1: ONE agent does every role, with the same tools and the same code checks.

    python single.py runs/andrews_seed1.json [--rounds 3] [--per-round 3] [--dry-run] [--confirm]

The question this answers: does splitting the work across agents (hypothesis, modeler, statistical) help, or
would one LLM call per round, with the same information and the same tools, do as well?

Kept the same as agents.py: the data, the description, the base model, the full list of rate laws in the graph
(here with each law's variables and their descriptions and units, which the modeler agent would otherwise see),
the code steps (translation, symbol swap, Layer 1, fit, Akaike weights, adequacy rule), the round limit, the
number of candidates per round, the confirmation round and its fixed-rule check, the log format.
Different: one system prompt covering all the roles, and one call per round that both judges the fits so far and,
if it does not accept, proposes the next laws WITH their symbol mappings. A mapping that fails the code checks is
not retried within the round (the agents' modeler gets one retry); the agent sees the problems next round.
Calls per run: about one per round, plus two for the confirmation (rivals, verdict).
Written to runs/single_<data>[_confirm].json, scored by score.py like the multi-agent runs.
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

import agents                      # the shared texts, the logged LLM call, adequacy rule and verdict rule
import candidates
import fit
import kg_query
import problem
import translate
from extractor import llm
from langgraph.graph import END, START, StateGraph

CACHE = agents.CACHE

SINGLE_T = """You are a process modeling agent. You build mechanistic process models from the literature and test them
against data. The model's balance equations are given; one term is unknown: <<UNKNOWN_TEXT>>. You do
every step yourself, with code tools that build, check and fit what you propose:
1. Choose candidate laws for <<UNKNOWN>> from a knowledge graph of published models. Choose laws whose mechanism
   could explain the process (<<HINT:mechanisms>>); prefer different
   mechanisms over near-duplicates; only laws whose variables the model has<<HINT:variables>>
2. For each chosen law, map each variable of the law (use the law's own symbols, as printed in the list) onto the
   model symbol that is the same physical quantity, by meaning, not by letter. The quantity the law computes maps
   to <<UNKNOWN>>. Parameters are not mapped: they become new parameters fitted to the data.
3. Judge the fitted candidates. For each you get chi2 and its 95% limit (above the limit: lack of fit, the model
   is systematically wrong), each parameter's estimate, 95% interval and t-value (t below 1: the data cannot pin
   it down), strongly correlated parameter pairs (|correlation| > 0.95: the data separates the two only poorly;
   a caveat to report, not a reason to reject a model whose t-values are all above 1), the Akaike weight, and
   where a misfit is.
   Accept only a candidate that fits within the noise and has every parameter estimated precisely (every
   t-value above 1); if several
   qualify, prefer the higher Akaike weight, and if practically tied, fewer parameters. If none fits within the
   noise but one is MARGINAL (just outside the 95% limit, inside the 99% limit), has every t-value above 1 and
   is clearly the best of all fitted (Akaike weight at least 0.9), you may accept it as marginal (decision
   accept_marginal): the best available model, with a slight lack of fit that may be noise. If none qualifies,
   use the misfits to choose the next laws (which experiment, <<HINT:misfit>>, over- or under-prediction).
   Stop as inconclusive only if the data cannot distinguish the remaining plausible candidates."""

CONFIRM = """Now test your accepted law before reporting it: choose its strongest rivals that have not been tried
(the same kind of mechanism in another functional form; or another mechanism that could give similar curves),
with their symbol mappings. Choose none if no real rival exists. 'decision' must be 'accept' and 'chosen' the
law you accepted."""

VERDICT = """Its rivals have now been fitted to the same data. You also get a table, computed by code, of the
adequate candidates (fit within the noise, every parameter precise, no fitting warnings) with their Akaike weights
among the adequate ones; if none fits within the noise, the table lists the marginal ones instead (inside the 99%
limit), marked so, and the verdict then applies to them. Decide:
- confirmed: the accepted candidate is the only adequate one, or it has Akaike weight of at least 0.9 among them;
- switch: another candidate is adequate and clearly better (weight at least 0.9); name it as final;
- indistinguishable: two or more are adequate and none reaches 0.9: name all, say how they differ, and describe an
  experiment that would separate them (which input, which range, which measurement).
'final' is the candidate label to report (for indistinguishable, the one with the highest weight)."""

S_SINGLE = {'type': 'object', 'additionalProperties': False,
            'required': ['decision', 'chosen', 'reasoning', 'candidates'],
            'properties': {
                'decision': {'type': 'string', 'enum': ['accept', 'accept_marginal', 'search_again', 'stop_inconclusive']},
                'chosen': {'type': 'string'}, 'reasoning': {'type': 'string'},
                'candidates': {'type': 'array', 'items': {
                    'type': 'object', 'additionalProperties': False, 'required': ['id', 'reason', 'mapping'],
                    'properties': {'id': {'type': 'integer'}, 'reason': {'type': 'string'},
                                   'mapping': {'type': 'array', 'items': {
                                       'type': 'object', 'additionalProperties': False,
                                       'required': ['law_symbol', 'model_symbol'],
                                       'properties': {'law_symbol': {'type': 'string'},
                                                      'model_symbol': {'type': 'string'}}}}}}}}}


def laws_with_variables(rates: list) -> str:
    """The graph's rate laws, each with its variables (symbol, description, unit), from the graph (no LLM)."""
    out = []
    for i, r in enumerate(rates):
        law = kg_query.get_law(r['iri'])
        own = [s for s, S in law['symbols'].items() if S.get('kind') == 'Variable']
        vs = '; '.join(f"{s}: {law['symbols'][s].get('description', '')[:60]} [{law['symbols'][s].get('unit') or '?'}]"
                       for s in own)
        out.append(f"[{i}] {r['source']} eq {r['eq']}: computes {r['computes']}\n     {r['latex']}\n"
                   f"     variables: {vs or '-'}")
    return '\n'.join(out)


def fits_text(log: dict) -> str:
    if not log['candidates']:
        return 'Nothing tried yet.'
    blocks = []
    for c in log['candidates']:
        if not c['built']:
            blocks.append(f"Candidate [{c['rate_id']}] {c['law']}: could not be built: {'; '.join(c['problems'])}")
            continue
        f = c['fit']
        blocks.append(f"Candidate C{c['rate_id']} ({c['law']}, rate law r = {c['rate_law']}):\n" +
                      (fit.report(f) if f.get('ok') else f"  fit failed: {f.get('error')}"))
    return 'Tried so far (all fitted on the same data):\n\n' + '\n\n'.join(blocks)


def ask(s: 'State', extra: str, schema: dict, name: str, label: str) -> dict | None:
    log = s['log']
    user = (f"Process: {s['data']['description']}\n\n{agents.data_summary(s['data'])}\n\n{agents.model_text(s['base'])}\n\nRate laws in the knowledge "
            f"graph:\n{s['laws_text']}\n\n{fits_text(log)}\n\nRound {log['round']} of {log['max_rounds']}. "
            f"Propose at most {s['per_round']} laws per round. " + extra)
    return llm.cached_ask(CACHE, s['system'], user, schema, name, label=label)


def build_and_fit(s: 'State', picks: list) -> None:
    log, rates = s['log'], s['rates']
    done = {c['rate_id'] for c in log['candidates'] if c['built']}   # a law that failed to build may be proposed again
    for p in [p for p in picks if 0 <= p['id'] < len(rates) and p['id'] not in done][:s['per_round']]:
        rate = rates[p['id']]
        law_tr = translate.translate_model(kg_query.get_law(rate['iri']))
        L = law_tr['laws'][0]
        entry = {'round': log['round'], 'rate_id': p['id'], 'law': f"{rate['source']} eq {rate['eq']}",
                 'iri': rate['iri'], 'reason': p['reason'], 'built': False, 'problems': []}
        if 'rhs' not in L or L['problems']:
            entry['problems'] = [f'transcription: {q}' for q in L['problems']] or ['no transcription']
        else:
            mapping = {candidates.resolve_symbol(law_tr, m['law_symbol']): m['model_symbol'] for m in p['mapping']
                       if m['model_symbol'] not in ('none', '')}  # printed symbol (or a variant) -> plain name
            cand = candidates.swap(s['base'], rate['iri'], mapping, law_tr)
            entry['mapping'] = mapping
            if cand['ok']:
                entry.update(built=True, rate_law=cand['laws'][-1]['rhs_text'])
                print(f"  round {log['round']}: [{p['id']}] fitting (limit {fit.TIME_LIMIT:.0f} s)...", flush=True)
                entry['fit'] = fit.fit(cand, s['data'])
            else:
                entry['problems'] = cand['problems']
        log['candidates'].append(entry)
        print(f"  round {log['round']}: [{p['id']}] {entry['law']}: " +
              (f"chi2 {entry['fit'].get('chi2', float('nan')):.0f}" if entry['built'] and entry['fit'].get('ok')
               else ('fit failed' if entry['built'] else 'not built: ' + '; '.join(entry['problems']))[:160]))
    fit.compare([c['fit'] for c in log['candidates'] if c['built']])


class State(TypedDict):
    system: str
    data: dict
    base: dict
    rates: list
    laws_text: str
    per_round: int
    confirm: bool
    log: dict


def node_agent(s: State) -> dict:
    """One call: judge what was fitted (if anything) and, unless accepting or stopping, propose the next laws."""
    log = s['log']
    log['round'] += 1
    last = log['round'] > log['max_rounds']                    # only a decision left, no new laws
    extra = ('Decide on the candidates above; no new laws (round limit reached).' if last else
             'Decide, and if you do not accept or stop, propose the next laws with their mappings.' if log['candidates']
             else "Nothing is fitted yet: choose the first laws to try (decision 'search_again').")
    ans = ask(s, extra, S_SINGLE, 'single', 'single agent')
    log['steps'].append({'agent': 'single', 'round': log['round'], 'answer': ans})
    if ans is None:
        log['decision'] = {'decision': 'stop_inconclusive', 'chosen': '', 'reasoning': 'dry run'}
        return {'log': log}
    ans = {**ans, 'chosen': agents.label(ans['chosen'])}
    print(f"  single agent: {ans['decision']} {ans['chosen']}  {ans['reasoning'][:140]}")
    if ans['decision'] in ('accept', 'accept_marginal', 'stop_inconclusive') and log['candidates'] or last:
        log['decision'] = {'decision': ans['decision'] if not last or ans['decision'] != 'search_again' else
                           'stop_inconclusive', 'chosen': ans['chosen'], 'reasoning': ans['reasoning']}
        return {'log': log}
    build_and_fit(s, ans['candidates'])
    return {'log': log}


def node_confirm(s: State) -> dict:
    log = s['log']
    log['phase'] = 'confirm'
    ans = ask(s, CONFIRM + f" You accepted {log['decision']['chosen']}.", S_SINGLE, 'single', 'single agent (rivals)')
    log['steps'].append({'agent': 'single_rivals', 'round': log['round'], 'answer': ans})
    if ans:
        log['round'] += 1
        build_and_fit(s, ans['candidates'])
    adequate = agents.adequacy(log)
    table = '\n'.join(f"  {o['label']} ({o['law']}){' MARGINAL' if o.get('marginal') else ''}: chi2 {o['chi2']:.1f}, {o['p']} parameters, AIC {o['aic']:.1f}, "
                      f"Akaike weight among adequate {o['weight']:.3f}" for o in adequate) or '  none'
    v = ask(s, VERDICT + f"\n\nAccepted before confirmation: {log['decision']['chosen']}\n\nAdequate candidates "
            f"(code):\n{table}", agents.S_VERDICT, 'single', 'single agent (verdict)')
    v = v or {'verdict': 'confirmed', 'final': log['decision']['chosen'], 'adequate': [], 'reasoning': 'dry run',
              'next_experiment': ''}
    log['steps'].append({'agent': 'single_verdict', 'round': log['round'], 'answer': v})
    v = {**v, 'final': agents.label(v['final']), 'adequate': [agents.label(a) for a in v['adequate']]}
    rule = agents.code_verdict(adequate, log['decision']['chosen'])
    log['confirmation'] = {'agent': v, 'rule': rule, 'adequate': adequate,
                           'agent_agrees_with_rule': v['verdict'] == rule['verdict'] and v['final'] == rule['final']}
    print(f"  confirmation: {v['verdict']} -> {v['final']}  (rule: {rule['verdict']} -> {rule['final']})")
    return {'log': log}


def after_agent(s: State) -> str:
    log = s['log']
    if log['decision'] is not None:
        return 'confirm' if s['confirm'] and log['decision']['decision'] in ('accept', 'accept_marginal') else END
    return 'agent'


def make_graph():
    g = StateGraph(State)
    g.add_node('agent', node_agent)
    g.add_node('confirm', node_confirm)
    g.add_edge(START, 'agent')
    g.add_conditional_edges('agent', after_agent, ['agent', 'confirm', END])
    g.add_edge('confirm', END)
    return g.compile()


def run(data: dict, max_rounds: int = 3, per_round: int = 3, tag: str = 'single', confirm: bool = False) -> dict:
    t0, calls0 = time.time(), len(agents.CALLS)
    log = {'tag': tag, 'mode': 'single', 'seed': data.get('seed'), 'max_rounds': max_rounds, 'round': 0,
           'candidates': [], 'steps': [], 'decision': None, 'confirm': confirm}
    p = problem.of(data)
    agents.use_problem(p)                       # same problem record and rate symbol as the multi-agent system
    log['problem'] = p
    rates = kg_query.find_rate_laws(p['unknown_dimension'])
    state = {'system': problem.render(SINGLE_T, p), 'data': data, 'base': translate.translate(p['base_model']), 'rates': rates,
             'laws_text': laws_with_variables(rates), 'per_round': per_round, 'confirm': confirm, 'log': log}
    log = make_graph().invoke(state, {'recursion_limit': 4 * max_rounds + 10})['log']
    c = log.get('confirmation')
    log['final'] = ({'model': c['agent']['final'], 'status': c['agent']['verdict']} if c else
                    {'model': (log['decision'] or {}).get('chosen', ''), 'status': (log['decision'] or {}).get('decision', 'none')})
    log['final']['marginal'] = bool((log['decision'] or {}).get('decision') == 'accept_marginal' or
                                    (c and c['adequate'] and c['adequate'][0].get('marginal')))
    log['seconds'] = round(time.time() - t0, 1)
    calls = agents.CALLS[calls0:]
    log['llm_calls'] = calls
    log['llm_total'] = {'calls': len(calls), 'agent_calls': sum(1 for x in calls if x['name'] != 'translate'),
                        'prompt_tokens': sum(x['prompt_tokens'] for x in calls),
                        'completion_tokens': sum(x['completion_tokens'] for x in calls)}
    return log


if __name__ == '__main__':
    args = sys.argv[1:]
    llm.DRY_RUN = '--dry-run' in args
    rounds = int(args[args.index('--rounds') + 1]) if '--rounds' in args else 3
    per = int(args[args.index('--per-round') + 1]) if '--per-round' in args else 3
    confirm = '--confirm' in args
    d = json.load(open(args[0]))
    tag = 'single_' + os.path.basename(args[0]).replace('.json', '') + ('_confirm' if confirm else '')
    out = run(d, rounds, per, tag=tag, confirm=confirm)
    out['data_file'] = os.path.basename(args[0])
    print('final:', out['final'])
    print('wrote', agents.save(out))
    print(llm.usage_line())
