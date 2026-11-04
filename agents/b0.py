"""b0.py -- baseline B0: what a standard chat LLM does with the same problem. Code only (you do the chatting).

    python b0.py prompts                # writes b0/<case>.txt, one prompt per data set, to paste into a chat
    python b0.py score                  # scores every answer in b0/answers/ against the hidden truth

How to run it (by hand, in a normal subscription chat; a new conversation for each prompt):
  1. Open b0/<case>.txt (e.g. b0/case03.txt), paste ALL of it into a new chat, send.
  2. Variant "chat":  say nothing else. If the chat asks to run code, answer "no code, answer from reasoning".
     Variant "code":  allow code execution / data analysis if the product offers it.
  3. The chat ends its answer with a JSON block. Save that block as b0/answers/<case>_<variant>.json
     (e.g. b0/answers/case03_chat.json). If it gives no JSON, ask once: "Please give the JSON block now."
  4. Note the chat product and model in the JSON's "product" field, by hand, e.g. "ChatGPT, GPT-6 Luna".
Case files are numbered, not named, so nothing in the prompt reveals the truth; b0/key.json (read only by
the scoring) says which case is which data set.

The prompt gives what our agents get, minus the knowledge graph: the process description, the balance equations
with r unknown, the operating conditions, every measurement, and the noise level. It asks for r as a formula in
x1, x2 with parameter values.

Scoring, per answer (code, no LLM):
  as given   chi2 of the answer's own formula and parameter values on the data (is the model it reports right?)
  refitted   the same formula refitted by our fit.py (is the form right, even if the numbers were not?):
             chi2 vs its 95% limit, t-values, and the Akaike weight against the true law refitted the same way
  form       is the formula the true law (numerically identical predictions after refitting, to 1e-3 relative),
             and its mechanism, which you label by hand in the JSON field "mechanism_label" after reading it
The numbers go next to the agents' in the thesis table; reasons and provenance (which paper, which checks) are
compared in words: the chat gives no source for its law and runs no unit or positivity check.
"""
from __future__ import annotations
import os
import sys
# macOS: the fits run in forked processes (fit.py). After an HTTPS call (urllib loads Apple's system frameworks),
# fork() can abort unless this is set before Python starts, so the script restarts itself once with it set.
if __name__ == '__main__' and sys.platform == 'darwin' and os.environ.get('OBJC_DISABLE_INITIALIZE_FORK_SAFETY') != 'YES':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.execv(sys.executable, [sys.executable] + sys.argv)
import copy
import glob
import json
import os
import sys

import numpy as np
import sympy as sp

import build
import fit
import translate

HERE = os.path.dirname(os.path.abspath(__file__))
B0 = os.path.join(HERE, 'b0')
RUNS = os.path.join(HERE, 'runs')
CASES = [f'{sc}_seed{s}' for s in (1, 2, 3) for sc in ('monod', 'contois', 'andrews', 'aiba', 'notingraph')]

PROMPT = """I have data from a bioprocess and I need a mechanistic model of it.

{description}

The model's balance equations (Asprey and Macchietto's fed-batch yeast model) are fixed:
  dx1/dt = (r - u1 - theta4) * x1
  dx2/dt = -r * x1 / theta3 + u1 * (u2 - x2)
x1: biomass [g/L], x2: substrate (glucose) [g/L], u1: dilution rate [1/h], u2: substrate concentration in the
feed [g/L], theta3: yield, theta4: decay/maintenance rate [1/h]. The specific growth rate r [1/h] is unknown.

Your task: find the growth rate law r (a function of x1 and/or x2) that explains the data, with values for all of
its parameters and for theta3 and theta4. Measurement error is normal with standard deviation 5% of the reading
plus 0.01 g/L.

{experiments}

End your answer with a JSON block in exactly this form (Python syntax for the formula, using x1, x2 and your own
parameter names, plus theta3 and theta4):
```json
{{"rate_law": "mumax*x2/(K + x2)", "parameters": {{"mumax": 0.3, "K": 0.1, "theta3": 0.5, "theta4": 0.05}},
 "mechanism": "one line: what the law represents", "source": "where the law comes from, if from the literature",
 "confidence": "how sure you are that this is the right law and why", "product": ""}}
```"""


def experiments_text(data: dict) -> str:
    out = []
    for e in data['experiments']:
        sw = '; '.join(f"from t = {a} h: {v}" for a, v in e['inputs']['u1'])
        sf = '; '.join(f"from t = {a} h: {v}" for a, v in e['inputs']['u2'])
        rows = '\n'.join(f"  {t:>4} {x1:8.3f} {x2:8.3f}" for t, x1, x2 in zip(e['samples'], e['y']['x1'], e['y']['x2']))
        out.append(f"Experiment {e['name']} ({e['tf']} h). Starting values x1 = {e['x0']['x1']}, x2 = {e['x0']['x2']} g/L.\n"
                   f"  u1 (1/h): {sw}\n  u2 (g/L): {sf}\n  Measurements:\n     t       x1       x2\n{rows}")
    return '\n\n'.join(out)


def write_prompts() -> None:
    os.makedirs(os.path.join(B0, 'answers'), exist_ok=True)
    key = {}
    order = list(CASES)
    np.random.default_rng(2026).shuffle(order)          # case numbers say nothing about the truth
    for i, case in enumerate(order, 1):
        data = json.load(open(os.path.join(RUNS, case + '.json')))
        name = f'case{i:02d}'
        key[name] = case
        open(os.path.join(B0, name + '.txt'), 'w').write(
            PROMPT.format(description=data['description'], experiments=experiments_text(data)))
    json.dump(key, open(os.path.join(B0, 'key.json'), 'w'), indent=1)
    print(f'wrote {len(CASES)} prompts to {B0} (key in b0/key.json; do not paste it)')


def model_from_answer(ans: dict, base: dict) -> dict:
    """Asprey's balances with the answer's r. Parameters: the answer's names (theta3, theta4 kept)."""
    m = copy.deepcopy(base)
    params = sorted(set(ans['parameters']) | {'theta3', 'theta4'})
    loc = {k: sp.Symbol(k) for k in params + ['x1', 'x2']}
    rhs = sp.sympify(ans['rate_law'].replace('^', '**'), locals=loc)
    extra = {str(s) for s in rhs.free_symbols} - set(loc)
    if extra:
        raise ValueError(f'the formula uses symbols that are neither x1, x2 nor its parameters: {sorted(extra)}')
    for L in m['laws']:
        if L.get('target') == 'r':
            L['rhs'], L['rhs_text'] = rhs, sp.sstr(rhs)
    used = {str(s) for L in m['laws'] if 'rhs' in L for s in L['rhs'].free_symbols}
    m['parameters'] = sorted(p for p in params if p in used)
    m['sup'] = 'chat answer: r = ' + sp.sstr(rhs)
    return m


def score_answer(path: str, key: dict, base: dict) -> dict:
    name = os.path.basename(path).replace('.json', '')
    case, variant = name.rsplit('_', 1)
    data_name = key[case]
    data = json.load(open(os.path.join(RUNS, data_name + '.json')))
    truth = json.load(open(os.path.join(RUNS, data_name + '.truth.json')))
    ans = json.load(open(path))
    row = {'answer': name, 'case': case, 'variant': variant, 'data': data_name, 'scenario': truth['scenario'],
           'product': ans.get('product', ''), 'rate_law': ans.get('rate_law'), 'mechanism': ans.get('mechanism', ''),
           'mechanism_label': ans.get('mechanism_label', ''), 'source': ans.get('source', '')}
    try:
        m = model_from_answer(ans, base)
    except Exception as e:
        row['error'] = f'could not use the formula: {e}'
        return row
    given = {p: float(ans['parameters'].get(p, 1.0)) for p in m['parameters']}
    try:
        row['chi2_as_given'] = fit.chi_square(m, data, given)
    except Exception as e:
        row['chi2_as_given'] = None
        row['note_as_given'] = f'did not simulate: {type(e).__name__}'
    f = fit.fit(m, data, given)
    import data as datamod
    t_tr, t_th = datamod.scenario(truth['scenario'])
    t_fit = fit.fit(t_tr, data, t_th or build.paper_values(t_tr))
    row.update({'refit_ok': f.get('ok'), 'chi2_refit': f.get('chi2'), 'chi2_limit': f.get('chi2_limit'),
                'fits_within_noise': f.get('fits_within_noise'),
                'all_t_above_1': all(t > 1 for t in f.get('t_values', {}).values()) if f.get('ok') else None,
                'theta_refit': f.get('theta'), 'true_law_chi2_refit': t_fit.get('chi2')})
    if f.get('ok') and t_fit.get('ok'):
        a, b = f['aic'], t_fit['aic']
        row['akaike_weight_vs_truth'] = float(1 / (1 + np.exp(-(b - a) / 2)))
        same = True
        for e in data['experiments']:
            p1, p2 = build.simulate(m, e, f['theta']), build.simulate(t_tr, e, t_fit['theta'])
            for s in ('x1', 'x2'):
                if max(abs(x - y) / max(abs(y), 1e-2) for x, y in zip(p1[s], p2[s])) > 1e-3:
                    same = False
        row['same_predictions_as_truth'] = same
    return row


if __name__ == '__main__':
    if sys.argv[1:] == ['prompts']:
        write_prompts()
    elif sys.argv[1:] == ['score']:
        key = json.load(open(os.path.join(B0, 'key.json')))
        base = translate.translate('asprey_2000_monod_kinetics')
        rows = [score_answer(p, key, base) for p in sorted(glob.glob(os.path.join(B0, 'answers', '*.json')))]
        for r in rows:
            if 'error' in r:
                print(f"{r['answer']:16} {r['scenario']:10} ERROR {r['error']}")
                continue
            print(f"{r['answer']:16} {r['scenario']:10} r = {r['rate_law'][:40]:40} chi2 given "
                  f"{r['chi2_as_given'] or float('nan'):10.0f}  refit {r['chi2_refit'] or float('nan'):8.1f} "
                  f"(limit {r['chi2_limit'] or 0:.0f}; true law {r['true_law_chi2_refit'] or float('nan'):.1f})  "
                  f"weight vs truth {r.get('akaike_weight_vs_truth', float('nan')):.2f}  "
                  f"same as truth {r.get('same_predictions_as_truth')}")
        json.dump(rows, open(os.path.join(B0, 'scores.json'), 'w'), indent=1, default=str)
        print('wrote', os.path.join(B0, 'scores.json'))
    else:
        print(__doc__)
