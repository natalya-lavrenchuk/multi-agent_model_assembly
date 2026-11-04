"""score.py -- score agent runs against the hidden truth. Code only, no LLM. The only file that reads truths.

    python score.py runs/agents_andrews_seed1_confirm.json        # one run, in detail
    python score.py runs/batch/                                    # every run log in a folder: table + summary
                                                                   # (also writes <folder>/scores.json, .csv)

A run is scored at three levels, strictest first:
  same law       the final model is the true law, or the same equation printed in another paper (FORMS below:
                 Andrews is Zentou 2019 eq 5 and Muloiwa 2020 eq 6; Monod is in four papers)
  same mechanism the final model has the true law's mechanism (MECHANISM below, assigned by hand from the
                 equations, one class per law): e.g. Aiba accepted when the truth is Andrews
  wrong          anything else, including accepting a law when the truth is not in the graph
                 (reported apart: 'nothing accepted (true law closest)', when nothing passed the tests but the
                 true law had the lowest AIC of everything fitted: a near miss of the chi-square test, not a
                 wrong pick; still counted as not right)
"Indistinguishable" verdicts are scored on the agent's own final pick, and separately the run records whether
the truth was among the candidates still adequate (not ruled out). For the truth that is not in the graph, the
right outcome is to accept nothing (verdict none / inconclusive, or no candidate adequate).
Parameter error: largest relative error over the parameters, when the final law is the true one (parameters
matched by name, or by their role in the graph for the same law printed elsewhere).
"""
from __future__ import annotations
import csv
import glob
import json
import os
import sys

KG = 'http://example.org/kg/'


def _iri(path: str) -> str:
    return KG + path


# The same equation in different papers (identical up to parameter names).
FORMS = {
    'Monod': ['asprey_2000_monod_kinetics/law/asprey_2000_eq17', 'zentou_2019_monod/law/zentou_2019_eq3',
              'muloiwa_2020_free_laws/law/muloiwa_2020_eq3', 'lopezmeza_2016/law/lopezmeza_2016_eq3'],
    'Contois': ['asprey_2000_contois_kinetics/law/asprey_2000_eq18', 'muloiwa_2020_free_laws/law/muloiwa_2020_eq9'],
    'Andrews': ['zentou_2019_andrews/law/zentou_2019_eq5', 'muloiwa_2020_free_laws/law/muloiwa_2020_eq6'],
    'Aiba': ['muloiwa_2020_free_laws/law/muloiwa_2020_eq11'],
    'Andrews, cubic term': ['muloiwa_2020_free_laws/law/muloiwa_2020_eq12'],
    'Luong': ['muloiwa_2020_free_laws/law/muloiwa_2020_eq13'],
    'Han-Levenspiel': ['muloiwa_2020_free_laws/law/muloiwa_2020_eq16'],
    'Webb': ['muloiwa_2020_free_laws/law/muloiwa_2020_eq17'],
    'Teissier': ['muloiwa_2020_free_laws/law/muloiwa_2020_eq7'],
    'Moser': ['muloiwa_2020_free_laws/law/muloiwa_2020_eq8'],
    'first order': ['muloiwa_2020_free_laws/law/muloiwa_2020_eq5'],
    'Monod with maintenance': ['muloiwa_2020_free_laws/law/muloiwa_2020_eq14'],
    'logistic': ['muloiwa_2020_free_laws/law/muloiwa_2020_eq15'],
    'logarithmic': ['muloiwa_2020_free_laws/law/muloiwa_2020_eq10'],
}
# Mechanism class of each form (by hand, from the equations).
MECHANISM = {
    'Monod': 'saturation', 'Teissier': 'saturation', 'Moser': 'saturation', 'first order': 'saturation',
    'Contois': 'biomass-dependent saturation',
    'Andrews': 'substrate inhibition', 'Aiba': 'substrate inhibition', 'Andrews, cubic term': 'substrate inhibition',
    'Luong': 'substrate inhibition', 'Han-Levenspiel': 'substrate inhibition', 'Webb': 'substrate inhibition',
    'Monod with maintenance': 'saturation with maintenance', 'logistic': 'biomass limitation',
    'logarithmic': 'empirical', 'not in the graph': 'biomass dependence with substrate inhibition (not in the graph)',
}
FORM_OF = {_iri(p): f for f, ps in FORMS.items() for p in ps}


def form_of(iri: str | None) -> str:
    if iri is None:
        return 'not in the graph'
    return FORM_OF.get(iri, 'other: ' + iri.split('/')[-1])


def mechanism_of(form: str) -> str:
    return MECHANISM.get(form, 'other')


def _truth_for(log_path: str, log: dict) -> dict | None:
    data_name = log.get('data_file') or os.path.basename(log_path)
    base = os.path.basename(data_name)
    for prefix in ('agents_', 'single_'):
        base = base.replace(prefix, '', 1) if base.startswith(prefix) else base
    stem = base.split('_seed')[0] + '_seed' + base.split('_seed')[1].split('_')[0].replace('.json', '')
    for folder in (os.path.dirname(log_path), os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')):
        p = os.path.join(folder, stem + '.truth.json')
        if os.path.exists(p):
            return json.load(open(p))
    return None


def _param_error(final_c: dict, truth: dict) -> float | None:
    if not final_c or not final_c.get('fit', {}).get('ok'):
        return None
    th, true = final_c['fit']['theta'], truth['theta']
    if final_c['iri'] != truth.get('law_iri') or set(th) != set(true):
        return None                       # a twin from another paper: names differ (reported as n/a)
    return max(abs(th[k] - v) / abs(v) for k, v in true.items())


def score(log_path: str) -> dict:
    log = json.load(open(log_path))
    truth = _truth_for(log_path, log)
    if truth is None:
        raise SystemExit(f'no truth file for {log_path}')
    true_form = form_of(truth.get('law_iri'))
    by_label = {f"C{c['rate_id']}": c for c in log['candidates'] if c.get('built')}
    final = log.get('final') or {}
    import re
    m = re.search(r'C(\d+)', final.get('model') or '')            # 'C17 (muloiwa eq 11)' -> 'C17'
    final_c = by_label.get(f'C{m.group(1)}' if m else '')
    status = final.get('status', 'none')
    accepted = final_c is not None and status in ('accept', 'accept_marginal', 'confirmed', 'switch', 'indistinguishable')
    marginal = bool(final.get('marginal')) and accepted
    final_form = form_of(final_c['iri']) if final_c else None
    conf = log.get('confirmation') or {}
    adequate_labels = [o['label'] for o in conf.get('adequate', [])] + \
        [s.split(' ')[0] for o in conf.get('adequate', []) for s in o.get('same_as', [])]
    adequate_forms = sorted({form_of(by_label[l]['iri']) for l in adequate_labels if l in by_label})
    if true_form == 'not in the graph':
        level = 'wrong (forced a law)' if accepted else 'right (nothing accepted)'
    elif not accepted:
        level = 'wrong (nothing accepted)'
    elif final_form == true_form:
        level = 'same law'
    elif mechanism_of(final_form) == mechanism_of(true_form):
        level = 'same mechanism'
    else:
        level = 'wrong'
    tried = [form_of(c['iri']) for c in log['candidates']]
    fitted = [c for c in log['candidates'] if c.get('built') and (c.get('fit') or {}).get('ok')]
    best = min(fitted, key=lambda c: c['fit']['aic']) if fitted else None
    truth_best = best is not None and form_of(best['iri']) == true_form   # lowest AIC among everything fitted
    if level == 'wrong (nothing accepted)' and truth_best:
        level = 'nothing accepted (true law closest)'
    tot = log.get('llm_total') or {}
    return {
        'run': os.path.basename(log_path), 'scenario': truth.get('scenario'), 'seed': truth.get('seed'),
        'mode': log.get('mode', 'multi'), 'true_form': true_form, 'true_mechanism': mechanism_of(true_form),
        'final_status': status, 'final_form': final_form, 'final_mechanism': mechanism_of(final_form) if final_form else None,
        'level': level + (' (marginal fit)' if marginal and level in ('same law', 'same mechanism', 'wrong') else ''),
        'marginal': marginal, 'right': level.startswith('same law') or level.startswith('right'),
        'right_mechanism': level in ('same law', 'same mechanism', 'right (nothing accepted)'),
        'truth_tried': true_form in tried, 'truth_ranked_best': truth_best,
        'best_chi2_vs_limit': [best['fit']['chi2'], best['fit']['chi2_limit']] if best else None, 'truth_adequate': true_form in adequate_forms if conf else None,
        'adequate_forms': adequate_forms, 'n_candidates': len(log['candidates']),
        'rounds': log.get('round'), 'param_max_rel_error': _param_error(final_c, truth),
        'final_chi2': (final_c or {}).get('fit', {}).get('chi2'), 'chi2_limit': (final_c or {}).get('fit', {}).get('chi2_limit'),
        'agent_agrees_with_rule': conf.get('agent_agrees_with_rule'),
        'llm_calls': tot.get('calls'), 'agent_calls': tot.get('agent_calls'),
        'prompt_tokens': tot.get('prompt_tokens'), 'completion_tokens': tot.get('completion_tokens'),
        'seconds': log.get('seconds'), 'next_experiment': (conf.get('agent') or {}).get('next_experiment', ''),
    }


def summary(rows: list) -> list:
    out = []
    keys = sorted({(r['mode'], r['scenario']) for r in rows})
    for mode, sc in keys + [(m, 'ALL') for m in sorted({r['mode'] for r in rows})]:
        rs = [r for r in rows if r['mode'] == mode and (sc == 'ALL' or r['scenario'] == sc)]
        def mean(k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return sum(v) / len(v) if v else None
        out.append({'mode': mode, 'scenario': sc, 'runs': len(rs),
                    'right': sum(r['right'] for r in rs), 'right_mechanism': sum(r['right_mechanism'] for r in rs),
                    'truth_tried': sum(r['truth_tried'] for r in rs),
                    'marginal': sum(r.get('marginal', False) for r in rs),
                    'nothing_accepted_truth_closest': sum(r['level'] == 'nothing accepted (true law closest)' for r in rs),
                    'agrees_with_rule': sum(1 for r in rs if r['agent_agrees_with_rule']),
                    'mean_rounds': mean('rounds'), 'mean_param_error': mean('param_max_rel_error'),
                    'mean_calls': mean('llm_calls'), 'mean_prompt_tokens': mean('prompt_tokens'),
                    'mean_completion_tokens': mean('completion_tokens'), 'mean_seconds': mean('seconds')})
    return out


def _fmt(v, nd=2):
    return '-' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


if __name__ == '__main__':
    target = sys.argv[1]
    if os.path.isdir(target):
        paths = sorted(p for p in glob.glob(os.path.join(target, '*.json'))
                       if not p.endswith(('.truth.json', 'scores.json', 'summary.json')) and
                       os.path.basename(p).startswith(('agents_', 'single_')))
        rows = [score(p) for p in paths]
        for r in rows:
            print(f"{r['mode']:6} {r['scenario']:10} seed {r['seed']}  {r['level']:26} final {_fmt(r['final_form']):22} "
                  f"({r['final_status']}) rounds {r['rounds']}  calls {r['llm_calls']}  "
                  f"tokens {r['prompt_tokens']}+{r['completion_tokens']}  {_fmt(r['seconds'], 0)} s")
        print()
        S = summary(rows)
        for s in S:
            print(f"{s['mode']:6} {s['scenario']:10} right {s['right']}/{s['runs']}  right mechanism "
                  f"{s['right_mechanism']}/{s['runs']}  truth tried {s['truth_tried']}/{s['runs']}  near miss "
                  f"{s['nothing_accepted_truth_closest']}  marginal {s['marginal']}  "
                  f"rounds {_fmt(s['mean_rounds'], 1)}  param err {_fmt(s['mean_param_error'], 3)}  "
                  f"calls {_fmt(s['mean_calls'], 1)}  tokens {_fmt(s['mean_prompt_tokens'], 0)}+"
                  f"{_fmt(s['mean_completion_tokens'], 0)}  {_fmt(s['mean_seconds'], 0)} s")
        json.dump({'runs': rows, 'summary': S}, open(os.path.join(target, 'scores.json'), 'w'), indent=1)
        if rows:
            with open(os.path.join(target, 'scores.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                for r in rows:
                    w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    else:
        print(json.dumps(score(target), indent=1))
