"""batch.py -- run the experiment: every data set x every mode, then score. Resumable; survives a dropped network.

    python batch.py --dry-run          # what would run, and the LLM calls/tokens it is expected to take; sends nothing
    caffeinate -i python batch.py      # the real run (caffeinate keeps the Mac awake; plug it in)
    python batch.py --only multi       # one mode only (multi or single)
    python batch.py --cases andrews_seed1,aiba_seed1

What it does, in order:
  0. makes any missing hidden-truth data set (data.py), then survey.py: translates every rate law of the graph
     once (cached) and writes the Layer 1 survey
  1. for each data set (seed 1 of every truth first, then seed 2, then seed 3) and each mode:
        multi   python agents.py <data> --confirm      the multi-agent system
        single  python single.py <data> --confirm      baseline B1, one agent
     each run in its own process; its log goes to runs/batch/. A finished run is never repeated, so stopping
     and starting again continues where it stopped.
  2. score.py runs/batch/   -> runs/batch/scores.json, scores.csv and the summary table
A run that fails (for example the network is down longer than the LLM code's own retries, about 80 s) is
logged in runs/batch/batch_log.txt and tried again at the end, up to 2 more times, after a pause. Everything it
had already done is cached (LLM answers and fits), so a retry costs only what was missing.
"""
from __future__ import annotations
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
OUT = os.path.join(RUNS, 'batch')
TRUTHS = ['monod', 'contois', 'andrews', 'aiba', 'notingraph']
SEEDS = [1, 2, 3]
MODES = {'multi': 'agents.py', 'single': 'single.py'}
PREFIX = {'multi': 'agents_', 'single': 'single_'}

# Expected tokens per LLM call, from the cached calls of 2-3 Nov (gpt-6-luna, effort high), for the dry-run estimate.
PER_CALL = {'hypothesis': (4100, 1500), 'challenger': (4000, 1800), 'modeler': (430, 240), 'statistician': (1700, 400),
            'verdict': (3000, 600), 'single': (5500, 1600), 'translate': (450, 90)}


def cases(only=None) -> list:
    all_ = [f'{t}_seed{s}' for s in SEEDS for t in TRUTHS]
    return [c for c in all_ if not only or c in only]


def out_path(mode: str, case: str) -> str:
    return os.path.join(OUT, f'{PREFIX[mode]}{case}_confirm.json')


def log(msg: str) -> None:
    line = time.strftime('%H:%M:%S ') + msg
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, 'batch_log.txt'), 'a').write(line + '\n')


def run_one(mode: str, case: str) -> bool:
    data = os.path.join(RUNS, case + '.json')
    t0 = time.time()
    log(f'start {mode} {case}')
    with open(os.path.join(OUT, f'{PREFIX[mode]}{case}.out.txt'), 'w') as f:
        rc = subprocess.call([sys.executable, MODES[mode], data, '--confirm'], cwd=HERE, stdout=f,
                             stderr=subprocess.STDOUT, env={**os.environ, 'OBJC_DISABLE_INITIALIZE_FORK_SAFETY': 'YES'})
    made = os.path.join(RUNS, f'{PREFIX[mode]}{case}_confirm.json')
    if rc == 0 and os.path.exists(made):
        os.replace(made, out_path(mode, case))
        r = json.load(open(out_path(mode, case)))
        tot = r.get('llm_total', {})
        log(f"done  {mode} {case} in {time.time() - t0:.0f} s: final {r['final']}, {tot.get('calls')} LLM calls, "
            f"{tot.get('prompt_tokens')} + {tot.get('completion_tokens')} tokens")
        return True
    tail = open(os.path.join(OUT, f'{PREFIX[mode]}{case}.out.txt')).read().strip().splitlines()[-3:]
    log(f"FAILED {mode} {case} (exit code {rc}) after {time.time() - t0:.0f} s: {' | '.join(tail)}")
    return False


def dry_run(todo: list) -> None:
    """Expected calls per run, from the design (a search round costs 1 hypothesis + 3 modeler + 1 statistician;
    the confirmation 1 challenger + 3 modeler + 1 verdict); between 1 and 3 search rounds."""
    multi_1 = {'hypothesis': 1, 'modeler': 3, 'statistician': 1, 'challenger': 1, 'verdict': 1}
    multi_1['modeler'] += 3
    multi_3 = dict(multi_1, hypothesis=3, statistician=3, modeler=12)
    single_1, single_3 = {'single': 4}, {'single': 6}

    def tokens(calls):
        return (sum(PER_CALL[k][0] * n for k, n in calls.items()), sum(PER_CALL[k][1] * n for k, n in calls.items()))
    lo = hi = (0, 0)
    for mode, case in todo:
        a, b = (tokens(multi_1), tokens(multi_3)) if mode == 'multi' else (tokens(single_1), tokens(single_3))
        lo, hi = (lo[0] + a[0], lo[1] + a[1]), (hi[0] + b[0], hi[1] + b[1])
    n = len(todo)
    print(f'{n} runs to do ({sum(1 for m, _ in todo if m == "multi")} multi, {sum(1 for m, _ in todo if m == "single")} single):')
    for mode, case in todo:
        print(f'   {mode:6} {case}')
    calls_lo = sum(sum(multi_1.values()) if m == 'multi' else 4 for m, _ in todo)
    calls_hi = sum(sum(multi_3.values()) if m == 'multi' else 6 for m, _ in todo)
    print(f'\nexpected LLM calls: {calls_lo} to {calls_hi} (1 to 3 search rounds per run), plus at most ~30 '
          f'translations once (survey)')
    print(f'expected tokens: {lo[0] / 1e3:.0f}k to {hi[0] / 1e3:.0f}k input, {lo[1] / 1e3:.0f}k to {hi[1] / 1e3:.0f}k '
          f'output (output includes the model\'s reasoning tokens)')
    print('cost = input tokens x your input price + output tokens x your output price (per million tokens).')
    print('time: about 5-15 min per run (fits on this computer + LLM replies), so roughly '
          f'{n * 5 / 60:.1f} to {n * 15 / 60:.1f} h.')
    print('(an upper estimate: calls whose answer is already cached cost nothing; e.g. the first hypothesis\n'
          ' call does not depend on the data, so it is asked once for all runs)')


if __name__ == '__main__':
    args = sys.argv[1:]
    only_mode = args[args.index('--only') + 1] if '--only' in args else None
    only_cases = args[args.index('--cases') + 1].split(',') if '--cases' in args else None
    todo = [(m, c) for c in cases(only_cases) for m in MODES if (not only_mode or m == only_mode)
            and not os.path.exists(out_path(m, c))]
    if '--dry-run' in args:
        missing = sorted({c for _, c in todo if not os.path.exists(os.path.join(RUNS, c + '.json'))})
        if missing:
            print(f'data sets to make first (code only, a few seconds each): {", ".join(missing)}\n')
        dry_run(todo)
        sys.exit()
    if not os.environ.get('LLM_API_KEY') and not os.environ.get('OPENAI_API_KEY'):
        sys.exit('set LLM_API_KEY first')
    os.makedirs(OUT, exist_ok=True)
    log(f'batch start: {len(todo)} runs')
    for case in sorted({c for _, c in todo}):                 # hidden-truth data sets not made yet (code only)
        if not os.path.exists(os.path.join(RUNS, case + '.json')):
            truth, seed = case.split('_seed')
            subprocess.call([sys.executable, 'data.py', truth, '--seed', seed], cwd=HERE)
            log(f'made data {case}')
    if not os.path.exists(os.path.join(HERE, 'b0', 'key.json')) and all(
            os.path.exists(os.path.join(RUNS, c + '.json')) for c in cases()):
        subprocess.call([sys.executable, 'b0.py', 'prompts'], cwd=HERE)      # baseline B0 prompts, for later
    if not os.path.exists(os.path.join(RUNS, 'survey_rate_laws.json')):
        log('survey: translating every rate law once')
        subprocess.call([sys.executable, 'survey.py'], cwd=HERE)
    failed = [(m, c) for m, c in todo if not run_one(m, c)]
    for attempt in (1, 2):
        if not failed:
            break
        log(f'retrying {len(failed)} failed runs in 2 min (attempt {attempt})')
        time.sleep(120)
        failed = [(m, c) for m, c in failed if not run_one(m, c)]
    if failed:
        log('still failed: ' + ', '.join(f'{m} {c}' for m, c in failed) + ' (run the batch again later to finish)')
    log('scoring')
    subprocess.call([sys.executable, 'score.py', OUT], cwd=HERE)
    log('batch end')
