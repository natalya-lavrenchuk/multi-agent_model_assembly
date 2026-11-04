# Overnight brief (cloud session, 5 Nov 2026)

You are running code-only experiments for an MSc thesis: a multi-agent LLM system that builds mechanistic
models from a literature knowledge graph and verifies them statistically (fits, chi-square, Akaike weights,
model-based design of experiments). The case study is baker's yeast fed-batch growth (Asprey and Macchietto
2000), with synthetic data from a hidden "true" growth law. Read `agents/README.md` first.

This session does NOT run any agent or LLM. There is no API key here and you must not add one. Every LLM
answer the code needs (the law translations) is already in `agents/cache/llm_cache/`. If any command tries to
make an LLM call (a cache miss, or an error about LLM_API_KEY), stop that task and write it down in FINDINGS.md.

## Ground rules

1. Do not change any system code: everything in `agents/*.py` and `extractor_agent/`. The results must come
   from the exact code that is committed (the same code as the agent batch running on the Mac). If something
   looks like a bug, do not fix it: describe it in `agents/cloud_tasks/FINDINGS.md` (file, line, what happens,
   how to reproduce) and carry on with the rest.
2. You may add files only under `agents/cloud_tasks/`, `agents/tests/`, `agents/runs/stats/`,
   `agents/runs/design_study/` and `agents/runs/design_*.json`.
3. Work on a branch `cloud-runs-5nov`. Commit and push results at least every hour (long runs are resumable,
   so a commit mid-run is fine). Never push to main.
4. Do not tune anything: no changes to seeds, tolerances, weights, criteria or candidate laws. The point is to
   measure the system as it is.
5. Keep the knowledge graph read-only.

## Step 0: environment (about 15 minutes)

    bash agents/cloud_tasks/setup.sh            # skip if the environment's setup script already ran it
    export IPOPT=/opt/ipopt/bin/ipopt
    cd agents
    python reference.py                         # every line must say PASS
    nproc; python -c "import scipy, pyomo, sympy; print(scipy.__version__, pyomo.__version__, sympy.__version__)"

Write the output of `pip freeze`, `nproc` and `ipopt --version` to `cloud_tasks/ENV.txt`. If reference.py does
not pass, stop: write what failed to FINDINGS.md, commit, push, and do nothing else.

Run the long jobs in the background with nohup so they survive between your turns, for example
`nohup python stats_study.py run --seeds 101-130 > runs/stats/run_101_130.log 2>&1 &`, and check on them.
With 8 or more cores you may run one job per truth in parallel (`--truths andrews`, etc.); each fit already
uses up to 4 processes for its starts, so do not start more jobs than nproc / 4.

## Task 1: statistics study (code only)

    python stats_study.py run --seeds 101-130          # all four truths; then
    python stats_study.py run --seeds 131-200          # if time allows (resumable)
    python stats_study.py report > runs/stats/report.txt

Leave FIT_WEIGHTS unset (model-curve weights, the default). Results: `runs/stats/<truth>.jsonl`.

## Task 2: experiment design study (code only)

    python cloud_tasks/design_study.py run --seeds 101-112     # 48 cases; then, if time allows:
    python cloud_tasks/design_study.py run --seeds 113-125
    python cloud_tasks/design_study.py report > runs/design_study/report.txt

Time the first case and write the per-case time in FINDINGS.md. If Tasks 1 and 2 both finish, rerun
seeds 101-112 with `--hr` (adds the Hunter-Reiner design to each case).

## Task 3: test suite (while Tasks 1 and 2 run)

Write pytest tests in `agents/tests/` that need no LLM and no network:
- translate: the cached laws translate (Monod, Contois from Asprey 2000; Andrews, Zentou eq 5; Aiba, Muloiwa
  eq 11) and give the expected SymPy expression up to simplification.
- symbolic and candidates: units balance for those four candidate models; `check_positivity` passes them;
  `resolve_symbol` maps \upmu, \mu, mu and the Unicode letter to the same symbol; a law with a wrong unit fails.
- kg_query: `find_rate_laws()` returns the 42 rate laws; corrected laws drop the symbols their correction
  removed.
- data and fit: data.make is reproducible for a fixed seed; on monod seed 101 the Monod fit recovers the true
  parameters inside its 95% intervals and fits within the noise (mark this one slow).
- score: the truth tables give "same law" and "same mechanism" where expected.
Every test must pass on the committed code. A test that fails because the code is wrong goes into FINDINGS.md
and is marked `xfail` with the reason, never "fixed" by changing system code. Add `cloud_tasks/run_tests.sh`.

## At the end

Write `agents/cloud_tasks/REPORT.md`: what ran (seeds, counts, times), the two report tables copied in full,
the test results, and every finding. Plain numbers, no interpretation beyond what the tables show. Commit
and push the branch.
