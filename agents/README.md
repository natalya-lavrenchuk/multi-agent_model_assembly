# agents

The model-building side of the thesis. It reads the knowledge graph in `../extractor_agent/kg/shared_graph.ttl`
and reuses the extractor's LLM calls (`../extractor_agent/extractor/llm.py`, same cache and dry run).

| File | What it does | LLM? |
|---|---|---|
| `problem.py` | The problem record: base model, unknown term, measured states, inputs, role mapping, prompt hints. Written by hand for the yeast case (stored in each data file); the interpreter will write it | no |
| `kg_query.py` | Fixed queries on the graph: `get_model(sup)`, `get_law(iri)`, `find_rate_laws()` (every law computing a per-time rate, with roles). Uses accepted misprint corrections | no |
| `translate.py` | A law's LaTeX -> SymPy, then code checks (symbols, target, kind vs the graph's form, units) | one call per law, cached |
| `symbolic.py` | Layer 1: units balance (unknown units inferred), each quantity computed once | no |
| `candidates.py` | Put a rate law from anywhere in the graph into a base model's balances, with a symbol mapping; checks the result | no |
| `build.py` | SymPy -> Pyomo.DAE, simulate one experiment with ipopt | no |
| `reference.py` | Asprey's two models written by hand (SciPy): the answer key. `python reference.py` is the week-1 test | no |
| `data.py` | Synthetic experiments from a hidden true model (scenarios monod, contois, andrews; 3 experiments, 20 h, 5% noise) | no |
| `fit.py` | Layer 3: parmest fit, confidence intervals, t-values, correlations, chi-square, where the misfit is, Akaike weights | no |
| `agents.py` | The loop as a LangGraph state graph: hypothesis agent -> modeler agent + code (build, Layer 1, fit) -> statistical agent, back to hypothesis while it asks for a new search; `--confirm` adds rivals + verdict | 3 agent roles |
| `single.py` | Baseline B1: one agent does every role, same tools and checks, one call per round | 1 agent |
| `b0.py` | Baseline B0: prompts to paste into a normal chat LLM, and scoring of its answers | you chat |
| `score.py` | Scores run logs against the hidden truth (same law / same mechanism / wrong), tokens, time | no |
| `batch.py` | The experiment: every data set x multi and single, resumable, retries, then scoring | via the above |
| `survey.py` | Layer 1 (units, structure, positivity) over all 42 rate laws of the graph | translations only |
| `stats_study.py` | The statistics alone over many noise seeds: false rejection, interval coverage, discrimination | no |

## Run

    cd ~/Documents/Data_Science_Masters/Thesis/agents
    source ../.venv/bin/activate
    pip install "langgraph==0.6.11"        # once; the last LangGraph version for Python 3.9
    export LLM_API_KEY=...
    export IPOPT=~/micromamba/envs/ipopt/bin/ipopt   # conda-forge ipopt 3.14 (the IDAES Mac build cannot run parmest)

Week 1 test (translation and building, Asprey's two models):

    python reference.py                                   # every line should say PASS

Make hidden-truth data (code only; scenarios monod, contois, andrews, aiba, notingraph):

    python data.py andrews --seed 1                       # runs/andrews_seed1.json (+ .truth.json, never read by agents)
    python data.py monod --seed 1

Fit chosen models by hand (code only):

    python fit.py runs/monod_seed1.json asprey_2000_monod_kinetics asprey_2000_contois_kinetics

The agents:

    python agents.py runs/andrews_seed1.json --dry-run    # sizes only, nothing sent
    python agents.py runs/andrews_seed1.json              # writes runs/agents_andrews_seed1.json
    python agents.py runs/andrews_seed1.json --confirm    # plus a confirmation round -> runs/agents_andrews_seed1_confirm.json
    python agents.py --graph                              # the loop as a Mermaid diagram (thesis figure)

Each candidate is fitted in a child process with a time limit (default 300 s, `export FIT_TIME_LIMIT=600` to change;
hard kill 60 s later). A law the solvers get stuck on, or crash on, is logged as "did not finish" / "crashed", not as
wrong, and the run goes on. Solver output of the last fit: `runs/solver_output_last_fit.log`. Solver warnings (LSODA, ipopt) are hidden while fitting.
Fit results are cached in `cache/fits/` (keyed by equations, data, start and settings), so a rerun does not refit.

The experiment (3 Nov plan: 5 truths x 3 seeds x multi and single = 30 runs):

    python batch.py --dry-run                             # what will run, expected calls and tokens
    caffeinate -i python batch.py                         # the real run; resumable (run it again to continue)
    python score.py runs/batch/                           # the table again, any time

Baseline B0 (by hand, see the top of b0.py): `python b0.py prompts`, paste b0/caseNN.txt into a chat, save its
JSON to b0/answers/caseNN_chat.json (or _code.json), then `python b0.py score`.

Fitting settings (env): `FIT_TIME_LIMIT` (300 s per candidate), `FIT_WORKERS` (starting points in parallel,
default up to 4), `FIT_PREFIT_RTOL` / `FIT_PREFIT_DIFF_STEP` (search tolerance 1e-6, step 1e-3), `FIT_ISOLATE=0`
(fit in the main process if forked processes ever fail on a computer; then no hard time limit).

The log holds every agent answer with its reasons, every candidate (mapping, Layer 1 result, fit), the decision,
the LLM calls and tokens, and the time.
