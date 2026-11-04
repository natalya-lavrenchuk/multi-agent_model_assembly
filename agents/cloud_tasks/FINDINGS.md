# Findings (cloud session, 4-5 Nov 2026)

## Environment

1. **The environment's setup script failed** (`agents/cloud_tasks/setup.sh`, line 5). The container's Python is 3.13;
   `numpy==1.26.4` and `scipy==1.13.1` have no Python 3.13 wheels, so pip tried to build scipy from source and
   stopped with `Unknown compiler(s): gfortran ...`. Also `micro.mamba.pm` is blocked by this environment's
   network policy (the micromamba download on line 7 fails). Done by hand instead: micromamba from
   `https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64`, then
   `micromamba create -p /opt/ipopt -c conda-forge ipopt=3.14 python=3.9` (Python 3.9.23, the Mac's version) and
   the same pinned packages with `/opt/ipopt/bin/python -m pip install ...`. Every run uses
   `PATH=/opt/ipopt/bin:$PATH`. Versions: `cloud_tasks/ENV.txt`. reference.py: every line PASS, 0 LLM calls.
2. **The container restarted** at about 21:00 UTC on 4 Nov, which killed the background jobs. The last
   result before that was andrews seed 103, at about 20:16. Restarted at 21:03 UTC with
   `cloud_tasks/run_all.sh` (the runs are resumable, so finished seeds are skipped). About 45 minutes of run
   time were lost.

## Code

(none so far)

## Status log
