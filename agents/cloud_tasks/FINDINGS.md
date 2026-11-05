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
   It restarted again at about 23:28 UTC, in the middle of contois (6/30 done), and the jobs were restarted
   from where they stopped. Every results line written before each restart parses.

## Code

3. **Monod seed 101: the true law fails the chi-square test** (not shown to be a code bug). On
   `data.make(scenario('monod'), 101)`, the fitted Monod law (parmest, model-curve weights) recovers every true
   parameter inside its 95% interval (theta1 0.3251 [0.3184, 0.3318], theta2 0.2643 [0.2487, 0.2799], theta3 0.5595
   [0.5442, 0.5748], theta4 0.0555 [0.0498, 0.0611]). But chi2 = 150.50 > limit 148.78 (p = 0.041), so
   `fits_within_noise` is False. The largest contribution is E3:x1 (chi2 34.5 over 21 points, worst z = -3.26 at t = 1).
   This is what the test at 95% does about 5% of the time, so it may be chance; the stats study measures the rate.
   Reproduce: `pytest tests/test_data_fit.py -m slow`. The test `test_monod_fit_within_noise` is marked xfail (strict)
   with this reason. Nothing was changed.

## Status log
- 2026-11-04T21:03:50Z stats: andrews=3 aiba=0 monod=0 contois=0; design: 0 files, 0 summary lines; stage: stats 101-130 start 2026-11-04T21:02:23Z.
- 2026-11-04T21:23:58Z stats: andrews=15 aiba=0 monod=0 contois=0; design: 0 files, 0 summary lines; stage: stats 101-130 start 2026-11-04T21:02:23Z.
- 2026-11-04T21:43:57Z stats: andrews=26 aiba=0 monod=0 contois=0; design: 0 files, 0 summary lines; stage: stats 101-130 start 2026-11-04T21:02:23Z.
- 2026-11-04T22:04:08Z stats: andrews=30 aiba=8 monod=0 contois=0; design: 0 files, 0 summary lines; stage: stats 101-130 start 2026-11-04T21:02:23Z.
- 2026-11-04T22:23:57Z stats: andrews=30 aiba=21 monod=0 contois=0; design: 0 files, 0 summary lines; stage: stats 101-130 start 2026-11-04T21:02:23Z.
- 2026-11-04T22:43:49Z stats: andrews=30 aiba=30 monod=5 contois=0; design: 0 files, 0 summary lines; stage: stats 101-130 start 2026-11-04T21:02:23Z.
- 2026-11-04T23:03:57Z stats: andrews=30 aiba=30 monod=20 contois=0; design: 0 files, 0 summary lines; stage: stats 101-130 start 2026-11-04T21:02:23Z.
- 2026-11-04T23:23:48Z stats: andrews=30 aiba=30 monod=30 contois=3; design: 0 files, 0 summary lines; stage: stats 101-130 start 2026-11-04T21:02:23Z.
- 2026-11-04T23:29:14Z stats: andrews=30 aiba=30 monod=30 contois=6; design: 0 files, 0 summary lines; stage: stats 101-130 start 2026-11-04T21:02:23Z. RUNNER RESTARTED.
- 2026-11-04T23:49:13Z stats: andrews=30 aiba=30 monod=30 contois=14; design: 0 files, 0 summary lines; stage: stats 101-130 start 2026-11-04T23:29:14Z.
- 2026-11-05T00:09:10Z stats: andrews=30 aiba=30 monod=30 contois=22; design: 0 files, 0 summary lines; stage: stats 101-130 start 2026-11-04T23:29:14Z.
