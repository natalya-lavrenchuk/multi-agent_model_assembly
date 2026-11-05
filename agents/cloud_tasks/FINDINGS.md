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

4. **Design study: every case ends with `KeyError: 'p'`** (a code bug, not fixed). `design.fit_all`
   (`agents/design.py:185-186`) returns each fit with only the keys ok, chi2, chi2_limit, p_value, fit_class, theta,
   t_values, aic, akaike_weight, error, warnings. It drops `p` and `fits_within_noise`. `agents._adequate`
   (`agents/agents.py:333`) reads `f.get('fits_within_noise')`, which is now always None, so no fit counts as
   within the noise. It then falls back to the marginal fits, and for a marginal fit it reads `f['p']`
   (`agents/agents.py:337`), which raises KeyError. When no fit is marginal there is no error, but the outcome is
   'none' even if a rival fits within the noise. So `cloud_tasks/design_study.py`'s `outcome()` cannot score
   design.py's fits. Reproduce: `python cloud_tasks/design_study.py run --seeds 101-101 --truths andrews` gives
   `{"error": "KeyError: 'p'"}` in runs/design_study/summary.jsonl.
   The same thing would happen in agents.py if it ever passes design.fit_all output to adequacy(); not checked.
   Each case takes 393 s (andrews seed 101: 3 fits of 2 rivals plus the D ranking of the designs).
   What was done: design.main writes every fit set to `runs/design_<truth>_seed<n>.json` before the error, so Task 2
   still runs as written, and `cloud_tasks/design_rescore.py` (new file) scores those saved outputs. It puts back
   `p = len(theta)` and `fits_within_noise = chi2 <= chi2_limit`, which are fit.py's own definitions (fit.py lines
   410 and 480), and applies the unchanged rule (design_study.outcome). No fit is redone and no system code is changed.
   design_study's `--hr` pass cannot run: the error comes before the HR design. Also, every case is already in
   summary.jsonl, so the pass skips them all. `design_rescore.py run --hr` runs the D_HR-best design through
   design.main and scores it the same way.
   Result of design_study.py over seeds 101-112 (48 cases): 11 cases ended with `KeyError: 'p'`. The other 37
   finished without an error but have outcome 'none' for before, designed and random alike, which is the silent
   form of the same bug. Re-scored by design_rescore.py: 48 cases, 0 errors (table in REPORT.md). Time per case
   (design_study.py): mean 355 s, from 247 s to 716 s (4.7 h in total).

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
- 2026-11-05T00:29:11Z stats: andrews=30 aiba=30 monod=30 contois=29; design: 0 files, 0 summary lines; stage: stats 101-130 start 2026-11-04T23:29:14Z.
- 2026-11-05T00:37:34Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 4 files, 1 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T00:57:27Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 10 files, 4 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T01:17:18Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 18 files, 8 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T01:37:08Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 26 files, 12 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T01:57:01Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 32 files, 15 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T02:16:59Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 36 files, 17 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T02:36:55Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 44 files, 21 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T02:56:53Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 50 files, 24 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T03:16:46Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 58 files, 28 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T03:36:35Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 64 files, 31 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T03:56:27Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 70 files, 34 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T04:16:19Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 78 files, 38 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T04:36:15Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 84 files, 41 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T04:56:13Z stats: andrews=30 aiba=30 monod=30 contois=30; design: 90 files, 44 summary lines; stage: design 101-112 start 2026-11-05T00:29:22Z.
- 2026-11-05T05:16:13Z stats: andrews=31 aiba=30 monod=30 contois=30; design: 96 files, 48 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T05:24:56Z stats: andrews=34 aiba=30 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T05:44:54Z stats: andrews=43 aiba=30 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T06:05:08Z stats: andrews=51 aiba=30 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T06:25:02Z stats: andrews=59 aiba=30 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T06:45:04Z stats: andrews=67 aiba=30 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T07:04:59Z stats: andrews=74 aiba=30 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T07:25:19Z stats: andrews=82 aiba=30 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T07:45:15Z stats: andrews=90 aiba=30 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T08:05:12Z stats: andrews=97 aiba=30 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T08:25:11Z stats: andrews=100 aiba=34 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T08:45:19Z stats: andrews=100 aiba=43 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T09:05:15Z stats: andrews=100 aiba=50 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T09:25:12Z stats: andrews=100 aiba=58 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T09:45:06Z stats: andrews=100 aiba=66 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T10:05:00Z stats: andrews=100 aiba=73 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T10:25:02Z stats: andrews=100 aiba=81 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T10:44:56Z stats: andrews=100 aiba=89 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T11:05:08Z stats: andrews=100 aiba=97 monod=30 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T11:24:58Z stats: andrews=100 aiba=100 monod=35 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T11:44:56Z stats: andrews=100 aiba=100 monod=44 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T12:04:47Z stats: andrews=100 aiba=100 monod=52 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T12:24:41Z stats: andrews=100 aiba=100 monod=59 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T12:44:52Z stats: andrews=100 aiba=100 monod=68 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T13:04:45Z stats: andrews=100 aiba=100 monod=77 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T13:24:50Z stats: andrews=100 aiba=100 monod=85 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T13:44:49Z stats: andrews=100 aiba=100 monod=93 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
- 2026-11-05T14:04:45Z stats: andrews=100 aiba=100 monod=100 contois=30; design: 96 files, 96 summary lines; stage: stats 131-200 start 2026-11-05T05:13:24Z.
