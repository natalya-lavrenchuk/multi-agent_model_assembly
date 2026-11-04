"""fit.py -- Layer 3 statistics: fit a candidate model to the data with parmest, and judge the fit. Code only.

    python fit.py runs/monod_seed1.json asprey_2000_monod_kinetics asprey_2000_contois_kinetics

fit(tr, data)        weighted least squares over all experiments at once (parmest, SSE_weighted, ipopt).
                     The measurement standard deviations are the data's noise formula applied to each reading.
                     Returns, for one candidate:
        theta, ci95       estimates with 95% confidence intervals (from parmest's covariance)
        t_values          estimate / (t_crit * standard error): above 1 = estimated precisely (Asprey &
                          Macchietto 2000); below 1 = the data cannot pin this parameter down
        correlations      pairs above 0.95: the two parameters cannot be told apart from this data
        chi2              weighted residual sum vs the chi-square 95% limit for n - p degrees of freedom:
                          above the limit = the model is systematically off, not just noisy
        aic               chi2 + 2p, for comparing candidates
compare(fits)        Akaike weights: the relative probability of each candidate given the data.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import logging
import math
import os
import sys
import time

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.parmest import parmest
from pyomo.contrib.parmest.experiment import Experiment
from scipy import stats

import build
import candidates
import data as datamod
import translate

CORR_LIMIT = 0.95
# parmest settings, adjustable without editing code (see diag_fit.py for which work on a given computer):
#   FIT_NFE_PER_HOUR   collocation elements per hour in the estimation problem (default 4; 2 halves its size)
#   FIT_IPOPT          extra ipopt options, e.g. "mumps_mem_percent=500" or "linear_solver=ma27,max_iter=5000"
NFE_PER_HOUR = int(os.environ.get('FIT_NFE_PER_HOUR', 4))
# Time limit per candidate, in seconds (env FIT_TIME_LIMIT). A hopeless law can keep the solvers busy for an hour;
# past the limit the SciPy starts stop, parmest gets at most 60 s more, and the fit is reported as not finished.
TIME_LIMIT = float(os.environ.get('FIT_TIME_LIMIT', 300))
# The SciPy search for a starting point (env FIT_PREFIT_RTOL, FIT_PREFIT_DIFF_STEP): integrator tolerance 1e-6
# (1e-8 for the final statistics) and finite-difference step 1e-3 on the log parameters. Tested 3 Nov on Andrews
# seed 1: the looser tolerance alone made the search SLOWER (noisy Jacobian from steps near the integration
# error); with the larger step, the starts took about half the time and chi2 changed by < 0.02 before parmest.
PREFIT_RTOL = float(os.environ.get('FIT_PREFIT_RTOL', 1e-6))
PREFIT_DIFF_STEP = float(os.environ.get('FIT_PREFIT_DIFF_STEP', 1e-3))


SOLVER_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'solver_output_last_fit.log')


@contextlib.contextmanager
def quiet():
    """Send output written below Python (LSODA's Fortran warnings, ipopt's log via Pyomo's logger) to
    runs/solver_output_last_fit.log while fitting, not the screen. Bad parameter values on the way to a fit make
    the solvers complain; that is expected, not news. The file holds the last fit's output only."""
    os.makedirs(os.path.dirname(SOLVER_LOG), exist_ok=True)
    sys.stdout.flush(); sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    null = os.open(SOLVER_LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)    # kept: a crash leaves its reason
    lg = logging.getLogger('pyomo'); level = lg.level
    try:
        os.dup2(null, 1); os.dup2(null, 2)
        lg.setLevel(logging.CRITICAL)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for f in (*saved, null):
            os.close(f)
        lg.setLevel(level)


def ipopt_options() -> dict:
    opts = {'max_iter': 3000}
    for item in filter(None, os.environ.get('FIT_IPOPT', '').split(',')):
        k, v = item.split('=')
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        opts[k.strip()] = v
    return opts


class _Exp(Experiment):
    """One experiment as parmest wants it: the model plus labels for outputs, unknowns and errors."""

    def __init__(self, tr, exp, theta0):
        self.tr, self.exp, self.theta0, self.model = tr, exp, theta0, None

    def get_labeled_model(self):
        if self.model is None:
            m = build.build(self.tr, self.exp, self.theta0, nfe_per_hour=NFE_PER_HOUR)
            build.solve(m)                       # refine the SciPy warm start; a failure here is not fatal
            m.theta.unfix()
            for p in m.P:
                m.theta[p].setlb(1e-6)
            m.del_component(m.obj)
            m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            for s, ys in self.exp['y'].items():
                if s not in m.S:
                    continue
                for t, y, e in zip(self.exp['samples'], ys, _sd(self.exp, s, ys)):
                    m.experiment_outputs[m.x[t, s]] = y
                    m.measurement_error[m.x[t, s]] = float(e)
            m.unknown_parameters.update((m.theta[p], pyo.ComponentUID(m.theta[p])) for p in m.P)
            self.model = m
        return self.model


def prefit(tr: dict, data: dict, theta0: dict) -> dict:
    """Weighted least squares with SciPy on the integrated model (build.integrate): the starting point for
    parmest, and the fallback if parmest fails. Parameters are fitted on a log scale so they stay positive.
    Returns theta, the covariance (J'J)^-1 of the weighted residuals, and whether it converged."""
    from scipy.optimize import least_squares
    names = list(tr['parameters'])

    def z(logp):
        if build.DEADLINE['t'] is not None and time.time() > build.DEADLINE['t']:
            raise build.TimeUp()
        th = dict(zip(names, np.exp(logp)))
        out = []
        for e in data['experiments']:
            sim = build.integrate(tr, e, th, rtol=PREFIT_RTOL, atol=PREFIT_RTOL * 1e-2)
            for s, ys in e['y'].items():
                if s not in tr['structure']['states']:
                    continue
                if sim is None:
                    out += [1e3] * len(ys)
                    continue
                pred = np.array([sim(t)[0][s] for t in e['samples']])
                out += list((np.asarray(ys) - pred) / _sd(e, s, ys))
        return np.nan_to_num(np.asarray(out), nan=1e3, posinf=1e3, neginf=-1e3)
    x0 = np.log([max(float(theta0.get(p, 1.0)), 1e-6) for p in names])
    try:
        import warnings
        with warnings.catch_warnings(), np.errstate(all='ignore'):   # overflow on the way is expected, not news
            warnings.simplefilter('ignore')
            r = least_squares(z, x0, method='trf', x_scale='jac', max_nfev=2000, diff_step=PREFIT_DIFF_STEP)
    except build.TimeUp:
        return {'ok': False, 'error': 'time limit', 'timed_out': True}
    except Exception as e:
        return {'ok': False, 'error': f'{type(e).__name__}: {e}'[:200]}
    th = {k: float(v) for k, v in zip(names, np.exp(r.x))}
    J = r.jac / np.exp(r.x)                                       # d z / d theta (chain rule from log scale)
    try:
        C = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        C = np.linalg.pinv(J.T @ J)
    return {'ok': bool(r.success), 'theta': th, 'cov': C, 'names': names, 'cost': float(2 * r.cost)}


def starts(theta0: dict, names: list) -> list:
    """Four starting points: the given guesses, all 1.0, and the guesses x0.1 and x10. Rate laws have ridges
    where a parameter runs off to infinity and the law becomes a simpler one (Andrews -> Monod as K_i grows);
    which start ends there depends on the start and the platform, so one start is not enough."""
    base = {p: max(float(theta0.get(p, 1.0)), 1e-6) for p in names}
    return [base, {p: 1.0 for p in names}, {p: v * 0.1 for p, v in base.items()}, {p: v * 10 for p, v in base.items()}]


# Weights (env FIT_WEIGHTS): 'model' (default since 4 Nov) or 'measured'. The noise grows with the value
# (data.sd: 5% + 0.01 g/L). Weighting by the MEASURED value, itself noisy, inflates chi-square: a correct model
# was rejected more often than the test's 5% (4 Nov: Monod and Contois seed 3 just over the limit). 'model'
# fits once with measured-value weights, then refits twice with weights from the fitted curve (iteratively
# reweighted least squares); chi-square is computed with the weights of the final curve.
WEIGHTS = os.environ.get('FIT_WEIGHTS', 'model')


def _sd(e: dict, s: str, ys) -> np.ndarray:
    """Noise standard deviation of each reading: from the weights set by reweight(), else from the reading."""
    w = e.get('sd', {}).get(s)
    return np.asarray(w, float) if w is not None else datamod.sd(ys)


def predict(tr: dict, e: dict, theta: dict) -> dict:
    """Model values at the sampling times of experiment e (ipopt, or SciPy if ipopt fails)."""
    m = build.build(tr, e, theta)
    if build.solve(m):
        return build.trajectories(m, e['samples'])
    sim = build.integrate(tr, e, theta)
    return {s: [sim(t)[0][s] for t in e['samples']] for s in m.S} if sim else build.trajectories(m, e['samples'])


def reweight(tr: dict, data: dict, theta: dict) -> dict:
    """A copy of the data whose readings carry noise levels computed from the model curve at theta."""
    out = dict(data)
    out['experiments'] = []
    for e in data['experiments']:
        try:
            pred = predict(tr, e, theta)
        except Exception:
            pred = {}
        e2 = dict(e)
        e2['sd'] = {s: [float(v) for v in datamod.sd(np.clip(np.asarray(pred[s], float), 0, None))]
                    for s in e['y'] if s in pred and np.all(np.isfinite(pred[s]))}
        out['experiments'].append(e2)
    return out


WORKERS = int(os.environ.get('FIT_WORKERS', min(4, os.cpu_count() or 1)))   # starts fitted at the same time
# Once at least two starts are back (and none is waiting), the others get at most FACTOR x the slowest finished
# start's time (and at least MIN_S seconds) before they are dropped. One start stuck in a stiff region otherwise
# held a good fit for the full time limit (Aiba on Andrews data, 3 Nov: 3 starts in ~20 s, the fourth > 4 min).
STRAGGLER_FACTOR, STRAGGLER_MIN_S = 3.0, 30.0


def _start_worker(conn, tr, data, st):
    try:
        conn.send(prefit(tr, data, st))
    except Exception as e:
        conn.send({'ok': False, 'error': f'{type(e).__name__}: {e}'[:200]})
    conn.close()


def _run_starts(tr: dict, data: dict, sts: list) -> list:
    """prefit from each start, up to WORKERS at a time in forked processes. A start still running at the time
    limit is killed and counted as timed out. Results come back in the order of the starts."""
    import multiprocessing as mp
    from multiprocessing.connection import wait
    if WORKERS <= 1:
        out = []
        for st in sts:
            if build.DEADLINE['t'] is not None and time.time() > build.DEADLINE['t']:
                out.append({'ok': False, 'error': 'time limit', 'timed_out': True})
            else:
                out.append(prefit(tr, data, st))
        return out
    ctx = mp.get_context('fork')
    out, todo, running = [None] * len(sts), list(range(len(sts))), {}
    durations, started = [None] * len(sts), [None] * len(sts)
    while todo or running:
        while todo and len(running) < WORKERS:
            i = todo.pop(0)
            recv, send = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_start_worker, args=(send, tr, data, sts[i]))
            proc.start()
            send.close()
            running[recv] = (i, proc)
            started[i] = time.time()
        left = None if build.DEADLINE['t'] is None else max(0.0, build.DEADLINE['t'] - time.time())
        done_times = [d for d in durations if d is not None]
        patience = None
        if len(done_times) >= 2 and not todo:                    # most starts are back: do not wait long for the rest
            allowed = max(STRAGGLER_MIN_S, STRAGGLER_FACTOR * max(done_times))
            patience = max(0.0, min(allowed - (time.time() - started[i]) for i, _ in running.values()))
            left = patience if left is None else min(left, patience)
        ready = wait(list(running), timeout=left)
        if not ready:                                            # time limit, or stragglers: stop what still runs
            why = ('abandoned: much slower than the other starts' if patience is not None and
                   (build.DEADLINE['t'] is None or time.time() < build.DEADLINE['t']) else 'time limit')
            if why == 'time limit':
                for recv, (i, proc) in running.items():
                    proc.kill()
                    proc.join()
                    out[i] = {'ok': False, 'error': why, 'timed_out': True}
                for i in todo:
                    out[i] = {'ok': False, 'error': why, 'timed_out': True}
                break
            allowed = max(STRAGGLER_MIN_S, STRAGGLER_FACTOR * max(done_times))
            for recv in [r for r, (i, _) in running.items() if time.time() - started[i] >= allowed - 1e-3]:
                i, proc = running.pop(recv)
                proc.kill()
                proc.join()
                out[i] = {'ok': False, 'error': why, 'abandoned': True}
            continue
        for recv in ready:
            i, proc = running.pop(recv)
            durations[i] = time.time() - started[i]
            try:
                out[i] = recv.recv()
            except EOFError:
                out[i] = {'ok': False, 'error': f'start crashed (exit code {proc.exitcode})'}
            proc.join()
    return out


def prefit_multi(tr: dict, data: dict, theta0: dict) -> dict:
    """prefit from every start; keep the best. 'agree' counts the starts that reached the best fit (within 1%)."""
    runs = _run_starts(tr, data, starts(theta0, list(tr['parameters'])))
    ok = [r for r in runs if r.get('ok') or 'cost' in r]
    if not ok:
        return {**runs[0], 'timed_out': any(r.get('timed_out') for r in runs)}
    best = min(ok, key=lambda r: r['cost'])
    best = dict(best)
    best['agree'] = sum(1 for r in ok if r['cost'] <= best['cost'] * 1.01)
    best['n_starts'] = len(runs)
    best['timed_out'] = any(r.get('timed_out') for r in runs)
    best['abandoned'] = sum(1 for r in runs if r.get('abandoned'))
    return best


PARMEST_BROKEN = {'reason': ''}       # set after parmest fails once: this computer's ipopt cannot run it


def runaway(theta: dict, theta0: dict) -> list:
    """Parameters that ran off (above 1e6, or a million times their start): the law has turned into a simpler one."""
    out = []
    for p, v in theta.items():
        if not math.isfinite(v) or v > 1e6 or v > 1e6 * max(float(theta0.get(p, 1.0)), 1e-6):
            out.append(f'{p} ran to {v:.3g}: the law reduced to a simpler form; this fit may be stuck, not the law wrong')
    return out


FIT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'fits')
FIT_CACHE_VERSION = 2      # bump when the fitting code changes in a way that changes results


def _fit_key(tr: dict, data: dict, theta0: dict | None) -> str:
    """Everything that decides a fit's result: the model's equations, parameters, inputs, the start, the data,
    and the settings. Same key, same result, so a rerun reuses it."""
    laws = sorted((L['target'], L['kind'], str(L['rhs'])) for L in tr['laws'] if 'rhs' in L)
    blob = json.dumps({'v': FIT_CACHE_VERSION, 'laws': laws, 'params': sorted(tr['parameters']),
                       'inputs': sorted(tr.get('inputs', [])), 'theta0': theta0, 'data': data['experiments'],
                       'nfe': NFE_PER_HOUR, 'ipopt': ipopt_options(), 'limit': TIME_LIMIT, 'weights': WEIGHTS}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def fit(tr: dict, data: dict, theta0: dict | None = None) -> dict:
    """_fit with the time limit and the solvers' chatter silenced, cached on disk (cache/fits/), so rerunning
    the agents does not refit a model it has already fitted. Records the seconds taken."""
    key = _fit_key(tr, data, theta0)
    path = os.path.join(FIT_CACHE, key + '.json')
    if os.path.exists(path):
        out = json.load(open(path))
        out['sup'], out['cached'] = tr['sup'], True
        return out
    t0 = time.time()
    if os.environ.get('FIT_ISOLATE', '1') == '0':     # fallback if forking fails on a computer: no hard time limit
        build.DEADLINE['t'] = time.time() + TIME_LIMIT
        try:
            with quiet():
                out = _fit(tr, data, theta0)
        finally:
            build.DEADLINE['t'] = None
    else:
        out = _fit_in_child(tr, data, theta0)
    out['seconds'] = round(time.time() - t0, 1)
    if 'crashed' in out.get('error', ''):
        return out                               # a crash may not repeat: do not keep it
    try:
        os.makedirs(FIT_CACHE, exist_ok=True)
        json.dump(out, open(path + '.tmp', 'w'), default=float)
        os.replace(path + '.tmp', path)
    except (TypeError, ValueError, OSError):
        pass                                     # caching is a convenience; a fit that cannot be saved still counts
    return out


def _child(conn, tr, data, theta0):
    os.setpgrp()                        # its own process group, so a kill also stops the ipopt it started
    build.DEADLINE['t'] = time.time() + TIME_LIMIT
    try:
        with quiet():
            out = _fit(tr, data, theta0)
    except Exception as e:
        out = {'sup': tr['sup'], 'ok': False, 'error': f'fit raised {type(e).__name__}: {e}'[:250]}
    conn.send(out)
    conn.close()


def _fit_in_child(tr: dict, data: dict, theta0: dict | None) -> dict:
    """Run the fit in a child process with a hard time limit. A stuck solver (a stiff blow-up inside LSODA's
    compiled code never returns to Python) or a crashed one (Python aborted on the Mac: "zsh: abort", 2 Nov)
    then costs this candidate only, not the run. Inside the child, the limit also stops new SciPy starts and
    caps ipopt; the kill at TIME_LIMIT + 60 s is the backstop."""
    import multiprocessing as mp
    import signal
    ctx = mp.get_context('fork')
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(send, tr, data, theta0))   # not a daemon: it starts its own workers
    proc.start()
    send.close()
    out = None
    try:
        if recv.poll(TIME_LIMIT + 60):
            out = recv.recv()
    except EOFError:
        pass                                     # the child died before sending
    if proc.is_alive() or out is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    proc.join(5)
    if out is None:
        crashed = proc.exitcode not in (None, 0, -signal.SIGKILL)
        out = {'sup': tr['sup'], 'ok': False, 'timed_out': not crashed,
               'error': (f'the solver crashed on this law (exit code {proc.exitcode}; see runs/solver_output_last_fit.log)'
                         if crashed else f'fit did not finish within the time limit ({TIME_LIMIT:.0f} s): the solver '
                         f'was stuck on this law') + '; not evidence for or against the law'}
    return out


def _fit(tr: dict, data: dict, theta0: dict | None = None) -> dict:
    """SciPy pre-fit, then parmest from that point (refines and gives the covariance). If parmest fails on this
    computer's ipopt, the SciPy result is used; 'method' in the result says which."""
    theta0 = theta0 or (candidates.start_values(tr) if 'law_iri' in tr else build.paper_values(tr))
    n = sum(len(ys) for e in data['experiments'] for s, ys in e['y'].items() if s in tr['structure']['states'])
    p = len(tr['parameters'])
    deadline = build.DEADLINE['t']
    pre = prefit_multi(tr, data, theta0)
    if 'theta' not in pre and pre.get('timed_out'):
        return {'sup': tr['sup'], 'ok': False, 'timed_out': True,
                'error': f'fit did not finish within the time limit ({TIME_LIMIT:.0f} s): the solver was stuck on this '
                         f'law (no starting point finished); not evidence for or against the law'}
    data0 = data
    if WEIGHTS == 'model' and 'theta' in pre:                    # two reweighting passes from the multi-start fit
        for _ in range(2):
            data = reweight(tr, data0, pre['theta'])
            again = prefit(tr, data, pre['theta'])
            if 'theta' in again:
                pre = {**pre, **{k: again[k] for k in ('theta', 'cov', 'names', 'cost')}}
        data = reweight(tr, data0, pre['theta'])
    start = pre['theta'] if 'theta' in pre else theta0
    build.find_ipopt()                                           # parmest calls ipopt by name
    method, note = 'parmest', ''
    try:
        if PARMEST_BROKEN['reason']:
            raise RuntimeError(PARMEST_BROKEN['reason'])
        exps = [_Exp(tr, e, start) for e in data['experiments']]
        left = max(60.0, build.DEADLINE['t'] - time.time()) if build.DEADLINE['t'] else None
        opts = {**ipopt_options(), **({'max_cpu_time': left} if left else {})}
        est = parmest.Estimator(exps, obj_function='SSE_weighted', solver_options=opts)
        _, theta = est.theta_est()
        cov = est.cov_est()
        keys = list(theta.keys())                               # a dict or a pandas Series, by Pyomo version
        names = [str(k).split('[')[-1].strip("]'") for k in keys]
        th = {nm: float(theta[k]) for nm, k in zip(names, keys)}
        C = np.asarray(cov.loc[keys, keys] if hasattr(cov, 'loc') else cov, float)
        build.DEADLINE['t'] = None                               # the limit is for searching, not for evaluating
        if 'theta' in pre and chi_square(tr, data, th) > 1.01 * chi_square(tr, data, pre['theta']):
            raise RuntimeError('parmest ended worse than its starting point')
    except Exception as e:                                       # ipopt failed here: use the SciPy fit
        if 'theta' not in pre:
            return {'sup': tr['sup'], 'ok': False, 'error': f'parmest: {type(e).__name__}: {e}'[:250] +
                    f"; scipy: {pre.get('error', 'did not converge')}"}
        timed_out = deadline is not None and time.time() > deadline
        if not PARMEST_BROKEN['reason'] and 'worse than' not in str(e) and not timed_out:
            PARMEST_BROKEN['reason'] = f'parmest failed earlier in this run ({type(e).__name__})'
        method, note = 'scipy', f'parmest not used ({str(e)[:60]}); SciPy least-squares result used'
        names, th, C = pre['names'], pre['theta'], pre['cov']
    build.DEADLINE['t'] = None
    se = np.sqrt(np.clip(np.diag(C), 0, None))
    dof = max(n - p, 1)
    t_crit = stats.t.ppf(0.975, dof)
    res = residuals(tr, reweight(tr, data0, th) if WEIGHTS == 'model' else data0, th)
    chi2 = sum(v['chi2'] for v in res.values())                  # own count: parmest scales its objective
    chi2_limit = float(stats.chi2.ppf(0.95, dof))
    p_value = float(stats.chi2.sf(chi2, dof))
    corr = C / np.outer(se, se) if np.all(se > 0) else np.full_like(C, np.nan)
    pairs = [(names[i], names[j], round(float(corr[i, j]), 3)) for i in range(p) for j in range(i + 1, p)
             if abs(corr[i, j]) > CORR_LIMIT]
    warnings = runaway(th, theta0)
    if pre.get('n_starts') and pre.get('agree', 0) < pre['n_starts']:
        if pre.get('abandoned') and not pre.get('timed_out'):
            warnings.append(f"only {pre['agree']} of {pre['n_starts']} starting points reached this fit; "
                            f"{pre['abandoned']} dropped for being much slower than the others")
        elif pre.get('timed_out'):
            warnings.append(f"time limit ({TIME_LIMIT:.0f} s) reached: only some of the {pre['n_starts']} starting "
                            f"points finished, so a better fit may exist")
        else:
            warnings.append(f"only {pre['agree']} of {pre['n_starts']} starting points reached this fit: other starts "
                            f"stopped at worse local optima")
    return {'sup': tr['sup'], 'ok': True, 'n': n, 'p': p, 'method': method, 'note': note, 'warnings': warnings,
            'theta': th,
            'ci95': {nm: [th[nm] - t_crit * s, th[nm] + t_crit * s] for nm, s in zip(names, se)},
            't_values': {nm: (th[nm] / (t_crit * s) if s > 0 else math.inf) for nm, s in zip(names, se)},
            'correlated_pairs': pairs,
            'chi2': chi2, 'chi2_limit': chi2_limit, 'fits_within_noise': chi2 <= chi2_limit,
            'p_value': p_value, 'chi2_limit_99': float(stats.chi2.ppf(0.99, dof)),
            'fit_class': 'within the noise' if p_value >= 0.05 else 'marginal' if p_value >= 0.01 else 'lack of fit',
            'weights': 'from the fitted curve (2 reweighting passes)' if WEIGHTS == 'model' else 'from the measured values',
            'aic': chi2 + 2 * p, 'residuals': res}


def residuals(tr: dict, data: dict, theta: dict) -> dict:
    """Per experiment and output: chi-square contribution, mean residual (over- or under-prediction) and the
    time of the worst miss. Tells the statistical agent where a model fails, not only that it fails.
    Weights: as carried by the data (reweight), else from the measured values."""
    out = {}
    for e in data['experiments']:
        pred = predict(tr, e, theta)
        for s, ys in e['y'].items():
            if s in pred:
                z = (np.asarray(ys) - np.asarray(pred[s])) / _sd(e, s, ys)
                out[f"{e.get('name', '?')}:{s}"] = {'chi2': float(np.sum(z ** 2)), 'n': len(z),
                                                   'mean_z': float(np.mean(z)),
                                                   'worst_t': float(e['samples'][int(np.argmax(np.abs(z)))]),
                                                   'worst_z': float(z[int(np.argmax(np.abs(z)))])}
    return out


def chi_square(tr: dict, data: dict, theta: dict) -> float:
    """sum over all readings of ((measured - predicted) / sd)^2; sd from the model curve at theta
    (WEIGHTS 'model') or from the measured value ('measured')."""
    d = reweight(tr, data, theta) if WEIGHTS == 'model' else data
    return sum(v['chi2'] for v in residuals(tr, d, theta).values())


def compare(fits: list) -> list:
    ok = [f for f in fits if f.get('ok')]
    if not ok:
        return fits
    best = min(f['aic'] for f in ok)
    w = {f['sup']: math.exp(-(f['aic'] - best) / 2) for f in ok}
    tot = sum(w.values())
    for f in ok:
        f['akaike_weight'] = w[f['sup']] / tot
    return fits


def report(f: dict) -> str:
    if not f.get('ok'):
        return f"{f['sup']}: fit FAILED ({f.get('error')})"
    cls = f.get('fit_class') or ('within the noise' if f['fits_within_noise'] else 'lack of fit')
    label = {'within the noise': 'fits within the noise', 'marginal': 'MARGINAL: just outside the 95% limit, '
             'inside the 99% limit', 'lack of fit': 'LACK OF FIT'}[cls]
    lines = [f"{f['sup']}: chi2 {f['chi2']:.1f} vs limit {f['chi2_limit']:.1f}"
             + (f" (p = {f['p_value']:.3f})" if 'p_value' in f else '') + f" ({label}); AIC {f['aic']:.1f}"
             + (f"; probability {f['akaike_weight']:.3f}" if 'akaike_weight' in f else '')
             + (f"  [{f['note']}]" if f.get('note') else '')]
    for nm in f['theta']:
        lo, hi = f['ci95'][nm]
        tv = f['t_values'][nm]
        lines.append(f"   {nm:8} {f['theta'][nm]:.4g}  95% CI [{lo:.4g}, {hi:.4g}]  t {tv:.2f}"
                     + ('' if tv > 1 else '  <- not estimated precisely'))
    for a, b, c in f['correlated_pairs']:
        lines.append(f'   ! {a} and {b} correlated ({c}): this data separates them only poorly (a caveat; their '
                     f't-values above say whether each is still estimated precisely)')
    for w in f.get('warnings', []):
        lines.append(f'   ! {w}')
    if not f['fits_within_noise']:
        worst = sorted(f['residuals'].items(), key=lambda kv: -kv[1]['chi2'])[:3]
        lines.append('   misfit mostly in: ' + '; '.join(
            f"{k} (chi2 {v['chi2']:.0f} over {v['n']} readings, model {'under' if v['mean_z'] > 0 else 'over'}-predicts "
            f"on average, worst at t={v['worst_t']:.0f} h)" for k, v in worst))
    return '\n'.join(lines)


if __name__ == '__main__':
    d = json.load(open(sys.argv[1]))
    fits = compare([fit(translate.translate(s), d) for s in sys.argv[2:]])
    for f in fits:
        print(report(f))
