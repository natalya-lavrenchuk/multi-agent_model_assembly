"""diag_fit.py -- can parmest run on this computer, and with which settings? Code only, no LLM.

    python diag_fit.py runs/andrews_seed1.json

Fits Asprey's Monod model to the data with parmest, starting (as fit.py does) from the SciPy fit, under a few
settings, and prints which work. ipopt's own output for each attempt goes to runs/diag_<setting>.log.
Why: on a Mac whose ipopt has no HSL library (no MA27/MA57), ipopt uses MUMPS, which can run out of working
memory on the estimation problem. Two remedies are tested: more memory for MUMPS, and a smaller problem
(a coarser time grid only for estimation).

Then set the one that works before running the agents, e.g.:
    export FIT_IPOPT="mumps_mem_percent=500"
    export FIT_NFE_PER_HOUR=2
"""
import json
import os
import sys
import time

from pyomo.contrib.parmest import parmest

import build
import fit
import translate

SETTINGS = [                                     # (name, elements per hour, extra ipopt options)
    ('default', 4, {}),
    ('mumps_mem_500', 4, {'mumps_mem_percent': 500}),
    ('mumps_mem_2000', 4, {'mumps_mem_percent': 2000}),
    ('grid2', 2, {}),
    ('grid2_mumps_mem_500', 2, {'mumps_mem_percent': 500}),
    ('ma27', 4, {'linear_solver': 'ma27'}),
]

if __name__ == '__main__':
    data = json.load(open(sys.argv[1]))
    tr = translate.translate('asprey_2000_monod_kinetics')
    print('ipopt:', build.find_ipopt())
    pre = fit.prefit_multi(tr, data, build.paper_values(tr))
    print('SciPy fit (reference):', {k: round(v, 4) for k, v in pre['theta'].items()},
          ' chi2', round(fit.chi_square(tr, data, pre['theta']), 1))
    works = []
    for name, nfe, opts in SETTINGS:
        fit.NFE_PER_HOUR = nfe
        exps = [fit._Exp(tr, e, pre['theta']) for e in data['experiments']]
        log = os.path.join('runs', f'diag_{name}.log')
        t0 = time.time()
        try:
            est = parmest.Estimator(exps, obj_function='SSE_weighted',
                                    solver_options={'max_iter': 3000, **opts, 'output_file': log})
            _, theta = est.theta_est()
            est.cov_est()
            th = {str(k).split('[')[-1].strip("]'"): float(v) for k, v in dict(theta).items()}
            same = all(abs(th[p] - pre['theta'][p]) <= 0.01 * abs(pre['theta'][p]) for p in th)
            print(f'{name:22} OK      {time.time() - t0:6.1f} s  '
                  f"{'matches the SciPy fit' if same else 'DIFFERENT from the SciPy fit'}  {({k: round(v, 4) for k, v in th.items()})}")
            if same:
                works.append((name, nfe, opts))
        except Exception as e:
            print(f'{name:22} FAILED  {time.time() - t0:6.1f} s   {type(e).__name__}: {str(e)[:90]}')
        if os.path.exists(log):
            tail = [l for l in open(log).read().splitlines()
                    if l.startswith('EXIT') or 'rror' in l or 'memory' in l.lower() or 'Unknown' in l]
            print('                       ipopt says:', ' | '.join(tail[-2:]) or '(no EXIT line)')
    if works:
        name, nfe, opts = works[0]
        print(f'\nUse this before running the agents ({name}):')
        if opts:
            print('    export FIT_IPOPT="' + ','.join(f'{k}={v}' for k, v in opts.items()) + '"')
        if nfe != 4:
            print(f'    export FIT_NFE_PER_HOUR={nfe}')
    else:
        print('\nNo setting worked: a different ipopt build is needed (conda-forge or Homebrew).')
