"""data and fit: reproducible synthetic data; the Monod fit recovers the truth on monod seed 101."""
import numpy as np
import pytest

import data as datamod


@pytest.fixture(scope='module')
def monod_truth():
    return datamod.scenario('monod')


def test_make_reproducible(monod_truth):
    tr, theta = monod_truth
    d1, t1 = datamod.make(tr, 101, theta)
    d2, t2 = datamod.make(tr, 101, theta)
    assert t1 == t2
    for e1, e2 in zip(d1['experiments'], d2['experiments']):
        for s in datamod.MEASURED:
            assert e1['y'][s] == e2['y'][s]


def test_make_seed_changes_noise(monod_truth):
    tr, theta = monod_truth
    a, _ = datamod.make(tr, 101, theta)
    b, _ = datamod.make(tr, 102, theta)
    assert any(ea['y']['x1'] != eb['y']['x1'] for ea, eb in zip(a['experiments'], b['experiments']))


def test_make_shape(monod_truth):
    tr, theta = monod_truth
    d, t = datamod.make(tr, 101, theta)
    assert len(d['experiments']) == 3
    for e in d['experiments']:
        for s in datamod.MEASURED:
            assert len(e['y'][s]) == len(e['samples']) == 21
            assert min(e['y'][s]) >= 0
    assert t['true_model'] == 'asprey_2000_monod_kinetics'


@pytest.fixture(scope='module')
def monod_fit_101(monod, monod_truth):
    import fit
    tr, theta = monod_truth
    d, t = datamod.make(tr, 101, theta)
    return fit.fit(monod, d), t


@pytest.mark.slow
def test_monod_fit_recovers_truth(monod_fit_101):
    f, t = monod_fit_101
    assert f['ok'], f.get('error')
    for p, v in t['theta'].items():
        lo, hi = f['ci95'][p]
        assert lo <= v <= hi, (p, v, lo, hi)
    assert np.isfinite(f['aic'])


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason='FINDINGS.md: on monod seed 101 the true Monod law gives chi2 150.5 > limit '
                   '148.8 (p = 0.041), so it does not fit within the noise at 95%. Seed and tolerances are not tuned.')
def test_monod_fit_within_noise(monod_fit_101):
    f, _ = monod_fit_101
    assert f['fits_within_noise'], (f['chi2'], f['chi2_limit'])
