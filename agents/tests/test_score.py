"""score: the truth tables give 'same law' and 'same mechanism' where expected."""
import json

import pytest

import score

KG = score.KG
P = {
    'monod': 'asprey_2000_monod_kinetics/law/asprey_2000_eq17',
    'monod_zentou': 'zentou_2019_monod/law/zentou_2019_eq3',
    'contois': 'asprey_2000_contois_kinetics/law/asprey_2000_eq18',
    'contois_muloiwa': 'muloiwa_2020_free_laws/law/muloiwa_2020_eq9',
    'andrews': 'zentou_2019_andrews/law/zentou_2019_eq5',
    'andrews_muloiwa': 'muloiwa_2020_free_laws/law/muloiwa_2020_eq6',
    'aiba': 'muloiwa_2020_free_laws/law/muloiwa_2020_eq11',
    'teissier': 'muloiwa_2020_free_laws/law/muloiwa_2020_eq7',
}


def iri(k):
    return KG + P[k]


@pytest.mark.parametrize('a,b', [('monod', 'monod_zentou'), ('contois', 'contois_muloiwa'),
                                 ('andrews', 'andrews_muloiwa')])
def test_same_form(a, b):
    assert score.form_of(iri(a)) == score.form_of(iri(b))


@pytest.mark.parametrize('a,b,same', [('andrews', 'aiba', True), ('monod', 'teissier', True),
                                      ('monod', 'contois', False), ('monod', 'andrews', False),
                                      ('contois', 'aiba', False)])
def test_mechanism(a, b, same):
    assert score.form_of(iri(a)) != score.form_of(iri(b))
    ma, mb = (score.mechanism_of(score.form_of(iri(x))) for x in (a, b))
    assert (ma == mb) == same


def test_every_form_has_a_mechanism():
    for f in score.FORMS:
        assert f in score.MECHANISM
    assert score.form_of(None) == 'not in the graph'
    assert score.mechanism_of(score.form_of(KG + 'x/law/unknown')) == 'other'


def _log(tmp_path, truth_iri, final_iri, scenario='andrews', status='accept'):
    truth = {'seed': 1, 'scenario': scenario, 'law_iri': truth_iri, 'theta': {'a': 1.0}}
    json.dump(truth, open(tmp_path / f'{scenario}_seed1.truth.json', 'w'))
    log = {'data_file': f'{scenario}_seed1.json',
           'candidates': [{'rate_id': 1, 'iri': final_iri, 'built': True,
                           'fit': {'ok': True, 'aic': 10.0, 'chi2': 5.0, 'chi2_limit': 9.0, 'theta': {'a': 1.0}}}],
           'final': {'model': 'C1 (x)', 'status': status}}
    p = tmp_path / f'agents_{scenario}_seed1.json'
    json.dump(log, open(p, 'w'))
    return str(p)


@pytest.mark.parametrize('final,level', [('andrews', 'same law'), ('andrews_muloiwa', 'same law'),
                                         ('aiba', 'same mechanism'), ('monod', 'wrong')])
def test_score_levels(tmp_path, final, level):
    s = score.score(_log(tmp_path, iri('andrews'), iri(final)))
    assert s['level'] == level


def test_score_not_in_graph(tmp_path):
    s = score.score(_log(tmp_path, None, iri('andrews'), scenario='notingraph'))
    assert s['level'] == 'wrong (forced a law)'
    s = score.score(_log(tmp_path, None, iri('andrews'), scenario='notingraph', status='none'))
    assert s['level'] == 'right (nothing accepted)'
