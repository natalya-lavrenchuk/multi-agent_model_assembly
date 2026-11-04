"""Shared setup for the agents' tests. No LLM, no network: every translation must come from cache/llm_cache/.
A cache miss makes llm.ask raise, so a test can never send a request."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = os.path.dirname(HERE)
sys.path.insert(0, AGENTS)
sys.path.insert(0, os.path.join(AGENTS, '..', 'extractor_agent'))
os.environ.setdefault('IPOPT', '/opt/ipopt/bin/ipopt')

from extractor import llm  # noqa: E402

KG = 'http://example.org/kg/'
ANDREWS = KG + 'zentou_2019_andrews/law/zentou_2019_eq5'
AIBA = KG + 'muloiwa_2020_free_laws/law/muloiwa_2020_eq11'
MAPPING = {'mu': 'r', 'S_': 'x2'}


def _no_llm(*a, **k):
    raise RuntimeError('LLM call attempted in a test (cache miss): tests must run from cache/llm_cache only')


llm.ask = _no_llm


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: fits a model (minutes)')


@pytest.fixture(scope='session')
def monod():
    import translate
    return translate.translate('asprey_2000_monod_kinetics')


@pytest.fixture(scope='session')
def contois():
    import translate
    return translate.translate('asprey_2000_contois_kinetics')


@pytest.fixture(scope='session')
def law_tr():
    import kg_query
    import translate
    return {'andrews': translate.translate_model(kg_query.get_law(ANDREWS)),
            'aiba': translate.translate_model(kg_query.get_law(AIBA))}


@pytest.fixture(scope='session')
def four(monod, contois, law_tr):
    import candidates
    return {'monod': monod, 'contois': contois,
            'andrews': candidates.swap(monod, ANDREWS, MAPPING, law_tr['andrews']),
            'aiba': candidates.swap(monod, AIBA, MAPPING, law_tr['aiba'])}
