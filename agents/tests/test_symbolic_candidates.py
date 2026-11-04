"""symbolic and candidates: units, positivity, symbol resolution, and a wrong unit failing."""
import pytest
import sympy as sp

import candidates
import symbolic


@pytest.mark.parametrize('name', ['monod', 'contois', 'andrews', 'aiba'])
def test_units_balance(four, name):
    c = four[name]
    assert c['ok'], c.get('problems')
    for L in c['laws']:
        assert L['problems'] == [], (name, L['eq'], L['problems'])


@pytest.mark.parametrize('name', ['monod', 'contois', 'andrews', 'aiba'])
def test_structure(four, name):
    s = four[name]['structure']
    assert s['ok'] and s['states'] == ['x1', 'x2'] and s['algebraic'] == ['r']
    assert s['equations'] == s['unknowns'] == 3


@pytest.mark.parametrize('name', ['monod', 'contois', 'andrews', 'aiba'])
def test_check_positivity(four, name):
    c = four[name]
    res = symbolic.check_positivity(c['laws'], set(c['inputs']), set(c['parameters']))
    assert res['ok'], res['problems']


def test_swapped_parameters(four):
    assert four['andrews']['parameters'] == ['KS', 'Ki', 'mumax', 'theta3', 'theta4']
    assert four['aiba']['parameters'] == ['KI', 'Ks', 'mumax', 'theta3', 'theta4']


@pytest.mark.parametrize('sym', ['\\upmu', '\\mu', 'mu', 'μ', 'µ'])
def test_resolve_symbol_mu(law_tr, sym):
    assert candidates.resolve_symbol(law_tr['andrews'], sym) == 'mu'
    assert candidates.resolve_symbol(law_tr['aiba'], sym) == 'mu'


def test_resolve_symbol_other(law_tr):
    t = law_tr['andrews']
    assert candidates.resolve_symbol(t, 'S') == 'S_'
    assert candidates.resolve_symbol(t, 'K_{S}') == 'KS'
    assert candidates.resolve_symbol(t, 'K_S') == 'KS'
    assert candidates.resolve_symbol(t, '\\mu_{max}') == 'mumax'
    assert candidates.resolve_symbol(t, 'nonsense') == 'nonsense'


def test_wrong_unit_fails():
    r, th1, th2, x2 = sp.symbols('r theta1 theta2 x2')
    good = {'r': '1 / h', 'theta1': '1 / h', 'theta2': 'g / l', 'x2': 'g / l'}
    assert symbolic.check_law('r', 'algebraic', th1 * x2 / (th2 + x2), good)['ok']
    bad = {**good, 'theta2': 'h'}                     # K_S in hours: cannot be added to a concentration
    res = symbolic.check_law('r', 'algebraic', th1 * x2 / (th2 + x2), bad)
    assert not res['ok'] and any('added' in p for p in res['problems'])
    res = symbolic.check_law('r', 'algebraic', th1 * x2, good)     # 1/h * g/l is not 1/h
    assert not res['ok']


def test_unit_inferred_when_missing():
    r, th1, th2, x2 = sp.symbols('r theta1 theta2 x2')
    res = symbolic.check_law('r', 'algebraic', th1 * x2 / (th2 + x2),
                             {'r': '1 / h', 'theta1': '1 / h', 'theta2': None, 'x2': 'g / l'})
    assert res['ok'] and res['inferred'] == {'theta2': 'mass / length^3'}


def test_wrong_mapping_fails(monod, law_tr):
    c = candidates.swap(monod, 'http://example.org/kg/zentou_2019_andrews/law/zentou_2019_eq5',
                        {'mu': 'r', 'S_': 'x9'}, law_tr['andrews'])
    assert not c['ok'] and any('x9' in p for p in c['problems'])
