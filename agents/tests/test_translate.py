"""translate: the cached laws translate and give the expected SymPy expressions."""
import sympy as sp

S = sp.Symbol


def same(a, b):
    return sp.simplify(a - b) == 0


def law(tr, target):
    return next(L for L in tr['laws'] if L['target'] == target)


def test_monod_asprey_2000(monod):
    assert monod['ok']
    th1, th2, th3, th4, x1, x2, u1, u2, r = (S(n) for n in 'theta1 theta2 theta3 theta4 x1 x2 u1 u2 r'.split())
    assert same(law(monod, 'r')['rhs'], th1 * x2 / (th2 + x2))
    assert same(law(monod, 'x1')['rhs'], (r - u1 - th4) * x1)
    assert same(law(monod, 'x2')['rhs'], -r * x1 / th3 + u1 * (u2 - x2))
    assert all(not L['problems'] for L in monod['laws'])


def test_contois_asprey_2000(contois):
    assert contois['ok']
    th1, th2, x1, x2 = S('theta1'), S('theta2'), S('x1'), S('x2')
    assert same(law(contois, 'r')['rhs'], th1 * x2 / (th2 * x1 + x2))


def test_andrews_zentou_eq5(law_tr):
    t = law_tr['andrews']
    assert t['ok']
    L = t['laws'][0]
    mumax, KS, Ki, s = S('mumax'), S('KS'), S('Ki'), S('S_')
    assert L['kind'] == 'algebraic' and L['target'] == 'mu'
    assert same(L['rhs'], mumax * s / (KS + s + s ** 2 / Ki))
    assert sorted(t['parameters']) == ['KS', 'Ki', 'mumax']


def test_aiba_muloiwa_eq11(law_tr):
    t = law_tr['aiba']
    assert t['ok']
    L = t['laws'][0]
    mumax, Ks, KI, s = S('mumax'), S('Ks'), S('KI'), S('S_')
    assert L['kind'] == 'algebraic' and L['target'] == 'mu'
    assert same(L['rhs'], mumax * s / (Ks + s) * sp.exp(-s / KI))


def test_no_llm_calls_made(monod, contois, law_tr):
    from extractor import llm
    assert llm.USAGE['calls'] == 0
