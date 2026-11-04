"""sympy_parser_eval.py -- how well does SymPy's OWN LaTeX parser read a paper's equations?

Input:  the equations as LaTeX (the reader's output, and the printed version from gold_laws.json)
Parser: sympy.parsing.latex.parse_latex, both backends ('antlr', 'lark'), as shipped.
        The only preprocessing is removing the $$ delimiters and trailing punctuation.
Truth:  gold_laws.json 'sympy_reference' (the printed equation as a hand-checked SymPy expression)

Comparison is fair to the parser: symbol NAMES are normalised on both sides before comparing
(SymPy writes F_out as 'F_{o*(t*u)}', which is still one consistent symbol, so that alone is not
counted as an error). What counts as wrong is a different MATHEMATICAL expression:
    fail          the parser raised an error
    wrong         parsed, but the expression differs from the reference
    correct       same expression as the reference after name normalisation
For 'wrong' the report says how: symbols missing / extra (e.g. [GLC] read as G*C*L), the time
derivative lost, terms dropped.

    python studies/sympy_parser_eval.py papers/kontoravdi_2010
"""
from __future__ import annotations
import json, re, sys, warnings
import sympy as sp
from sympy.parsing.latex import parse_latex

GREEK = {'varepsilon': 'epsilon', 'lambda': 'lam', 'nu': 'v'}


def norm_name(name: str) -> str:
    """Comparable name: 'F_{o*(t*u)}' and 'F_out' -> 'F_otu'; 'X_{nu}' and 'X_v' -> 'X_v'."""
    base, _, sub = name.partition('_')
    base = GREEK.get(base.lstrip('\\'), base.lstrip('\\'))
    sub = re.sub(r'[^A-Za-z0-9]', '', sub)
    sub = GREEK.get(sub, sub)
    # SymPy reads a subscript as a product and reorders it ('out' -> 'o*(t*u)'), but it stays one
    # consistent symbol. Compare subscript letters as a sorted multiset so that alone is not an error.
    sub = ''.join(sorted(sub))
    return base + ('_' + sub if sub else '')


def normalise(expr, reference=False):
    """The reference names a concentration [X] as C_X; SymPy, reading [ ] as brackets, names a
    single-symbol [H] just H. That is a naming difference, not an error, so the C_ prefix is
    dropped on the reference side. A multi-letter [GLC] still differs: SymPy makes it G*L*C."""
    def n(s):
        name = str(s)
        if reference and name.startswith('C_'):
            name = name[2:]
        return sp.Symbol(norm_name(name))
    return expr.xreplace({s: n(s) for s in expr.free_symbols})


def as_zero(eq):
    """lhs - rhs with every derivative frozen into a named symbol. Without this, simplify()
    evaluates d(H)/dt of a plain symbol to 0, and any two derivatives would compare equal."""
    z = eq.lhs - eq.rhs
    return z.replace(lambda e: isinstance(e, sp.Derivative),
                     lambda e: sp.Symbol('D[' + sp.sstr(e.expr) + ']'))


def clean(tex: str) -> str:
    return tex.strip().strip('$').strip().rstrip('.,').strip()


def judge(tex, ref, backend):
    out = {'backend': backend}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got = parse_latex(clean(tex), backend=backend)
    except Exception as ex:
        out.update(verdict='fail', detail=f'{type(ex).__name__}: {str(ex).splitlines()[0][:90]}')
        return out
    out['parsed'] = sp.sstr(got)
    if not isinstance(got, sp.Equality):
        out.update(verdict='wrong', detail='not an equation')
        return out
    g, r = normalise(got), normalise(ref, reference=True)
    same = False
    try:
        same = sp.simplify(as_zero(g) - as_zero(r)) == 0
    except Exception:
        pass
    if same:
        out['verdict'] = 'correct'
        return out
    why = []
    gs, rs = {str(s) for s in g.free_symbols}, {str(s) for s in r.free_symbols}
    if rs - gs:
        why.append('missing ' + ', '.join(sorted(rs - gs)))
    if gs - rs:
        why.append('extra ' + ', '.join(sorted(gs - rs)))
    if r.atoms(sp.Derivative) and not g.atoms(sp.Derivative):
        why.append('time derivative lost')
    nr, ng = len(sp.Add.make_args(sp.expand(r.rhs))), len(sp.Add.make_args(sp.expand(g.rhs)))
    if ng < nr:
        why.append(f'{nr - ng} term(s) dropped on the right-hand side')
    out.update(verdict='wrong', detail='; '.join(why) or 'different expression')
    return out


def require_backends():
    """A missing backend would make every equation 'fail' and look like a parser result. Refuse instead."""
    missing = []
    for mod, pkg in (('antlr4', 'antlr4-python3-runtime==4.11.*'), ('lark', 'lark')):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise SystemExit('install the SymPy LaTeX backends first: pip install ' + ' '.join(f'"{m}"' for m in missing))


def main(paper_dir):
    require_backends()
    gold = json.load(open(f'{paper_dir}/gold_laws.json'))['laws']
    reader = json.load(open(f'{paper_dir}/paddle.json'))['formulas']
    rows = []
    for g, f in zip(gold, reader):
        ref = eval(g['sympy_reference'], {k: getattr(sp, k) for k in dir(sp)})   # srepr -> expression
        for source, tex in (('printed', g['latex_printed']), ('reader', f['latex'])):
            for backend in ('antlr', 'lark'):
                rows.append({'eq': g['eq'], 'source': source, 'latex': clean(tex), **judge(tex, ref, backend)})
    summary = {}
    for r in rows:
        k = f"{r['source']}/{r['backend']}"
        summary.setdefault(k, {'correct': 0, 'wrong': 0, 'fail': 0})[r['verdict']] += 1
    json.dump({'summary': summary, 'rows': rows}, open(f'{paper_dir}/sympy_parser_eval.json', 'w'),
              indent=1, ensure_ascii=False)
    print('of', len(gold), 'equations:')
    for k, v in summary.items():
        print(f'  {k:15} correct {v["correct"]:2}   wrong {v["wrong"]:2}   fail {v["fail"]:2}')
    print('\nprinted LaTeX, antlr backend:')
    for r in rows:
        if r['source'] == 'printed' and r['backend'] == 'antlr':
            print(f"  eq {r['eq']:>2} {r['verdict']:8} {r.get('detail', '')}")
    print(f'\nwritten -> {paper_dir}/sympy_parser_eval.json')


if __name__ == '__main__':
    main(sys.argv[1])
