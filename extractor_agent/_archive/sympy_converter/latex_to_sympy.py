"""latex2sympy_det.py -- deterministic LaTeX -> SymPy for printed process-model equations.

No LLM. Rules written for how these papers typeset models:
  [GLC], [H_2L_2]_{ER}   -> one concentration symbol (C_GLC, C_H2L2_ER), never G*L*C
  X_{abc}, K_{d,amm}     -> one symbol (X_abc, K_d_amm), never X_a*b*c
  \\frac{d(...)}{dt}      -> Derivative(..., t)
  adjacent operands      -> implicit multiplication (V X_v -> V*X_v)
  ', \\quad n > 1'        -> stripped and kept as a side condition
Anything it cannot handle raises, so a failure is loud, never a silent wrong parse.
"""
from __future__ import annotations
import re
import sympy as sp

GREEK = {'alpha','beta','gamma','delta','epsilon','varepsilon','zeta','eta','theta','kappa',
         'lambda','mu','nu','xi','pi','rho','sigma','tau','phi','varphi','chi','psi','omega',
         'Gamma','Delta','Theta','Lambda','Sigma','Phi','Psi','Omega'}
FUNCS = {'exp','ln','log','sqrt'}

def _group(s, i, o='{', c='}'):
    """s[i] == o; return (content, index after the closing bracket)."""
    assert s[i] == o, (s, i)
    depth = 0
    for j in range(i, len(s)):
        depth += (s[j] == o) - (s[j] == c)
        if depth == 0:
            return s[i+1:j], j + 1
    raise ValueError(f'unbalanced {o}{c} in {s!r}')

def _clean_sub(t):
    t = re.sub(r'\\(max|min|mathrm|text|rm)\b', r'\1', t).replace('mathrm', '').replace('text', '')
    t = re.sub(r'[{}\s\\]', '', t)
    return re.sub(r'[^A-Za-z0-9]+', '_', t).strip('_')

def preprocess(tex):
    cond = None
    m = re.search(r',?\s*\\quad\s*(.+)$', tex)
    if m:
        cond, tex = m.group(1).strip(' .,'), tex[:m.start()]
    tex = tex.strip().strip('$').strip().strip(' .,')
    tex = tex.replace('\\nu', 'v')       # PROVISIONAL: Kontoravdi's italic v is read as nu by some OCR
    tex = re.sub(r'\\(left|right|,|;|!|quad|qquad)', ' ', tex)
    # LaTeX ignores spaces in math ('F_{o u t}' == 'F_{out}'), except the one ending a \command
    tex = re.sub(r'(\\[A-Za-z]+)\s+', lambda m: m.group(1) + '\x00', tex)
    tex = re.sub(r'\s+', '', tex).replace('\x00', ' ')
    tex = tex.replace('\\cdot', '*').replace('\\times', '*')
    return tex, cond

def tokenize(s):
    toks, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1; continue
        if s.startswith('\\frac', i):
            num, j = _group(s, i + 5)
            den, k = _group(s, j)
            dm = re.fullmatch(r'\s*d\s*(.+?)\s*', num)
            if den.strip() == 'dt' and dm:                      # derivative
                body = dm.group(1)
                if body[0] in '([' and _group(body, 0, body[0], ')' if body[0] == '(' else ']')[1] == len(body) and body[0] == '(':
                    body = body[1:-1]
                toks.append(('FUNC', 'Derivative')); toks.append(('(', '('))
                toks += tokenize(body); toks.append(('OP', ',')); toks.append(('NAME', 't'))
                toks.append((')', ')'))
            else:
                toks.append(('(', '(')); toks += [('(', '(')] + tokenize(num) + [(')', ')')]
                toks.append(('OP', '/')); toks += [('(', '(')] + tokenize(den) + [(')', ')')]
                toks.append((')', ')'))
            i = k; continue
        if ch == '[':                                           # concentration
            inner, j = _group(s, i, '[', ']')
            name = 'C_' + _clean_sub(inner)
            if j < len(s) and s[j] == '_':
                sub, j = (_group(s, j + 1) if s[j+1] == '{' else (s[j+1], j + 2))
                name += '_' + _clean_sub(sub)
            toks.append(('NAME', name)); i = j; continue
        if ch == '\\':
            m = re.match(r'\\([A-Za-z]+)', s[i:])
            word = m.group(1); i += m.end()
            if word in FUNCS:
                toks.append(('FUNC', {'ln': 'log'}.get(word, word))); continue
            if word not in GREEK:
                raise ValueError(f'unknown command \\{word}')
            name = word.replace('varepsilon', 'epsilon').replace('lambda', 'lam')
        elif ch.isalpha():
            name = ch; i += 1
        elif ch.isdigit() or ch == '.':
            m = re.match(r'\d+(\.\d+)?', s[i:]); toks.append(('NUM', f"Rational('{m.group()}')")); i += m.end(); continue
        elif ch in '()':
            toks.append((ch, ch)); i += 1; continue
        elif ch == '{':
            inner, i = _group(s, i); toks += [('(', '(')] + tokenize(inner) + [(')', ')')]; continue
        elif ch in '+-*/=,':
            toks.append(('OP', ch)); i += 1; continue
        elif ch == '^':
            toks.append(('OP', '**')); i += 1
            if s[i] == '{':
                inner, i = _group(s, i); toks += [('(', '(')] + tokenize(inner) + [(')', ')')]
            else:
                toks += tokenize(s[i]); i += 1
            continue
        else:
            raise ValueError(f'unexpected {ch!r} in {s!r}')
        # a symbol: attach its subscript
        if i < len(s) and s[i] == '_':
            sub, i = (_group(s, i + 1) if s[i+1] == '{' else (s[i+1], i + 2))
            name += '_' + _clean_sub(sub)
        toks.append(('NAME', name))
    return toks

def to_python(toks):
    out, prev = [], None
    for kind, val in toks:
        if prev in ('NAME', 'NUM', ')') and kind in ('NAME', 'NUM', '(', 'FUNC'):
            out.append('*')                                     # implicit multiplication
        out.append(val); prev = kind
    return ' '.join(out)

def parse(tex):
    body, cond = preprocess(tex)
    if body.count('=') != 1:
        raise ValueError('expected exactly one "="')
    lhs, rhs = body.split('=')
    lp, rp = to_python(tokenize(lhs)), to_python(tokenize(rhs))
    names = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', lp + ' ' + rp)) - {'Derivative', 'exp', 'log', 'sqrt', 'Rational'}
    ns = {n: sp.Symbol(n) for n in names}
    ns.update(Rational=sp.Rational, Derivative=sp.Derivative, exp=sp.exp, log=sp.log, sqrt=sp.sqrt)
    eq = sp.Eq(eval(lp, {'__builtins__': {}}, ns), eval(rp, {'__builtins__': {}}, ns), evaluate=False)
    return eq, cond

def warnings_for(tex):
    """Ambiguities a deterministic rule cannot settle: these go to the LLM or a human."""
    w = []
    body, _ = preprocess(tex)
    plain = re.sub(r'\[[^\]]*\](_\{[^}]*\}|_.)?|_\{[^}]*\}|_.|\\[A-Za-z]+', ' ', body)
    for run in re.findall(r'[A-Za-z]{3,}', plain):
        w.append(f'bare letter run "{run}": product of {len(run)} symbols, or one name?')
    return w
