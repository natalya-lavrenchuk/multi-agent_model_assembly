"""split.py -- one printed block, several equations: split it, one law per equation.

Papers print systems under one number (Xing eq 6: six ODEs in one brace; Vanden Bussche [3]: both rate
laws; Nagy eq 11: the moment equations as one matrix equation). The rule 'one printed equation = one
Law' means each of those equations is its own law. Parts get ids '6.1', '6.2', ... and 'part_of': 6.

    split_block(latex) -> [(latex_of_part, condition_or_None), ...] if the block holds 2+ equations, else None

A line belongs to the equation before it when it has no '=' of its own (the right-hand side continued on
the next line) or starts with '=' (a further step of the same equation). A line starting with 'if',
'for' or 'where' is a condition on the equation before it. A matrix equation [a1; a2] = [b1; b2] with
the same number of rows on both sides becomes a1 = b1, a2 = b2.
"""
from __future__ import annotations
import re

_ENV = r'(?:aligned|align\*?|cases|array|gathered|split|eqnarray\*?|gather\*?)'


def _strip(tex: str) -> str:
    t = tex.strip()
    t = re.sub(r'^\$\$|\$\$$', '', t).strip()
    t = re.sub(r'^\\\[|\\\]$', '', t).strip()
    t = re.sub(r'^\\left\\\{|\\right\.\s*$', '', t).strip()          # a brace around a system
    t = re.sub(r'^\\left\\\{|\\right\.\s*$', '', t).strip()
    return t


def _top_level_split(s: str, sep: str) -> list[str]:
    """Split at sep where it is outside braces and outside nested \\begin..\\end."""
    out, depth, env, i, start = [], 0, 0, 0, 0
    while i < len(s):
        if s.startswith('\\begin{', i):
            env += 1
        elif s.startswith('\\end{', i):
            env -= 1
        c = s[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        elif depth == 0 and env == 0 and s.startswith(sep, i) and not (sep == '=' and _in_command(s, i)):
            out.append(s[start:i])
            i += len(sep)
            start = i
            continue
        elif c == '\\' and i + 1 < len(s) and s[i + 1] != '\\':
            i += 1                                          # skip an escaped character ('\{', '\,')
        i += 1
    out.append(s[start:])
    return out


def _in_command(s, i):
    return False


def _has_eq(line: str) -> bool:
    return len(_top_level_split(line, '=')) > 1


def _matrix_rows(m: str) -> list[str]:
    return [r.strip() for r in _top_level_split(m, '\\\\') if r.strip()]


_MATRIX = re.compile(r'^\\begin\{([bpvB]?matrix)\}(.*)\\end\{\1\}$', re.S)


def _matrix_equation(t: str):
    sides = _top_level_split(t.replace('&', ''), '=')
    if len(sides) != 2:
        return None
    a, b = (_MATRIX.match(x.strip()) for x in sides)
    if not (a and b):
        return None
    left, right = _matrix_rows(a.group(2)), _matrix_rows(b.group(2))
    if len(left) == len(right) >= 2 and all('&' not in r for r in left + right):
        return [(f'{l} = {r}', None) for l, r in zip(left, right)]
    return None


def split_block(tex: str):
    t = _strip(tex)
    m = re.fullmatch(rf'\\begin\{{{_ENV}\}}(?:\{{[^}}]*\}})?(.*)\\end\{{{_ENV}\}}', t, re.S)
    inner = m.group(1) if m else t
    mat = _matrix_equation(inner.strip())
    if mat:
        return mat
    if not m:
        return None
    parts = []                                             # [latex, condition]
    for line in _top_level_split(inner, '\\\\'):
        line = re.sub(r'\\(?:q)?quad\b', ' ', line.replace('&', ' ')).strip().rstrip(',;.').strip()
        if not line:
            continue
        word = re.sub(r'\\text\{\s*|\\mathrm\{\s*', '', line).lower()
        if re.match(r'(if|for|where|when)\b', word) and parts:
            parts[-1][1] = ((parts[-1][1] or '') + ' ' + line).strip()
            continue
        sides = _top_level_split(line, '=')
        if len(sides) > 1 and sides[0].strip() and parts is not None:
            parts.append([line, None])                     # a new equation: its own left-hand side
        elif parts:
            parts[-1][0] += ' ' + line                     # the right-hand side goes on, or '= ...' again
        else:
            parts.append([line, None])
    eqs = [p for p in parts if _has_eq(p[0])]
    return [(p[0], p[1]) for p in eqs] if len(eqs) >= 2 else None
