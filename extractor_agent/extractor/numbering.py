"""numbering.py -- equation numbers from the OCR blocks, for any paper layout.

    python -m extractor.numbering papers/xing_2010            # rewrite paddle.json (backup: paddle.orig.json)
    python -m extractor.numbering papers/xing_2010 --check    # only print what it would change

Rebuilds the 'formulas' list of paddle.json from paddle_blocks.json (both main paper and supplementary).
Run it on a paper straight after OCR, before any LLM step; not on a paper whose paddle.json was
corrected by hand (it would undo the corrections; the backup keeps the first version only).

Numbers are kept as printed, as text: (12) -> 12, (2a) -> '2a', (A1) -> 'A1', [3] -> 3, （21） -> 21.
Which number belongs to which formula:
    1. a \\tag{} or a trailing (n) / [n] inside the formula's own LaTeX
    2. otherwise by position, not reading order (the OCR's order of number blocks is unreliable): a number
       to the right of the formula in the same column, on the same line, or just below it (a number
       printed under a wide formula). Each number goes to one formula; numbers level with the formula
       win, then the closest.
A formula block that is only a condition or an index range ('d = 0,1,...13', 'if [AMM] <= 4.6 mM then
...', 'k = 4,5,6; j = 2,3') and carries a number gives the number to the formula just before it, and
becomes that formula's 'condition'. Formulas without a number keep eq = None (paper.py gives them ids).
"""
from __future__ import annotations
import json
import os
import re
import shutil
import sys

NUM = r'[A-Z]{0,2}\s*[-.]?\s*\d+\s*[a-z]?'
_FULLWIDTH = str.maketrans('（）［］', '()[]')


def number_of(text: str):
    """'(12)' -> 12, '(2a)' -> '2a', '(A-1)' -> 'A-1', '[3]' -> 3; None if the text is not a number."""
    t = (text or '').translate(_FULLWIDTH).strip().strip('$').strip()
    m = re.fullmatch(rf'[(\[]\s*({NUM})\s*[)\]]\.?', t) or re.fullmatch(rf'({NUM})', t)
    return _clean(m.group(1)) if m else None


def own_number(tex: str):
    """A number written inside the formula: \\tag{12}, or a trailing (12) / [3]."""
    t = (tex or '').translate(_FULLWIDTH)
    m = re.search(r'\\tag\{([^}]+)\}', t) or re.search(rf'[(\[]\s*({NUM})\s*[)\]]\s*\$*\s*$', t)
    return _clean(m.group(1)) if m else None


def _clean(s: str):
    s = re.sub(r'\s', '', s)
    return int(s) if s.isdigit() else s


_CONDITION = re.compile(r'''^\s*(?:\\left\\\{|\\begin\{(?:aligned|cases|array)\}(?:\{[^}]*\})?|&|\\quad|\s)*
    (?:if\b|\\text\{\s*if|for\b|\\text\{\s*for|where\b|
       [a-z](?:_\{?\w+\}?)?\s*=\s*\d+\s*,\s*(?:\d+\s*,\s*)*(?:\\l?dots|\\cdots|\.\.\.|\d))''', re.X | re.I)


def is_condition(tex: str) -> bool:
    """A block that only states when or for which index the equation holds."""
    t = re.sub(r'\$', '', tex or '').strip()
    return bool(_CONDITION.match(t))


def _box(b):
    return list(map(float, b['bbox']))


def _same_line(a, b):
    return min(a[3], b[3]) - max(a[1], b[1]) > 0.5 * min(a[3] - a[1], b[3] - b[1])


def renumber(blocks: list[dict]) -> list[dict]:
    out = []
    by_page = {}
    for i, b in enumerate(blocks):
        by_page.setdefault((b.get('source', 'main'), b['page']), []).append(i)
    for (source, page), idx in by_page.items():
        bl = [blocks[i] for i in idx]
        is_num = lambda b: 'number' in b['label'] and 'formula' in b['label']
        is_formula = lambda b: 'formula' in b['label'] and 'number' not in b['label'] and 'inline' not in b['label']
        got = {}
        formulas = [k for k, b in enumerate(bl) if is_formula(b)]
        numbers = [k for k, b in enumerate(bl) if is_num(b)]
        for k in formulas:                               # 1: a number written inside the formula
            got[k] = own_number(bl[k]['content'])
        width = max(_box(b)[2] for b in bl)              # page width in pixels, for the column test
        pairs = []                                       # 2: geometry, one number per formula
        for k in formulas:
            if got[k] is not None:
                continue
            f = _box(bl[k])
            for j in numbers:
                x = _box(bl[j])
                gap, cy = x[0] - f[2], (x[1] + x[3]) / 2
                # to the right, in the same column; the formula's box often runs into the number's
                if not (gap > -70 and x[0] > (f[0] + f[2]) / 2 and gap < 0.5 * width):
                    continue
                if not (f[1] - 10 <= cy <= f[3] + 45):    # same line, or just below a tall/wide formula
                    continue
                inside = f[1] <= cy <= f[3]
                nxt = 0 if j == k + 1 else 1             # the OCR's reading order breaks ties
                pairs.append((0 if inside else 1, nxt, abs(cy - (f[1] + f[3]) / 2), k, j))
        used = set()
        for *_, k, j in sorted(pairs):
            if got[k] is None and j not in used:
                got[k] = number_of(bl[j]['content'])
                used.add(j)
        rows = []
        for k, b in enumerate(bl):
            if not is_formula(b):
                continue
            row = {'eq': got.get(k), 'page': page, 'label': b['label'], 'latex': b['content'], 'bbox': _box(b),
                   'route': 'pipeline', 'source': source}
            prev = rows[-1] if rows else None
            between = bl[prev['_k'] + 1:k] if prev is not None else None
            joined = prev is not None and all(x['label'] == 'text' and len((x['content'] or '').strip()) <= 20
                                              for x in between)       # nothing, or only 'where' / 'with'
            if is_condition(b['content']) and joined:
                # a condition or index range: belongs to the formula just before it
                if row['eq'] is not None and prev['eq'] is None:
                    prev['eq'] = row['eq']
                prev['condition'] = ((prev.get('condition') or '') + ' ' + b['content'].strip().strip('$').strip()).strip()
                prev['_k'] = k
                continue
            row['_k'] = k
            rows.append(row)
        for r in rows:
            r.pop('_k', None)
        out += rows
    return out


def report(formulas: list[dict]) -> list[str]:
    """Problems a person should look at: repeated numbers, numbers that go down in reading order."""
    msgs = []
    for source in ('main', 'supplementary'):
        nums = [f['eq'] for f in formulas if f.get('source', 'main') == source and f['eq'] is not None]
        seen = set()
        for n in nums:
            if n in seen:
                msgs.append(f'{source}: number {n} appears twice')
            seen.add(n)
        ints = [n for n in nums if isinstance(n, int)]
        for a, b in zip(ints, ints[1:]):
            if b <= a:
                msgs.append(f'{source}: ({b}) follows ({a}) in reading order: OCR misread a number?')
        if ints:
            gaps = sorted(set(range(1, max(ints) + 1)) - set(ints))
            if gaps:
                msgs.append(f'{source}: no equation numbered {gaps} (a number in the text, a table, or missed by OCR?)')
    return msgs


def main(paper_dir: str, check: bool):
    blocks = json.load(open(f'{paper_dir}/paddle_blocks.json'))['blocks']
    path = f'{paper_dir}/paddle.json'
    old = json.load(open(path))
    new = renumber(blocks)
    before = [f['eq'] for f in old['formulas']]
    after = [f['eq'] for f in new]
    print(f'{len(old["formulas"])} formulas before, {len(new)} after; '
          f'{sum(e is None for e in before)} -> {sum(e is None for e in after)} without a number')
    for f in new:
        if f.get('condition'):
            print(f"  eq {f['eq']}: condition '{f['condition'][:70]}'")
    for m in report(new):
        print('  check:', m)
    if check:
        print('  ids now:', after)
        return
    if not os.path.exists(f'{paper_dir}/paddle.orig.json'):
        shutil.copy(path, f'{paper_dir}/paddle.orig.json')
    json.dump({**old, 'formulas': new, 'numbered_by': 'extractor/numbering.py'}, open(path, 'w'), indent=1, ensure_ascii=False)
    print(f'-> {path} (first version kept as paddle.orig.json)')


if __name__ == '__main__':
    main(sys.argv[1], '--check' in sys.argv)
