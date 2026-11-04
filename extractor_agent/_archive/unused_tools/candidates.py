"""candidates.py -- for every symbol the equations use, find the places in the paper that
might define it. Code only, no LLM.

A candidate is one snippet: a sentence, one line of a list (e.g. a nomenclature), or one
table row. Snippets come from the reader's blocks (paddle_blocks.json). Nothing here
depends on a paper HAVING a nomenclature or a table; those are just more snippets.

A snippet is a candidate for symbol S when
  - it mentions S (symbol names compared after normalising notation, see key()), or
  - it is the sentence just before an equation that computes S (or first uses it,
    if nothing computes it). That catches implied definitions such as
    'The specific cell growth rate is determined by ...' right above mu = ...

Every snippet also gets a TEXT-LAYER CHECK: its words are compared with the words the
PDF itself stores in the same region (pymupdf). The reader (Paddle) is a vision-language
model and can substitute plausible words ('growth' for 'death'); the text layer is an
independent copy of the same page, so a disagreement is flagged. Scanned PDFs have no
text layer: their snippets are marked 'unverifiable', never 'agrees'.
"""
from __future__ import annotations
import html, json, re, unicodedata
from dataclasses import dataclass, field, asdict

import pymupdf

# ----------------------------------------------------------------------
# Symbol keys
# ----------------------------------------------------------------------
GREEK_ALIASES = {'varepsilon': 'epsilon', 'lambda': 'lam', 'nu': 'v', 'varphi': 'phi'}
# Characters OCR confuses with each other. Used only for the 'confusable' match level.
CONFUSABLE = str.maketrans({'I': 'l', '1': 'l', '|': 'l', 'O': '0', 'o': '0'})

def _alnum(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', s)

def key(name: str) -> str:
    """Symbol name -> comparable key (works on LaTeX names too: tokens("K_{d,amm}") gives the same key).
    'K_d_amm' -> 'K_damm', 'mu_d_max' -> 'mu_dmax', 'C_H_2L_2_ER' -> 'C_H2L2ER'."""
    base, _, sub = name.partition('_')
    base = GREEK_ALIASES.get(base, base)
    return base + ('_' + _alnum(sub) if sub else '')

def flat(k: str) -> str:
    return k.replace('_', '')

_TOKEN = re.compile(r'\[([^\]]+)\](?:_\{([^}]*)\}|_([A-Za-z0-9]))?'          # [GLC]_{in}
                    r'|(\\?[A-Za-z]+)(?:_\{([^}]*)\}|_([A-Za-z0-9]))?')      # K_{d,amm}, \mu_{max}, KI_{lac}

def tokens(text: str) -> set[str]:
    """Symbol-like tokens in a snippet, as keys."""
    s = re.sub(r'([A-Za-z])\s*\$\s*(_\{[^}]*\})\s*\$', r'\1\2', text)    # 'K $ _{A} $' -> 'K_{A}'
    s = s.replace('$', ' ')
    s = re.sub(r'_\{\s*([^}]*?)\s*\}', lambda m: '_{' + m.group(1).replace(' ', '') + '}', s)
    out = set()
    for m in _TOKEN.finditer(s):
        if m.group(1) is not None:
            sub = m.group(2) or m.group(3) or ''
            out.add('C_' + _alnum(m.group(1)) + _alnum(sub))
        else:
            base = m.group(4).lstrip('\\')
            base = GREEK_ALIASES.get(base, base)
            sub = m.group(5) or m.group(6) or ''
            out.add(base + ('_' + _alnum(sub) if sub else ''))
    return out

def match_level(sym_key: str, tok: str) -> str | None:
    """How a snippet token matches a symbol. None = no match."""
    if tok == sym_key:
        return 'exact'
    a, b = flat(sym_key), flat(tok)
    if len(a) >= 3 and a == b:
        return 'notation'                     # K_Ilac vs KI_lac: same letters, split differently
    if len(a) >= 3 and a.translate(CONFUSABLE) == b.translate(CONFUSABLE):
        return 'confusable'                   # KI_amm vs K_lamm: OCR I/l confusion
    if len(a) >= 3 and len(b) == len(a) + 1 and b.translate(CONFUSABLE).endswith(a.translate(CONFUSABLE)):
        return 'partial'                      # equation has I_lac, paper defines KI_lac: one name split in two?
    return None

# ----------------------------------------------------------------------
# Snippets
# ----------------------------------------------------------------------
@dataclass
class Snippet:
    id: int
    page: int
    kind: str                 # sentence | line | table_row | caption
    text: str
    block: int                # index into the blocks list
    textlayer: str = ''       # agrees | disagrees | unverifiable | too_short
    textlayer_diff: list = field(default_factory=list)

_SENT = re.compile(r'(?<=[.;:])\s+(?=[A-Z(\[$\\])')
_ABBREV = re.compile(r'\b(?:al|Fig|Figs|Eq|Eqs|e\.g|i\.e|vs|ca|resp)\.$')

def sentences(text: str) -> list[str]:
    parts, buf = [], ''
    for piece in _SENT.split(text):
        buf = f'{buf} {piece}'.strip() if buf else piece
        if not _ABBREV.search(buf):
            parts.append(buf); buf = ''
    if buf:
        parts.append(buf)
    return [p for p in parts if p.strip()]

def table_rows(html_table: str) -> list[str]:
    rows = []
    for tr in re.findall(r'<tr>(.*?)</tr>', html_table, re.S):
        cells = [html.unescape(re.sub(r'<[^>]+>', '', c)).strip()
                 for c in re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)]
        rows.append(cells)
    if not rows:
        return []
    header = rows[0]
    out = []
    for r in rows[1:]:
        if len(header) == len(r) and any(header):
            out.append(' | '.join(f'{h}: {c}' if h else c for h, c in zip(header, r)))
        else:
            out.append(' | '.join(r))
    return out

SKIP = {'header', 'number', 'footnote', 'reference_content', 'header_image', 'image',
        'chart', 'formula_number', 'display_formula', 'inline_formula', 'seal', 'aside_text'}

def build_snippets(blocks: list[dict]) -> list[Snippet]:
    out = []
    for bi, b in enumerate(blocks):
        lab, text = b['label'], b['content'] or ''
        if lab in SKIP or not text.strip():
            continue
        if lab == 'table':
            pieces, kind = table_rows(text), 'table_row'
        elif lab in ('figure_title', 'table_title'):
            pieces, kind = [text], 'caption'
        elif text.count('\n') >= 3:                    # a list laid out line by line (e.g. nomenclature)
            pieces, kind = [l for l in text.split('\n') if l.strip()], 'line'
        else:
            pieces, kind = sentences(text.replace('\n', ' ')), 'sentence'
        for p in pieces:
            out.append(Snippet(len(out), b['page'], kind, p.strip(), bi))
    return out

# ----------------------------------------------------------------------
# Text-layer check
# ----------------------------------------------------------------------
def _words(s: str) -> list[str]:
    s = unicodedata.normalize('NFKC', s).lower()      # also turns the 'fi' ligature into 'fi'
    s = re.sub(r'[\u2010-\u2015\u2212]', '-', s)        # en/em dashes and minus signs -> hyphen
    s = re.sub(r'\$[^$]*\$', ' ', s)                   # math is checked elsewhere; text layer garbles it
    s = re.sub(r'([a-z])-\s*([a-z])', r'\1\2', s)       # 'estima- tion', 'fed-batch' -> one word, both sides alike
    return re.findall(r'[a-z]{3,}', s)

def align(seq: list[str], ref: list[str]) -> list:
    """Semi-global alignment (Needleman-Wunsch with free gaps at both ends of ref).
    Returns, for each word of seq: None if matched, the ref word it was SUBSTITUTED by,
    or '' if it has no counterpart. Unlike difflib (longest run first), this prefers one
    substituted word over shifting a whole phrase, so an error lands on the right line."""
    n, m = len(seq), len(ref)
    MATCH, MIS, GAP = 2, -1, -1
    score = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]       # 0 diag, 1 up (gap in ref), 2 left (gap in seq)
    for i in range(1, n + 1):
        score[i][0] = i * GAP; back[i][0] = 1
    for j in range(1, m + 1):
        back[0][j] = 2                                   # leading ref words are free
    for i in range(1, n + 1):
        si, row, prev = seq[i - 1], score[i], score[i - 1]
        for j in range(1, m + 1):
            d = prev[j - 1] + (MATCH if si == ref[j - 1] else MIS)
            u = prev[j] + GAP
            l = row[j - 1] + (GAP if i < n else 0)       # trailing ref words are free
            best = max(d, u, l)
            row[j] = best
            back[i][j] = 0 if best == d else (1 if best == u else 2)
    out = [None] * n
    i, j = n, max(range(m + 1), key=lambda jj: score[n][jj])
    while i > 0:
        bk = back[i][j] if j > 0 else 1
        if bk == 0:
            if seq[i - 1] != ref[j - 1]:
                out[i - 1] = ref[j - 1]
            i -= 1; j -= 1
        elif bk == 1:
            out[i - 1] = ''; i -= 1
        else:
            j -= 1
    return out

def textlayer_check(snippets: list[Snippet], blocks: list[dict], pdf_path: str, dpi: int):
    """Align each block's words with the PDF's own words in the same region, as one
    sequence, then attribute disagreements to the snippets of that block. Aligning the
    whole block (not one snippet at a time) matters: 'maximum specific growth rate'
    exists elsewhere in the same list, so a per-snippet search would wrongly pass it."""
    doc = pymupdf.open(pdf_path)
    has_layer = any(p.get_text().strip() for p in doc)
    scale = 72.0 / dpi
    by_block: dict[int, list[Snippet]] = {}
    for s in snippets:
        by_block.setdefault(s.block, []).append(s)
    for bi, snips in by_block.items():
        if not has_layer:
            for s in snips: s.textlayer = 'unverifiable'
            continue
        b = blocks[bi]
        x0, y0, x1, y1 = (v * scale for v in b['bbox'])
        page = doc[b['page']]
        region = pymupdf.Rect(x0 - 6, y0 - 6, x1 + 6, y1 + 6)
        layer = _words(' '.join(w[4] for w in page.get_text('words', clip=region, sort=True)))
        if b['label'] == 'table':
            # The PDF stores tables column by column, so word ORDER means nothing here.
            # Check that every word of the row exists somewhere in the table region.
            pool = set(layer)
            for s in snips:
                ws = _words(s.text)
                missing = [w for w in ws if w not in pool]
                s.textlayer = 'too_short' if len(ws) < 2 else ('disagrees' if missing else 'agrees')
                s.textlayer_diff = [f'reader "{w}" not in pdf table' for w in missing]
            continue
        seq, owner = [], []
        for s in snips:
            ws = _words(s.text)
            seq += ws; owner += [s.id] * len(ws)
        res = align(seq, layer)
        if seq and sum(r is None for r in res) < 0.8 * len(seq):
            # Paddle sometimes merges text across columns, so its box misses part of the
            # block. Fall back to the whole page.
            res = align(seq, _words(page.get_text()))
        bad: dict[int, list] = {}
        for k, r in enumerate(res):
            if r is not None:
                bad.setdefault(owner[k], []).append((seq[k], r))
        for s in snips:
            n = len(_words(s.text))
            if n < 2:
                s.textlayer = 'too_short'
            elif s.id in bad:
                s.textlayer = 'disagrees'
                s.textlayer_diff = [f'reader "{a}" vs pdf "{b}"' if b else f'reader "{a}" not in pdf'
                                    for a, b in bad[s.id]]
            else:
                s.textlayer = 'agrees'

# ----------------------------------------------------------------------
# Candidates per symbol
# ----------------------------------------------------------------------
def context_sentences(blocks: list[dict], snippets: list[Snippet], formula_bboxes: list) -> dict[int, int]:
    """formula index (reading order) -> id of the last sentence before it on the same page."""
    by_bbox = {tuple(map(float, b['bbox'])): i for i, b in enumerate(blocks) if b['label'] == 'display_formula'}
    out = {}
    for fi, bbox in enumerate(formula_bboxes):
        bi = by_bbox.get(tuple(map(float, bbox)))
        if bi is None:
            continue
        prev = [s for s in snippets if s.kind == 'sentence' and s.page == blocks[bi]['page'] and s.block < bi]
        if prev:
            out[fi] = max(prev, key=lambda s: (s.block, s.id)).id
    return out

def find_candidates(symbols: dict, snippets: list[Snippet], ctx: dict[int, int], max_per_symbol: int = 10):
    """symbols: {name: {'computed_by': [law idx], 'used_by': [law idx]}}"""
    tok = {s.id: tokens(s.text) for s in snippets}
    out = {}
    for name, info in symbols.items():
        k = key(name)
        found = []
        for s in snippets:
            levels = [lv for t in tok[s.id] if (lv := match_level(k, t))]
            if levels:
                best = min(levels, key=['exact', 'notation', 'confusable', 'partial'].index)
                found.append((s.id, best))
        laws = info['computed_by'] or info['used_by'][:1]
        for li in laws:
            if li in ctx and ctx[li] not in [f[0] for f in found]:
                found.append((ctx[li], 'context'))
        out[name] = found[:max_per_symbol]
    return out
