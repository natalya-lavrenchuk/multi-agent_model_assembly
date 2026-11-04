"""paper.py -- read one paper's OCR output: the equations and the text.

Input files in papers/<id>/ (made on Colab by colab/paddleocr_vl_extract.ipynb):
    paddle.json         equations as LaTeX, with equation number and page
    paddle_blocks.json  every text block in reading order; tables as HTML
    gold_laws.json      hand-checked equations (only for the reference graph)

Code only, no LLM:
    read_equations(formulas)    per equation: symbols on each side of '=', ODE / PDE /
                                algebraic, independent variables (t, z, ...), side condition.
                                A symbol lister, NOT a parser: nothing is translated to SymPy.
    load_snippets(paper_dir)    the text cut into snippets: sentences, list lines, table rows
    mention(symbol, text)       does this text mention the symbol (and how closely)?
    table_value(symbol, snips)  the first table row with a number that mentions the symbol
"""
from __future__ import annotations
import html
import json
import os
import re

# =============================================================================
# Loading
# =============================================================================

def load_model(paper_dir: str, gold: bool = False, model: dict | None = None,
               allow_proposed: bool = False) -> tuple[list[dict], list[dict], str]:
    """The paper's model equations, read: (formulas, reads, source name).
    paper.json says which printed equations are the model (a paper also prints estimation set-ups,
    general forms, feed profiles). A paper with rival models lists several under 'models': then
    model=<one of models(cfg)> gives that model's equations, and no model gives all of them (the
    symbols and laws of a paper are described once, whichever model uses them).
    Multi-letter names written as words in the paper's prose (DOT, not XV) are read as one symbol."""
    formulas, source = load_equations(paper_dir, gold)
    cfg = json.load(open(f'{paper_dir}/paper.json')) if os.path.exists(f'{paper_dir}/paper.json') else {}
    if cfg.get('status') == 'proposed' and not allow_proposed:
        raise SystemExit(f'{paper_dir}/paper.json is a proposal: check it (python -m extractor.review), then '
                         f'python -m extractor.sup {paper_dir} --approve')
    chosen = [model] if model else models(cfg)
    if all(m.get('equations') for m in chosen):
        wanted = {str(e) for m in chosen for e in m['equations']}   # ids may be written 12 or '12' (or 'A-3')
        formulas = [f for f in formulas if str(f.get('eq')) in wanted or str(f.get('part_of')) in wanted]   # '6' = all of 6.1 ... 6.6
    return formulas, read_equations(formulas, prose_names(paper_dir)), source


def models(cfg: dict) -> list[dict]:
    """The Sups of one paper. paper.json describes one model (flat) or lists rival models under
    'models' (Monod vs Contois, Seidel's detailed and simplified kinetics). Each comes back with an
    'id' (the Sup id: the folder name, or folder name + model name) and its 'equations' (None: all).
    Laws the paper fits on their own, without a model around them ('free_laws': rate expressions compared
    to each other), come last as one group with 'free': True. They are not a Sup: graph.py writes them as
    Laws with no Sup, carrying the fit context on their own Metadata."""
    free = cfg.get('free_laws') or []
    group = [{'id': f"{cfg.get('sup')}_free_laws", 'name': 'free-floating laws (no Sup)', 'free': True,
              'equations': [str(x['eq']) for x in free], 'law_names': {str(x['eq']): x.get('name') for x in free},
              'model_assumptions': []}] if free else []
    if cfg.get('models'):
        slug = lambda t: re.sub(r'[^a-z0-9]+', '_', t.lower()).strip('_')[:40]
        return [{**m, 'id': m.get('sup') or f"{cfg.get('sup')}_{slug(m.get('name') or str(i + 1))}"}
                for i, m in enumerate(cfg['models'])] + group
    if free and not cfg.get('equations'):                 # the paper only compares single laws (Muloiwa 2020)
        return group
    keep = {k: v for k, v in cfg.items() if k not in ('excluded', 'checks', 'code_analysis', 'models', 'free_laws')}
    return [{**keep, 'id': cfg.get('sup'), 'name': cfg.get('model_name') or cfg.get('sup')}] + group


def prose_names(paper_dir: str) -> set[str]:
    """Upper-case words of 2-5 letters used as plain words in the text (outside $...$): 'DOT' in
    'dissolved oxygen DOT'. Evidence that 'D O T' in an equation is one name, not D*O*T."""
    words = set()
    for snip in load_snippets(paper_dir):
        prose = re.sub(r'\$[^$]*\$', ' ', snip['text'])
        words |= set(re.findall(r'\b([A-Z]{2,5})(?=\d|\b)', prose))     # 'DOT', and 'CO' in 'CO2'
        # inline math that is nothing but one formula name: '$CO_{2}$', '$\\mathrm{CO}_2$'
        words |= set(re.findall(r'\$\s*(?:\\mathrm\s*\{)?\s*([A-Z]{2,5})\s*\}?\s*(?:_\s*\{?\s*\d+\s*\}?)?\s*\$', snip['text']))
    for snip in load_snippets(paper_dir):                 # math in tables and nomenclature lines: 'KI_{amm} (mM)'
        if snip['kind'] in ('table_row', 'line'):
            for piece in re.findall(r'\$([^$]*)\$', snip['text']):
                words |= set(re.findall(r'(?<![A-Za-z\\])([A-Z]{2,3})\s*_', piece))
    eq = os.path.join(paper_dir, 'paddle.json')           # '\mathrm{KI}_{lac}' in the equations themselves
    if os.path.exists(eq):
        for f in json.load(open(eq))['formulas']:
            words |= set(re.findall(r'\\mathrm\s*\{\s*([A-Z]{2,3})\s*\}\s*_', f.get('latex') or ''))
    return words


def load_equations(paper_dir: str, gold: bool = False) -> tuple[list[dict], str]:
    """[{'eq', 'latex', 'page'}], and a name for the source. gold=True reads the hand-checked set."""
    if gold:
        laws = json.load(open(f'{paper_dir}/gold_laws.json'))['laws']
        return [{'eq': r['eq'], 'latex': r['latex_printed'], 'page': 6} for r in laws], 'gold_laws.json (hand-checked)'
    f = json.load(open(f'{paper_dir}/paddle.json'))
    have = {x['eq'] for x in f['formulas']}
    extra = [x for x in _table_equations(paper_dir) if x['eq'] not in have]
    return _ids_and_parts(f['formulas'] + extra), f"paddle.json ({f.get('reader', 'reader')})"


_RELATION = re.compile(r'=|\\approx|\\equiv|\\cong|\\simeq|\\propto|\\leq?|\\geq?|<|>|\\rightleftharpoons|\\rightarrow|\\to\\b')


def _ids_and_parts(formulas: list[dict]) -> list[dict]:
    """Every equation gets an id, and a block that holds several equations becomes one entry per equation.
    Unnumbered equations (older papers number only a few: Vanden Bussche 1996 numbers 2 of 33) get
    'p5-2' (page 5, second unnumbered one on that page; 'S-p5-2' in a supplementary file). Unnumbered
    blocks with no relation at all ('67', 't^*': OCR fragments) are dropped. A block with several
    equations (Xing eq 6, six ODEs) becomes '6.1' ... '6.6', each with 'part_of': 6 (see split.py)."""
    from extractor.split import split_block
    out, count = [], {}
    for f in formulas:
        f = dict(f)
        if f.get('eq') is None:
            if not _RELATION.search(f.get('latex') or ''):
                continue
            key = (f.get('source', 'main'), f.get('page', 0))
            count[key] = count.get(key, 0) + 1
            f['eq'] = f"{'S-' if key[0] == 'supplementary' else ''}p{key[1] + 1}-{count[key]}"
            f['unnumbered'] = True
        parts = split_block(f.get('latex') or '')
        if not parts:
            out.append(f)
            continue
        for k, (tex, cond) in enumerate(parts, 1):
            c = ' '.join(x for x in (cond, f.get('condition')) if x) or None
            out.append({**f, 'eq': f"{f['eq']}.{k}", 'part_of': f['eq'], 'latex': f'$$ {tex} $$', 'condition': c})
    return out


def _table_equations(paper_dir: str) -> list[dict]:
    """Equations printed inside a table (Wang 2019: 'Table 1. Full Set of ODEs'): a row with a cell
    '(n)' and a cell whose math has '='. The math pieces of that cell are joined; notes after 'See'
    ('See P defined in Eqn (12)') are not part of the equation."""
    path = f'{paper_dir}/paddle_blocks.json'
    out = []
    for b in json.load(open(path))['blocks'] if os.path.exists(path) else []:
        if b['label'] != 'table':
            continue
        for row in re.findall(r'<tr>(.*?)</tr>', b['content'] or '', re.S):
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)
            num = next((int(m.group(1)) for c in cells if (m := re.fullmatch(r'\s*\((\d+)\)\s*', c))), None)
            maths = [_leading_math(c) for c in cells]
            tex = next((' '.join(m) for m in maths if '=' in ''.join(m)), None)
            if num is not None and tex:
                out.append({'eq': num, 'page': b['page'], 'label': 'table_formula', 'latex': f'$$ {tex} $$',
                            'bbox': b['bbox'], 'route': 'table', 'source': b.get('source', 'main')})
    return out


def _leading_math(cell: str) -> list[str]:
    """The math of a table cell up to its first line of words: the equation, not the note under it
    ('See P defined in Eqn (12), $Q_n$ in Eqn (18)'), whatever the note says."""
    out = []
    # the OCR writes line breaks in a cell as a newline or as the two characters '\\n' (not '\\nu', '\\neq')
    for line in re.split(r'\n|\\n(?![a-z])|<br\s*/?>', cell):
        pieces = re.findall(r'\$([^$]*)\$', line)
        words = re.sub(r'\$[^$]*\$', ' ', line)
        if re.search(r'[A-Za-z]{2,}', words) or (not pieces and words.strip()):
            break                                          # a line of text: the equation is over
        out += pieces
    return out


TEXT_LABELS = {'text', 'abstract', 'paragraph_title', 'figure_title', 'table_title', 'footnote', 'algorithm'}
_SENTENCE_END = re.compile(r'(?<=[.;:])\s+(?=[A-Z(\[$\\])')
_ABBREV = re.compile(r'\b(?:al|Fig|Figs|Eq|Eqs|e\.g|i\.e|vs|ca|resp)\.$')


def load_snippets(paper_dir: str, tables: bool = True) -> list[dict]:
    """[{'id', 'page', 'kind', 'text'}] in reading order. kind: sentence | line | table_row.
    A block with 3+ line breaks (a nomenclature list) is cut into lines; tables into rows,
    each row prefixed with the table's caption ('Table 2 Parameter estimation results ...: ...')."""
    path = f'{paper_dir}/paddle_blocks.json'
    if not os.path.exists(path):
        return []
    blocks = json.load(open(path))['blocks']
    out = []
    for n, b in enumerate(blocks):
        text = (b['content'] or '').strip()
        if not text:
            continue
        if b['label'] == 'table':
            caption = _caption(blocks, n)
            rows = _table_rows(text) if tables else []
            pieces, kind = [f'{caption}: {r}' if caption else r for r in rows], 'table_row'
        elif b['label'] not in TEXT_LABELS:
            continue
        elif text.count('\n') >= 3:
            pieces, kind = [l for l in text.split('\n') if l.strip()], 'line'
        else:
            pieces, kind = _sentences(text.replace('\n', ' ')), 'sentence'
        out += [{'id': len(out) + i, 'page': b['page'], 'kind': kind, 'text': p.strip()} for i, p in enumerate(pieces)]
    return out


def _caption(blocks: list[dict], n: int) -> str:
    """The title of table block n, found by layout, not wording: the nearest title or heading block
    next to it on the same page (above first, then below), skipping page furniture (header, page
    number, footer) and stopping at body text, another table or a figure. OCR often drops 'Table 2'
    and keeps only the title, and nomenclature tables have a section heading instead. A figure's
    caption ('Fig. 6 ...') is never taken. 'Table 3 continued' gets the first part's full title."""
    SKIP = {'header', 'footer', 'number', 'page_number', 'header_image', 'footer_image', 'footnote'}
    TITLE = {'figure_title', 'table_title', 'paragraph_title', 'doc_title', 'chart_title'}
    for step in (-1, 1):
        k = n + step
        while 0 <= k < len(blocks) and blocks[k]['page'] == blocks[n]['page'] and blocks[k]['label'] in SKIP:
            k += step
        if not (0 <= k < len(blocks)) or blocks[k]['page'] != blocks[n]['page']:
            continue
        if step == -1 and blocks[k]['label'] == 'text' and len(blocks[k]['content'] or '') < 200 and k > 0 \
                and re.fullmatch(r'\s*Table\s+[A-Z]?\d+\.?\s*', blocks[k - 1]['content'] or '', re.I):
            k -= 1                                     # OCR split: 'Table 4' heading + 'Stoichiometric matrix ...' text
            return re.sub(r'\s+', ' ', f"{blocks[k]['content']} {blocks[k + 1]['content']}").strip().rstrip('.')
        if blocks[k]['label'] not in TITLE:
            continue
        t = re.sub(r'\s+', ' ', blocks[k]['content'] or '').strip().rstrip('.')
        if not t or re.match(r'(fig|figure|scheme)\b', t, re.I):
            continue
        if m := re.match(r'Table\s+([A-Z]?\d+)\s*\(?(cont|continued)\b', t, re.I):
            first = next((c for c in (_caption(blocks, j) for j in range(n - 1, -1, -1) if blocks[j]['label'] == 'table')
                          if re.match(rf'Table\s+{m.group(1)}\b', c, re.I) and 'ontinued' not in c), None)
            if first:
                return first + ' (continued)'
        return t
    return ''


def _sentences(text: str) -> list[str]:
    parts, buf = [], ''
    for piece in _SENTENCE_END.split(text):
        buf = f'{buf} {piece}' if buf else piece
        if not _ABBREV.search(buf):                       # 'et al.' does not end a sentence
            parts.append(buf)
            buf = ''
    return [p for p in parts + [buf] if p.strip()]


def _table_rows(html_table: str) -> list[str]:
    rows = []
    for tr in re.findall(r'<tr>(.*?)</tr>', html_table, re.S):
        row = []
        for attrs, c in re.findall(r'<td([^>]*)>(.*?)</td>', tr, re.S):
            span = re.search(r'colspan="(\d+)"', attrs)          # a header spanning 3 columns counts 3 times
            row += [html.unescape(re.sub(r'<[^>]+>', '', c)).strip()] * (int(span.group(1)) if span else 1)
        rows.append(row)
    if not rows:
        return []
    header, out = rows[0], []
    for r in rows[1:]:
        if len(header) == len(r) and any(header):
            out.append(' | '.join(f'{h}: {c}' if h else c for h, c in zip(header, r)))
        else:
            out.append(' | '.join(r))
    return out

# =============================================================================
# Equations: symbols in the LaTeX
# =============================================================================
# Symbol rules (usual LaTeX conventions for printed models, nothing paper-specific):
#   [X] or [X]_{sub}   one concentration symbol: [GLC]_{in}, [H_2L_2]_{ER}
#   \greek_{sub}       one symbol: \mu_{d,max}
#   letter_{sub}       one symbol: K_{glc}, X_{v}
#   bare letters       one symbol each: adjacent letters are a product ('VX_v' is V and X_v)
#   digits, operators, brackets, \frac, \left, \right: not symbols
#   marks and accents  part of the name: f^{\prime}, DOT^{*}, \Theta^{\odot}, \dot{n}_{r}, \tilde{\Theta}
#   \sum_{i=1}^{N}     i is an index and N a count: neither is a symbol
#   words of 4+ letters one name: 'Flexibility', R_total
#   symbol = number unit  a value, not a law (notes say so)
# Names come back in one canonical LaTeX form: spaces removed, subscripts always braced
# ('X_v' -> 'X_{v}', 'K_{g l c}' -> 'K_{glc}'). Otherwise the reader's spelling is kept,
# so a misread like '[|H_{2}L]' stays visible.

GREEK = {'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', 'zeta', 'eta', 'theta',
         'vartheta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma', 'tau',
         'upsilon', 'phi', 'varphi', 'chi', 'psi', 'omega',
         'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Phi', 'Psi', 'Omega'}
LAYOUT = {'frac', 'dfrac', 'tfrac', 'left', 'right', 'cdot', 'times', 'mathrm', 'mathit', 'exp', 'ln', 'log',
          'boldsymbol', 'mathbf', 'bm', 'sqrt', 'mathcal', 'tag', 'erf', 'tanh', 'int',
          'bigl', 'bigr', 'Bigl', 'Bigr', 'biggl', 'biggr', 'Biggl', 'Biggr', ' ',
          'text', 'operatorname', 'big', 'Big', 'bigg', 'Bigg', 'quad', 'qquad', ',', ';', '!',
          'lg', 'limits', 'nolimits', 'displaystyle'}
ACCENTS = {'dot', 'ddot', 'tilde', 'widetilde', 'hat', 'widehat', 'bar', 'overline', 'vec'}   # part of the name: \dot{n} is not n
_MARKS = r"\*|\\ast|\\star|\\odot|\\otimes|\\oplus|\\circ|\\bullet|\\dagger|(?:\\prime)+|'+"
# Theta^{\otimes^{4}} or Theta^{\otimes 4}: the mark is the name, the 4 a power
_MARK = re.compile(r'\s*\^\s*\{\s*(' + _MARKS + r')\s*(?:\^?\s*\{?\s*\d+\s*\}?)?\s*\}|\s*\^\s*(' + _MARKS + r')|\s*(\'+)')


def clean(tex: str) -> str:
    """Remove the $$ delimiters and the punctuation that ends the printed line."""
    return tex.strip().strip('$').strip().rstrip('.,').strip()


def read_equations(formulas: list[dict], names: set[str] = frozenset()) -> list[dict]:
    """read_equation for every printed equation, then drop the independent variables (t, z, ...)
    from the symbol lists everywhere in the paper: they are domains, not Variables or Parameters."""
    reads = [read_equation(f['latex'], names) for f in formulas]
    _split_exponents(reads)
    indep = {w for r in reads for w in r['wrt']}
    for r in reads:
        r['lhs'] = [x for x in r['lhs'] if x not in indep or (x in r['states'] and x not in r['wrt'])]   # Ho: t(V)
        r['rhs'] = [x for x in r['rhs'] if x not in indep]
    return reads


def read_equation(tex: str, names: set[str] = frozenset()) -> dict:
    """{'latex', 'lhs', 'rhs', 'form', 'wrt', 'condition', 'notes'} for one printed equation.
    form: 'PDE' if it has a partial derivative, 'ODE' if it has an ordinary one, else 'algebraic'.
    wrt: the independent variables the derivatives are taken with respect to (t, z, r, ...)."""
    s = _normalise(clean(tex))
    condition = None
    w = re.search(r'[;,]\s*(?:w\s*i\s*t\s*h|w\s*h\s*e\s*r\s*e|f\s*o\s*r|i\s*f|w\s*h\s*e\s*n)(?![A-Za-z])', s)
    if w:                                                 # '...; with [S]* = ...': the rest is a side condition
        condition = s[w.end():].strip() or None
        s = s[:w.start()].strip()
    m = re.search(r',?\s*\\q?quad', s)                    # 'n > 1' printed after \quad
    if m and re.search(r'[<>]|\\(geq?|leq?|in|neq)\b', s[m.end():]):
        condition = ' '.join(x for x in (s[m.end():].strip().rstrip('.,').strip(), condition) if x) or None
        s = s[:m.start()].strip()

    s = _functions(s)
    k = _top_level_index(s, '=')
    approx = False
    if k < 0 and '\\approx' in s:                          # 'mu_X ~ 0.5 mu_X (1 - tanh(...))' (Goldrick 2015)
        s = s.replace('\\approx', '=', 1)
        k, approx = _top_level_index(s, '='), True
    if k < 0:                                             # a reaction scheme or an expression: no model law
        return {'latex': clean(tex), 'lhs': [], 'rhs': [], 'form': 'unknown', 'wrt': [], 'condition': condition,
                'notes': ['no = sign: not read as a model equation (e.g. a reaction scheme)']}
    lhs, rhs = s[:k].strip(), s[k + 1:].strip()
    lab = re.split(r';|\\colon|:(?!=)', lhs)              # 'M_I(theta, u, t); r = ...': a label before the law
    if len(lab) > 1 and lab[-1].strip():
        lhs = lab[-1].strip()
    lhs = re.sub(r'^\\q?quad\s*', '', lhs)
    fn = re.fullmatch(r'(.+?)\s*\(\s*([^()]*?)\s*\)', lhs)   # 'C^{*}(T) = ...': C* as a function of T
    if fn and not re.search(r'[-+*/=]|\\frac|\\cdot|\\times', re.sub(r'\^\{[^}]*\}|\^.', '', fn.group(1))) and fn.group(2) \
            and not re.search(r'\\(?:dot|frac)|d\s*/', lhs) \
            and not re.fullmatch(r'\s*\\?(?:lg|ln|log|log_\{?10\}?|exp|sqrt|sin|cos|tan)\s*', fn.group(1)):   # lg(K_P1) = ...: K_P1 stays
        lhs = fn.group(1)                                  # the argument is what it depends on, not computed
    value = _value(rhs)
    if value:                                             # 'Delta G_1 = 1.13 x 10^3 J mol^-1': a number, not a law
        return {'latex': clean(tex), 'lhs': symbols(lhs, names)[0], 'rhs': [], 'form': 'algebraic', 'wrt': [],
                'condition': condition, 'value': value, 'states': [], 'notes': [f'a value, not a law: {value}']}
    rhs, ivar = _strip_integral(rhs)
    lhs, d1, notes = _strip_derivatives(lhs)                # d(VX)/dt -> (VX): the state stays, d and dt go
    rhs, d2, n2 = _strip_derivatives(rhs)
    sens = [(op, w) for op, w in d2 if '_' in w]          # d Theta_i / d p_i on the right: a sensitivity, not an axis
    if sens:
        n2.append(f'derivative w.r.t. {", ".join(w for _, w in sens)} on the right: not read as an axis')
        d2 = [x for x in d2 if x not in sens]
    lhs_syms, n3 = symbols(lhs, names)
    rhs_syms, n4 = symbols(rhs, names)
    if len(lhs_syms) == 2:                                # 'A/A_0 = ...', 'R/R_0 = ...': X relative to its initial value
        x, x0 = lhs_syms
        if x0 in (f'{x}_{{0}}', f'{x}_0'):
            lhs_syms = [x]
            rhs_syms = rhs_syms + [x0] if x0 not in rhs_syms else rhs_syms
            n3 = n3 + [f'{x}/{x0} on the left: computes {x}, {x0} is its initial value']
    own = [x for x in lhs_syms if x != 't']              # x-dot(t) = ...
    dot = re.fullmatch(r'\\dot\{(.+?)\}(.*)', own[0]) if len(own) == 1 and not d1 else None
    if dot:                                               # y-dot_i = ...: Newton's notation for d y_i / dt
        lhs_syms, d1 = [dot.group(1) + dot.group(2)], [('d', 't')]
        notes.append(f'{dot.group(0)} on the left read as d/dt')
    ops = [op for op, _ in d1 + d2]
    out = {'latex': clean(tex), 'lhs': lhs_syms, 'rhs': rhs_syms,
           'form': 'PDE' if 'partial' in ops else 'ODE' if ops else 'algebraic',
           'wrt': sorted({w for _, w in d1 + d2} | ivar), 'condition': condition,
           'states': lhs_syms if d1 else [],             # differentiated on the left: kept even if an axis elsewhere
           'notes': notes + n2 + n3 + n4}
    if approx:
        out['notes'].append('printed with ~ (approximately equal), read as =')
    if not ops and len(lhs_syms) == 1 and lhs_syms[0] in rhs_syms:
        # 'mu_X = 0.5 mu_X (1 - tanh(...))': not an equation to solve but an update, applied in printed
        # order: the new mu_X is this expression of the mu_X before it (a factor stacked on a rate)
        out['updates'] = lhs_syms[0]
        out['notes'].append(f'updates {lhs_syms[0]} in place: the new value is this expression of the old one')
    return out


def _value(rhs: str) -> str | None:
    """'1.1348\\times10^{3} J mol^{-1}' -> the value as printed, if the right side is one number with
    an optional unit (letters, powers, '/'). Otherwise None."""
    m = re.fullmatch(r'\s*(-?\s*\d[\d.\s]*(?:\\times\s*10\s*\^\s*\{?\s*-?\d+\s*\}?)?)'
                     r'((?:\s*[A-Za-z]+(?:\s*\^\s*\{?\s*-?\d+\s*\}?)?|\s*/)*)\s*', rhs)
    return re.sub(r'\s+', ' ', m.group(0)).strip() if m else None


def _normalise(s: str) -> str:
    """Markup that changes the look, not the maths: \\mathrm{x} -> x (upright letters, subscripts,
    \\mathrm{d}), '~' spaces, and the line breaks and alignment marks of a multi-line equation
    (\\begin{aligned} ... \\\\ & ...), which is one equation printed over several lines."""
    s = s.replace('~', ' ').replace('\\_', '_').replace('\\triangle', '\\Delta')            # I_{avg\_R}: an escaped underscore is an underscore
    s = _drop_labels(s)
    # \upmu, \upalpha (upright Greek, 'upgreek'): the same letter as \mu, \alpha
    s = re.sub(r'\\up(alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Sigma|Phi|Psi|Omega)(?![A-Za-z])',
               r'\\\1', s)
    s = _subscript_words(s)
    # '\mathrm{e x p}', '\operatorname{e r f}': the OCR spaces out a function's letters; it is still the function
    s = re.sub(r'\\(?:mathrm|operatorname|rm|text)\s*\{\s*((?:[a-z]\s*){2,5})\}',
               lambda m: ('\\' + re.sub(r'\s', '', m.group(1)) + ' ') if re.sub(r'\s', '', m.group(1)) in _FUNCS else m.group(0), s)
    while (m := re.search(r'\\(?:mathrm|mathit|mathsf|rm|text|operatorname|boldsymbol|mathbf|bm)\s*(?=\{)', s)):
        body, j = _group(s, m.end())                      # nested braces too: \mathrm{avg_{R}}
        s = s[:m.start()] + ' ' + body + ' ' + s[j:]      # spaces: '\times\operatorname{erf}' must not become '\timeserf'
    s = re.sub(r'\\up([A-Za-z])(?![A-Za-z])', r'\1', s)  # \upS: an upright S is S
    # 'h(x,t)_{agg}': the subscript names the function, h_{agg}(x,t)
    s = re.sub(r'(\\?[A-Za-z]+)\s*\(([^()]*)\)\s*_\s*\{([^{}]*)\}', r'\1_{\3}(\2)', s)
    # 'd[X]/dt', 'dS/dt' written inline: the same derivative as \frac{d[X]}{dt}
    s = re.sub(r'(?<![A-Za-z\\])d\s*(\[[^\]]+\]|[A-Za-z](?:\s*_\s*\{[^}]*\})?)\s*/\s*d\s*t(?![A-Za-z])',
               lambda m: '\\frac{d ' + m.group(1) + '}{d t}', s)
    s = re.sub(r'\\(begin|end)\s*\{[^}]*\}(?:\s*\{[lcr|]+\})?', ' ', s)   # \begin{array}{cccc}: column spec too
    s = s.replace('\\\\', ' ').replace('&', ' ')
    return re.sub(r'\\q?quad(?=\s*\\times)', ' ', s)       # '\\quad\\times' continues the line


_FUNCS = {'exp', 'ln', 'log', 'lg', 'erf', 'erfc', 'tanh', 'sinh', 'cosh', 'sin', 'cos', 'tan', 'sign', 'sgn'}


def _subscript_words(s: str) -> str:
    """m_{g\\ln} -> m_{gln}: the OCR reads the letters 'ln' (gln, glutamine) as the log operator. Inside a
    subscript a function name is only letters. (\\max, \\min stay as they are: symbol names keep them.)"""
    out, i = [], 0
    for m in re.finditer(r'_\s*\{', s):
        if m.start() < i:
            continue
        body, j = _group(s, m.end() - 1)
        out.append(s[i:m.end() - 1] + '{' + re.sub(r'\\(ln|log|lg)(?![A-Za-z])', r'\1', body) + '}')
        i = j
    return ''.join(out) + s[i:]


def _drop_labels(s: str) -> str:
    """\\underbrace{F_{in}A/V}_{dilution} -> F_{in}A/V: the label under a term is a word, not symbols."""
    while (m := re.search(r'\\(underbrace|overbrace)\s*(?=\{)', s)):
        body, j = _group(s, m.end())
        lab = re.match(r'\s*[_^]\s*(?=\{)', s[j:])
        if lab:
            _, j = _group(s, j + lab.end())
        s = s[:m.start()] + '{' + body + '}' + s[j:]
    return s


def _strip_integral(s: str) -> tuple[str, set]:
    """\\int_0^{z_t} ... dl: the integration variable l is a domain, like t (dropped paper-wide);
    the 'd l' at the end goes. The limits stay (z_t is a quantity of the model)."""
    if '\\int' not in s:
        return s, set()
    m = re.search(r'(?<![A-Za-z\\])d\s*([A-Za-z])\s*\)?\s*$', s)
    return (s[:m.start()] + s[m.end(1):], {m.group(1)}) if m else (s, set())


def _functions(s: str) -> str:
    """Layout and function names written as plain letters, which would otherwise be read as symbols:
    \\begin{aligned}, 'ln(' or 'l n(' (= \\ln), 'exp(', 'log(', and e^{...} (= \\exp)."""
    s = re.sub(r'\\(begin|end)\s*\{[^}]*\}', ' ', s)
    s = re.sub(r'(?<![A-Za-z\\])(l\s*n|e\s*x\s*p|l\s*o\s*g|e\s*r\s*f|t\s*a\s*n\s*h)(?=\s*(\\left|\\big|\\Big)?\s*[(\[{])',
               lambda m: '\\' + re.sub(r'\s', '', m.group(1)) + ' ', s)
    return re.sub(r'(?<![A-Za-z\\])e(?=\s*\^)', r'\\exp ', s)


_OP = r'\s*(d|\\partial|\\mathrm\s*\{\s*d\s*\})\s*(?:\^\s*\{?\s*\d+\s*\}?)?\s*'   # d, \partial, \mathrm{d}, optional ^2


def _strip_derivatives(expr: str) -> tuple[str, list[tuple[str, str]], list[str]]:
    r"""Replace every \frac{d X}{d s} (also \partial, \mathrm{d}, d^2) by (X). Returns the new
    expression, [(op, s)] for each derivative found, and notes."""
    found, notes, out, i = [], [], '', 0
    for m in re.finditer(r'\\[dt]?frac\s*(?=\{)', expr):
        if m.start() < i:
            continue
        num, j = _group(expr, m.end())
        rest = expr[j:]
        if not rest.lstrip().startswith('{'):
            continue
        den, j2 = _group(rest, len(rest) - len(rest.lstrip()))
        dm = re.fullmatch(_OP + r'([A-Za-z]|\\[A-Za-z]+)(\s*_\s*(?:\{[^{}]*\}|[A-Za-z0-9]))?'
                          + r'\s*(?:\^\s*\{?\s*\d+\s*\}?)?\s*', den)          # dt, dV_{b} (bed volume), dz^2
        if not dm:
            continue
        nm = re.match(_OP + r'(?=[^_\s^])', num)          # 'd V', 'dm_H', 'd(VX_v)'; not 'd_p'
        if not nm:
            notes.append(f'\\frac{{...}}{{{den.strip()}}} but the numerator has no d: derivative marker missing?')
            continue
        op = 'partial' if 'partial' in dm.group(1) else 'd'
        found.append((op, symbols(dm.group(2) + (dm.group(3) or ''))[0][0]))
        out += expr[i:m.start()] + '(' + num[nm.end():] + ')'
        i = j + j2
    return out + expr[i:], found, notes


def symbols(expr: str, names: set[str] = frozenset()) -> tuple[list[str], list[str]]:
    """Symbols in a LaTeX expression in order of first appearance, and notes on anything skipped.
    names: multi-letter names to read as one symbol (spaces between the letters allowed: 'D O T')."""
    found, notes, i, s, index = [], [], 0, expr, set()
    runs = [(n, re.compile(r'\s*'.join(n) + r'(?![A-Za-z])')) for n in sorted(names, key=len, reverse=True)]
    while i < len(s):
        c = s[i]
        k = s.find(']', i) if c == '[' else -1
        if c == '[' and k > 0 and re.fullmatch(r'[A-Za-z0-9_{}\s|,]+', _CHARGE.sub('', s[i + 1:k])) \
                and not re.search(r'\\(left|right|[bB]igg?)\s*$', s[:i]):
            # concentration [X] with optional subscript; a bracket around an expression is just a bracket
            name = '[' + _compact(s[i + 1:k]) + ']'
            sub, i = _subscript(s, k + 1)
            found.append(name + sub)
        elif c == '\\':
            m = re.match(r'\\([A-Za-z]+|.)', s[i:])
            cmd, i = m.group(1), i + m.end()
            if cmd in ('sum', 'prod'):                     # \sum_{i=1}^{N_K}: i is an index, N_K a count
                i = _skip_limits(s, i, index, notes, cmd)
            elif cmd in ACCENTS:
                m2 = re.match(r'\s*', s[i:]); i += m2.end()
                body, i = _group(s, i) if s[i:i + 1] == '{' else (s[i:i + 1], i + 1)
                pre, i = _marks(s, i)
                sub, i = _subscript(s, i)
                post, i = _marks(s, i)
                found.append('\\' + cmd + '{' + _compact(body) + '}' + sub + _join_marks(pre, post))
            elif cmd in GREEK:
                nxt = re.match(r'\s*([A-Za-z])(?![a-z])', s[i:]) if cmd == 'Delta' else None
                if nxt:                                   # \Delta H_{b}, \Delta t: one symbol
                    cmd, i = cmd + ' ' + nxt.group(1), i + nxt.end()
                pre, i = _marks(s, i)
                sub, i = _subscript(s, i)
                post, i = _marks(s, i)
                sup, i = _label_sup(s, i, sub) if not (pre or post) else ('', i)
                found.append('\\' + cmd + sub + _join_marks(pre, post) + sup)
            elif cmd not in LAYOUT:
                notes.append(f'unknown command \\{cmd} skipped')
        elif c.isalpha():
            m = next(((n, r.match(s, i)) for n, r in runs if r.match(s, i)), None)
            name, end = (m[0], m[1].end()) if m else (c, i + 1)
            word = None if m else re.match(r'[A-Za-z]{4,}(?![A-Za-z])', s[i:])
            if word:                                       # 'Flexibility', 'actual': a word, not a product of letters
                name, end = word.group(0), i + word.end()
                notes.append(f'{name} read as one word')
            pre = re.match(r'\s*\^\s*\{\s*([A-Za-z]+)\s*\}(?=\s*_)', s[end:])   # C^{p}_{L} = (C_L)^p
            if pre:
                end += pre.end()
            mk1, end = _marks(s, end) if not pre else ('', end)
            sub, i = _subscript(s, end)
            mk2, i = _marks(s, i)                         # DOT^* (saturation) is not DOT; f' is not f
            star = _join_marks(mk1, mk2)
            sup = '^{' + pre.group(1) + '}' if pre else ''
            if not sup and not star:
                sup, i = _label_sup(s, i, sub)
            found.append(name + sub + star + sup)
        else:
            i += 1
    found = [x for x in found if x not in index]
    return list(dict.fromkeys(found)), notes


def _marks(s: str, i: int) -> tuple[str, int]:
    """A mark printed as a superscript that is part of the name: f' (not f), DOT^* (saturation),
    Theta^odot (coverage of oxidised sites, Seidel). ('\\prime', index after it) or ('', i)."""
    m = _MARK.match(s, i)
    if not m:
        return '', i
    mk = m.group(1) or m.group(2) or m.group(3)
    mk = '*' if mk == '\\ast' else '\\prime' * len(mk) if mk.startswith("'") else mk
    return mk, m.end()


def _join_marks(a: str, b: str) -> str:
    return '^{' + a + b + '}' if a or b else ''


def _skip_limits(s: str, i: int, index: set, notes: list, cmd: str) -> int:
    """After \\sum or \\prod: skip \\limits and the limits; the lower limit's letter is an index."""
    i += re.match(r'\s*(?:\\n?o?limits)?\s*', s[i:]).end()
    lims = {}
    for _ in range(2):
        m = re.match(r'\s*([_^])\s*', s[i:])
        if not m:
            break
        j = i + m.end()
        body, i = _group(s, j) if s[j:j + 1] == '{' else (s[j:j + 1], j + 1)
        lims[m.group(1)] = body.strip()
    ix = re.match(r'([A-Za-z])\s*(=|$)', lims.get('_', ''))
    if ix:
        index.add(ix.group(1))
    notes.append(f"\\{cmd} over {lims.get('_', '?')}..{lims.get('^', '?')}: index and limits not read as symbols")
    return i


def _label_sup(s: str, i: int, sub: str) -> tuple[str, int]:
    """A letter superscript on a subscripted symbol, \\mu_{m}^{A}: kept for now as '^{A}'. Whether it
    is part of the name or an exponent is decided in read_equations, which sees the whole paper."""
    m = re.match(r'\s*\^\s*\{\s*([A-Za-z]+)\s*\}|\s*\^\s*([A-Za-z])(?![A-Za-z])', s[i:]) if sub else None
    return ('^{' + (m.group(1) or m.group(2)) + '}', i + m.end()) if m else ('', i)


def _split_exponents(reads: list[dict]) -> None:
    """'X_{s}^{a}' is a name if 'a' is something the model computes (\\mu_{m}^{A}: growth on
    ammonium A, Nausch); otherwise 'a' is an exponent and both are symbols (C_{L}^{p}: C_L to the
    power p, Birol). Nobody raises a rate to the power of a concentration."""
    computed = {x for r in reads for x in r['lhs']}
    for r in reads:
        for side in ('lhs', 'rhs'):
            out = []
            for x in r[side]:
                m = re.fullmatch(r'(.+)\^\{([A-Za-z]+)\}', x)
                if m and m.group(2) not in computed and x not in computed:
                    out += [m.group(1)] + list(m.group(2))
                else:
                    out.append(x)
            r[side] = list(dict.fromkeys(out))


_CHARGE = re.compile(r'\^\s*\{?\s*\d*\s*[+\-]\s*\}?')        # the + in [H^{+}]


def _group(s: str, i: int) -> tuple[str, int]:
    """s[i] == '{': (content, index after the matching '}')."""
    depth = 0
    for j in range(i, len(s)):
        depth += {'{': 1, '}': -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
    return s[i + 1:], len(s)


def _subscript(s: str, i: int) -> tuple[str, int]:
    """If a subscript starts at s[i] (spaces allowed): ('_{...}', index after it); else ('', i)."""
    m = re.match(r'\s*_\s*', s[i:])
    if not m:
        return '', i
    j = i + m.end()
    if j < len(s) and s[j] == '{':
        body, j = _group(s, j)
    elif j < len(s) and s[j] == '\\':
        cmd = re.match(r'\\[A-Za-z]+', s[j:]).group(0)
        body, j = cmd, j + len(cmd)
    else:
        w = re.match(r'[A-Za-z]{3,}(?![A-Za-z])', s[j:])     # R_total as printed: the whole word
        body, j = (w.group(0), j + w.end()) if w else (s[j:j + 1], j + 1)
    return '_{' + _compact(body) + '}', j


def _compact(s: str) -> str:
    """'g l c' -> 'glc'; '{a m m}' -> 'amm'; 'H_{2}L_{2}' -> 'H_2L_2'."""
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'_\{([^{}])\}', r'_\1', s)
    return re.sub(r'(?<!_)\{([^{}]*)\}', r'\1', s)


def _top_level_index(s: str, ch: str) -> int:
    """Index of the first ch outside any brackets, or -1."""
    depth = 0
    for k, c in enumerate(s):
        if c in '{([':
            depth += 1
        elif c in '})]':
            depth -= 1
        elif c == ch and depth == 0:
            return k
    return -1

# =============================================================================
# Symbol mentions in text, and table values
# =============================================================================
# Symbols and table text are both reduced to a KEY so different notations compare:
# 'K_{d,amm}' and 'K_{d, amm}' -> 'K_damm'; '[GLC]_{in}' -> 'C_GLCin'; '\varepsilon_1' -> 'epsilon_1'.

GREEK_ALIASES = {'varepsilon': 'epsilon', 'lambda': 'lam', 'nu': 'v', 'varphi': 'phi'}
CONFUSABLE = str.maketrans({'I': 'l', '1': 'l', '|': 'l', 'O': '0', 'o': '0'})   # OCR look-alikes
LEVELS = ['exact', 'notation', 'confusable', 'partial']                          # best first
_TOKEN = re.compile(r'\[([^\]]+)\](?:_\{([^}]*)\}|_([A-Za-z0-9]))?'              # [GLC]_{in}
                    r'|(\\?[A-Za-z]+)(?:_\{([^}]*)\}|_([A-Za-z0-9]))?')          # K_{d,amm}, \mu_{max}


def mention(symbol: str, text: str) -> str | None:
    """Best match level of the symbol in the text ('exact', 'notation', 'confusable', 'partial'), or None."""
    k = _keys(symbol)
    k = next(iter(k)) if len(k) == 1 else re.sub(r'[^A-Za-z0-9_]', '', symbol)
    levels = [lv for t in _keys(text) if (lv := _match_level(k, t))]
    return min(levels, key=LEVELS.index) if levels else None


def table_value(symbol: str, snippets: list[dict]) -> tuple[str, str] | None:
    """(row text, match level) for the first table row with a number that mentions the symbol, or None."""
    for s in snippets:
        if s['kind'] == 'table_row' and re.search(r'\|[^|]*\d', s['text']) and (lv := mention(symbol, s['text'])):
            return f"p.{s['page'] + 1}: {s['text']}", lv
    return None


def _keys(text: str) -> set[str]:
    """Symbol-like tokens in a piece of text (a table row, or one symbol), as keys."""
    s = re.sub(r'([A-Za-z])\s*\$\s*(_\{[^}]*\})\s*\$', r'\1\2', text)    # 'K $ _{A} $' -> 'K_{A}'
    s = s.replace('$', ' ')
    s = re.sub(r'_\{\s*([^}]*?)\s*\}', lambda m: '_{' + m.group(1).replace(' ', '') + '}', s)
    alnum = lambda x: re.sub(r'[^A-Za-z0-9]', '', x)
    out = set()
    for m in _TOKEN.finditer(s):
        if m.group(1) is not None:
            out.add('C_' + alnum(m.group(1)) + alnum(m.group(2) or m.group(3) or ''))
        else:
            base = m.group(4).lstrip('\\')
            base = GREEK_ALIASES.get(base, base)
            sub = alnum(m.group(5) or m.group(6) or '')
            out.add(base + ('_' + sub if sub else ''))
    return out


def _match_level(key: str, tok: str) -> str | None:
    if tok == key:
        return 'exact'
    a, b = key.replace('_', ''), tok.replace('_', '')
    if len(a) < 3:
        return None
    if a == b:
        return 'notation'       # K_Ilac vs KI_lac: same letters, split differently
    kb, _, ks = key.partition('_')
    tb, _, ts = tok.partition('_')
    if kb == tb and ks and ks.lower() == ts.lower():
        return 'notation'       # Y_{X_V/glc} vs Y_{Xv/glc}: same subscript, other case
    if a.translate(CONFUSABLE) == b.translate(CONFUSABLE):
        return 'confusable'     # OCR I/l confusion
    if len(b) == len(a) + 1 and b.translate(CONFUSABLE).endswith(a.translate(CONFUSABLE)):
        return 'partial'        # equation has I_lac, table has KI_lac: one name split in two?
    return None


READ_LABELS = {'doc_title', 'abstract', 'paragraph_title', 'text', 'figure_title', 'table_title', 'footnote',
               'vision_footnote', 'aside_text', 'algorithm'}   # not: references, headers, footers, page numbers, images
MAX_PAPER_CHARS = 120_000                                      # longest paper so far ~75k with references
PAPER_NOTE = ('\n\nThe whole paper comes first in the message, for context: look there for definitions, units, '
              'values and anything else the task needs. The material for this task follows after the line =====.')


def paper_text(paper_dir: str, references: bool = False) -> str:
    """The whole paper in reading order: text, titles, captions, footnotes and tables (as rows). Every LLM
    step gets it first, so the calls of one step share the same prefix (the provider caches it)."""
    blocks = json.load(open(f'{paper_dir}/paddle_blocks.json'))['blocks']
    out = []
    for n, b in enumerate(blocks):
        text = (b.get('content') or '').strip()
        if not text:
            continue
        if b['label'] == 'table':
            cap = _caption(blocks, n)
            out.append((f'[{cap}]\n' if cap else '[table]\n') + '\n'.join(_table_rows(text)))
        elif b['label'] in READ_LABELS or references and b['label'] == 'reference_content':
            out.append(('## ' if b['label'] == 'paragraph_title' else '') + text)
    return '\n\n'.join(out)[:MAX_PAPER_CHARS]


def with_paper(paper_dir: str, msg: str) -> str:
    return f'Paper:\n\n{paper_text(paper_dir)}\n\n=====\n\n{msg}'
