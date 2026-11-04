# Can SymPy's own LaTeX parser read a printed process model?

Test case: the 28 equations of the Kontoravdi et al. CHO model (2010 paper, Appendix A).
Script: `studies/sympy_parser_eval.py`. Output: `papers/kontoravdi_2010/sympy_parser_eval.json`.
SymPy 1.14.0, `sympy.parsing.latex.parse_latex`, both backends (`antlr`, `lark`), used as shipped.

## Setup

- **Input LaTeX**, two versions of each equation:
  - *printed*: the hand-checked transcription in `gold_laws.json`.
  - *reader*: what PaddleOCR-VL produced (`paddle.json`).

  The only preprocessing is removing the `$$` delimiters and the punctuation that ends the line.
- **Reference**: `sympy_reference` in `gold_laws.json`, the printed equation as a checked SymPy expression. These references were produced once by the rule-based converter (now archived), then frozen as data.
- **Fair to the parser**:
  - Symbol *names* are normalised before comparing. SymPy writes `F_out` as `F_{o*(t*u)}`, which is ugly but still one consistent symbol, so it is not counted as an error.
  - `[H]` named `H` instead of `C_H` is not counted either.
  - Derivatives are frozen before simplifying. Otherwise `simplify` evaluates d(symbol)/dt to 0 and any two ODEs look equal.
- **Verdicts**:
  - `fail`: the parser raised an error.
  - `wrong`: it parsed, but into a different mathematical expression.
  - `correct`: same expression after name normalisation.

## Result

| LaTeX source | backend | correct | wrong (silent) | fail (error) |
|---|---|---:|---:|---:|
| printed | antlr | 7 | 12 | 9 |
| printed | lark | 1 | 5 | 22 |
| reader (Paddle) | antlr | 8 | 10 | 10 |
| reader (Paddle) | lark | 0 | 2 | 26 |

At best, 7 or 8 of 28 are right. The worse problem is the **silent** column: 10–12 equations come back as valid SymPy objects with the wrong mathematics, and nothing warns you.

The reader's LaTeX scores one point better than the printed version on antlr. That is not because it is more accurate. Paddle writes `X_{\nu}` where the printed form has `X_v`, and that happens to parse.

## What goes wrong (printed LaTeX, antlr)

| eq | verdict | what SymPy did |
|---:|---|---|
| 1 | correct | |
| 2 | wrong | `\mu_d V X_v` → one symbol `mu_{dV}` times `X_v`: the subscript swallowed the V |
| 3, 4 | correct | |
| 5 | wrong | `[GLC]` → `G·C·L`, a product of three symbols |
| 6 | wrong | `[LAC]`, `[AMM]` → products of letters |
| 7 | fail | comma in a subscript (`\mu_{d,max}`) |
| 8 | wrong | `[GLC]` → `G·C·L`; the term `−F_out[GLC]` is **dropped**; `[GLC]_in` is lost |
| 9 | fail | comma in a subscript (`Y_{x,glc}`) |
| 10 | wrong | as eq 8: `−F_out[GLN]` dropped, `[GLN]_in` lost |
| 11 | fail | comma in a subscript |
| 12, 13 | wrong | concentrations → products of letters |
| 14 | fail | comma in a subscript |
| 15 | wrong | concentrations → products of letters |
| 16 | fail | comma in a subscript |
| 17, 18 | fail | `\frac{dm_H}{dt}` not read as a derivative |
| 19–21 | correct | single-letter concentrations `[H]`, `[L]` happen to survive |
| 22 | wrong | `[H_2L]` → `H_2·L`: the species H₂L becomes the product of two *other* species, [H₂]·[L] |
| 23 | correct | |
| 24 | wrong | same `[H_2L]` collapse |
| 25, 26 | fail | subscript on a bracket, `[H_2L_2]_{ER}` |
| 27 | wrong | `[MAb]` → `M·A·b` |
| 28 | wrong | `[H_2L_2]_G` → `H_2·L_2`: the compartment subscript G is dropped |

The `lark` backend is worse. It fails on any `[` and on most subscripts. Where it does not fail, it can evaluate the whole equation to the Python value `False` (eqs 1, 17, 18) instead of returning an equation.

## Why this matters for the thesis

- **Notation is the problem, not the parser.** The parser follows general-maths conventions: juxtaposed letters are a product, and `[ ]` are brackets. Process models are printed with domain conventions: `[GLC]` is one concentration, `Y_{x,glc}` is one yield, `[H_2L_2]_{ER}` is one species in one compartment. The failures cluster exactly there.
- **Silent errors are the dangerous ones.** A dropped outflow term (eq 8) or a species turned into a product of two others (eq 22) gives a model that still solves. This is the "converges cleanly, confidently wrong" failure the verification layers are meant to catch.
- **Options for producing `mkg:formula` later (open):**
  - A rule-based converter. Built and archived: 28/28 here, but the rules were written for this one paper.
  - An LLM translation, checked by code. Possible checks:
    - The symbol set must equal the one `latex_symbols.py` lists.
    - Every concentration stays one symbol.
    - The number of terms matches.
  - A combination of the two.

  For now (v0) the graph stores LaTeX only (`mkg:latex`), and `mkg:formula` is left empty.
