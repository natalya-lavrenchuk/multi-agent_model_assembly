"""
pass2.py -- extract equations (formula, computes, uses) from one document.

PIPELINE
    LaTeX (from the IMAGE, via pix2tex)  +  Pass 1 symbol table (voted)
      -> LLM (strict JSON)      SymPy string, symbol bindings, law name
      -> gate (deterministic)   parses? symbols bind? structure matches the
                                PDF? dimensionally consistent?
      -> accepted / flagged
      -> Turtle

THE LLM'S JOB HERE IS BINDING, NOT READING
    The equation already exists: pix2tex read it off the page. The model
    converts that LaTeX into a SymPy-parseable string and maps each symbol to
    a row in the Pass 1 table. It is NOT asked what the equation is, because
    for a benchmark case like Monod it already knows, and a recalled equation
    is indistinguishable from a read one in the output. Here the LaTeX sits in
    the prompt and every gate compares against it.

WHAT PASS 2 CATCHES THAT PASS 1 CANNOT
    Pass 1's quantity_class is an unanchored guess: class_stated is false on
    essentially every row, so the document never said how any quantity is
    obtained. Structure is the anchor it never had.

      Ks in 'Ks + S'    added to a concentration -> a concentration
      Ks in 'Ks*X + S'  multiplied by biomass    -> DIMENSIONLESS

    Same printed symbol, incompatible dimensions, derived rather than
    recalled. Pass 1 returned one Ks and no Pass 1 gate could catch it:
    symbol_conflict needs two parseable units, and every unit on a prose
    paper is null.

    Likewise mu_max and Ks classed 'Constant': fixed for the run, computed by
    no equation, appearing inside a rate law -> Parameter by structure.

NO UNITS ARE REQUIRED
    Units are case-specific (g/L or mol/m3 is the operator's choice).
    DIMENSIONS are forced by the formulas. So Pass 2 DERIVES dimensional
    relations from structure and uses Pass 1's units only as a cross-check
    where they exist. On a prose paper they never do.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field

import sympy

from harness import Harness, log_call, tier_report
from mkg_ids import (PREFIXES, check_formula_names, doc_iri, formula_name,
                     law_iri, quantity_iri, split_doc_id)


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------

PASS2_SCHEMA = {
    "type": "object",
    "properties": {
        "laws": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "local_id": {
                        "type": "string",
                        "description": "Short unique id within this document, e.g. L1."
                    },
                    "law_name": {
                        "type": ["string", "null"],
                        "description": (
                            "The name the DOCUMENT gives this law, e.g. 'Monod'. "
                            "Use null if the surrounding text does not name it. "
                            "An unnamed law is a correct answer."
                        )
                    },
                    "law_name_span": {
                        "type": ["string", "null"],
                        "description": (
                            "The sentence naming the law, copied WORD FOR WORD from the "
                            "surrounding text. Required when law_name_source is 'prose'. "
                            "Null otherwise."
                        )
                    },
                    "law_name_source": {
                        "type": ["string", "null"],
                        "enum": ["prose", "section", None],
                        "description": (
                            "'prose' when a sentence in the surrounding text names it, and "
                            "law_name_span quotes that sentence. 'section' when only the "
                            "section heading names it, which is weaker evidence because a "
                            "heading is not a statement about this particular equation. "
                            "Null when neither names it."
                        )
                    },
                    "sympy_expr": {
                        "type": "string",
                        "description": (
                            "The equation as a SymPy expression string, e.g. "
                            "'Eq(mu, mu_max*S/(K_s + S))'. Use ASCII identifiers only. "
                            "Transcribe the LaTeX you were given. Do NOT correct it, "
                            "simplify it, or substitute an equation you recognise."
                        )
                    },
                    "symbol_map": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sympy_name": {
                                    "type": "string",
                                    "description": "The ASCII name used in sympy_expr, e.g. 'K_s'."
                                },
                                "latex_symbol": {
                                    "type": "string",
                                    "description": "The symbol as it appears in the LaTeX, e.g. '\\\\mu_{max}'."
                                },
                                "pass1_symbol": {
                                    "type": ["string", "null"],
                                    "description": (
                                        "The symbol EXACTLY as it appears in the quantity table "
                                        "you were given, or null if no row matches. Null is the "
                                        "correct answer when the OCR produced a symbol that is "
                                        "not in the table; do not force a match."
                                    )
                                }
                            },
                            "required": ["sympy_name", "latex_symbol", "pass1_symbol"],
                            "additionalProperties": False
                        }
                    },
                    "computes": {
                        "type": ["string", "null"],
                        "description": (
                            "The sympy_name of the quantity this law determines, normally "
                            "the left-hand side. Null if the equation is not solved for one "
                            "quantity."
                        )
                    },
                    "law_kind": {
                        "type": "string",
                        "enum": ["RateLaw", "BalanceLaw", "ClosureRelation", "Definition"],
                        "description": (
                            "RateLaw: an empirical constitutive expression for a rate, specific "
                            "to a material or organism and carrying fitted coefficients. "
                            "BalanceLaw: conservation over a control volume, accumulation = in "
                            "- out + generation. Carries no fitted coefficients. "
                            "ClosureRelation: an auxiliary constitutive relation such as an "
                            "isotherm or an equilibrium expression. "
                            "Definition: defines a quantity in terms of measurements, e.g. a "
                            "specific growth rate from two biomass readings."
                        )
                    },
                    "kind_stated": {
                        "type": "boolean",
                        "description": (
                            "True only if the surrounding text says what kind of relation this "
                            "is. False if you inferred it. Most papers will be false."
                        )
                    }
                },
                "required": ["local_id", "law_name", "law_name_span",
                             "law_name_source", "sympy_expr", "symbol_map",
                             "computes", "law_kind", "kind_stated"],
                "additionalProperties": False
            }
        }
    },
    "required": ["laws"],
    "additionalProperties": False
}


SYSTEM_PROMPT = """\
You convert an equation that has ALREADY BEEN READ from a paper into a SymPy \
expression, and bind its symbols to a table of quantities extracted from the \
same paper.

You are NOT being asked what the equation is. It is given to you as LaTeX, \
produced by optical character recognition of the printed page. Your job is \
transcription and binding, not recall.

Rules:

1. Transcribe the LaTeX you are given. Do not correct it, do not simplify it, \
do not substitute a well-known equation that it resembles. If the LaTeX says \
K.X then write K_dot_X and bind it to null. A faithful transcription of a \
flawed reading is useful; a corrected equation is not, because it hides \
whether the reading worked.

2. Use ASCII identifiers in sympy_expr: mu, mu_max, K_s, S, X. Write \
multiplication explicitly with *. Write the whole equation as Eq(lhs, rhs).

3. For every distinct symbol in sympy_expr add one symbol_map entry. \
pass1_symbol must be copied EXACTLY from the quantity table given to you, \
character for character. If no row matches, use null. Do not force a match \
and do not invent a row.

4. law_name comes from the SURROUNDING TEXT, not from your own knowledge of \
the equation. If a sentence names it, set law_name_source to "prose" and quote \
that sentence word for word in law_name_span. If only the section heading names \
it, set law_name_source to "section" and law_name_span to null. If neither names \
it, set all three to null. An unnamed law is a correct answer.

5. Set kind_stated to true ONLY if the surrounding text says what kind of \
relation this is. Choosing law_kind yourself is expected and fine, but then \
kind_stated is false.

Return JSON only.\
"""


USER_TEMPLATE = """\
Document: {doc_id}
Section: {section}

QUANTITY TABLE extracted from this paper. Bind to these symbols exactly.
{table}

EQUATION, read from the printed page by OCR:
{latex}

SURROUNDING TEXT from the paper:
{context}

Convert the equation to SymPy and bind every symbol.\
"""


# ----------------------------------------------------------------------
# Dimensional derivation
# ----------------------------------------------------------------------

class DimensionError(Exception):
    pass


# Functions a mechanistic model may legitimately contain. Everything else in
# the SymPy namespace is deliberately hidden -- see safe_sympify.
_ALLOWED_FUNCS = {
    "exp": sympy.exp, "log": sympy.log, "ln": sympy.log, "sqrt": sympy.sqrt,
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan, "tanh": sympy.tanh,
    "Abs": sympy.Abs, "Eq": sympy.Eq, "Derivative": sympy.Derivative,
    # the auto_number transformation rewrites literals as Integer(2) etc,
    # so the constructors must be visible or every exponent fails to parse
    "Integer": sympy.Integer, "Float": sympy.Float,
    "Rational": sympy.Rational, "Symbol": sympy.Symbol,
}


def safe_sympify(expr_str: str, names):
    """Parse an expression with EVERY bare name forced to a plain Symbol.

    sympy.sympify() evaluates against the whole SymPy namespace, where a
    number of single letters are already taken:

        S  -> SingletonRegistry   (substrate concentration, in every equation
                                   in a bioprocess paper)
        E  -> exp(1)
        I  -> the imaginary unit
        N, O, Q, beta, gamma, zeta, lambda ...

    So the obvious call silently mis-parses or raises on exactly the symbols a
    chemical engineering model uses most. Restricting global_dict to a short
    list of functions makes every other name resolve to a Symbol.
    """
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations

    local = {n: sympy.Symbol(n) for n in names}
    return parse_expr(expr_str, local_dict=local,
                      global_dict=dict(_ALLOWED_FUNCS),
                      transformations=standard_transformations,
                      evaluate=True)


def _dim_symbol(name: str) -> sympy.Symbol:
    return sympy.Symbol(f"dim_{name}")


def derive_dimensions(expr) -> tuple[list, dict, list]:
    """Linear constraints on the dimensions of every symbol in an expression.

    Each quantity's dimension is an unknown exponent vector over the base
    dimensions (M, L, T, N, Theta). The constraints generated below are
    IDENTICAL in every component, so one scalar unknown per symbol produces
    exactly the same system as five vector unknowns would, and an
    inconsistency in the scalar system is an inconsistency in the real one.

    Rules, straight from what the operators mean:
      Add   every term must have the same dimension
      Mul   dimensions add
      Pow   integer or rational exponent: the dimension multiplies.
            SYMBOLIC exponent (Moser's S**n): the EXPONENT must be
            dimensionless, but the base need not be. The result's dimension is
            n*dim(base), which is nonlinear, so a fresh unknown is introduced
            and the check is recorded as PARTIAL rather than forcing
            dimensionlessness. Forcing it would flag every empirical power law
            in the paper as broken.
      exp/log/trig   the argument must be dimensionless
      a bare number is dimensionless

    Returns (equations, symbol->unknown map, partial_reasons). Nothing is
    solved here.
    """
    eqs = []
    dmap: dict[str, sympy.Symbol] = {}
    partial: list[str] = []
    counter = [0]

    def fresh():
        counter[0] += 1
        return sympy.Symbol(f"dim_pow{counter[0]}")

    def dim(e):
        if e.is_Number:
            return sympy.Integer(0)
        if e.is_Symbol:
            return dmap.setdefault(e.name, _dim_symbol(e.name))
        if e.is_Add:
            ds = [dim(a) for a in e.args]
            for other in ds[1:]:
                eqs.append(sympy.Eq(ds[0], other))
            return ds[0]
        if e.is_Mul:
            return sum(dim(a) for a in e.args)
        if e.is_Pow:
            base, exp_ = e.args
            if exp_.is_Integer:
                return dim(base) * int(exp_)
            if exp_.is_Rational:
                return dim(base) * sympy.Rational(exp_)
            eqs.append(sympy.Eq(dim(exp_), 0))     # exponents are dimensionless
            partial.append(f"symbolic exponent in {e}")
            return fresh()
        if isinstance(e, sympy.Function):
            for a in e.args:
                eqs.append(sympy.Eq(dim(a), 0))    # transcendental -> dimensionless
            return sympy.Integer(0)
        if e.is_Equality:
            l, r = dim(e.lhs), dim(e.rhs)
            eqs.append(sympy.Eq(l, r))
            return l
        raise DimensionError(f"unhandled node {type(e).__name__}: {e}")

    dim(expr)
    return eqs, dmap, partial


def solve_dimensions(eqs, dmap, anchors: dict[str, int] | None = None):
    """Solve the dimensional system. Returns (consistent, relations).

    anchors pins known dimensions so relations resolve to numbers rather than
    to each other. For a bioprocess the natural anchors are the concentrations:
    with S and X both at 1, Contois's Ks falls out as 0, i.e. dimensionless,
    while Monod's stays at 1.
    """
    system = list(eqs)
    for name, value in (anchors or {}).items():
        if name in dmap:
            system.append(sympy.Eq(dmap[name], value))

    unknowns = sorted(dmap.values(), key=lambda s: s.name)
    if not system:
        return True, {}

    sol = sympy.solve(system, unknowns, dict=True)
    if not sol:
        return False, {}

    s = sol[0]
    rel = {}
    for name, d in dmap.items():
        v = s.get(d, d)
        rel[name] = v
    return True, rel


# ----------------------------------------------------------------------
# Gate
# ----------------------------------------------------------------------

@dataclass
class GatedLaw:
    record: dict
    expr: object | None = None
    status: str = "flagged"
    failed: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


def _norm(s: str) -> str:
    """NFKC then whitespace-collapse, same as Pass 1.

    PDFs emit ligatures (U+FB01 in 'specific') and the micro sign where Greek
    mu belongs. Without NFKC a span that IS verbatim fails on a character
    difference no reader would see.
    """
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip().lower()


_LATEX_CMD = re.compile(r"\\[A-Za-z]+")
_LITERAL_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


_GREEK_LATEX = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "omega": "ω",
}
# longest first, so \lambda is not read as \l + ambda
_GREEK_CMD = re.compile(r"\\(" + "|".join(sorted(_GREEK_LATEX, key=len,
                                                  reverse=True)) + r")")
_GREEK_ASCII_HEAD = re.compile(r"^(" + "|".join(sorted(_GREEK_LATEX, key=len,
                                                       reverse=True)) + r")(?=_|$)")
_FONT_CMD = re.compile(r"\\(mathrm|mathit|mathbf|text|operatorname|rm|it)\b")


def normalize_symbol(s: str | None) -> str:
    """One canonical spelling for a symbol written in any of its encodings.

    The same quantity reaches Pass 2 spelled four ways:
        'Ks'           Pass 1 table, copied from the paper's prose
        'K_s'          LaTeX subscript, from the OCR
        'mu_max'       the ASCII SymPy name the model chose
        '\\mu_{max}'   LaTeX, or '\\mumax' with the braces lost, or the
                       literal escape '\\u03bcmax'
    These are one symbol in different encodings, which makes matching them a
    normalisation, like NFKC for ligatures -- not a judgement for the LLM.
    Leaving it to the LLM produced a stochastic binding: K_s -> Ks in one run,
    K_s -> null in the next.

    CASE IS PRESERVED on purpose. 'S' and 's' are different symbols in these
    models, and an OCR that reads uppercase S as lowercase s must stay
    unbound so the gate catches it.
    """
    if not s:
        return ""
    s, _ = decode_literal_escapes(s)
    s = _FONT_CMD.sub("", s)
    s = _GREEK_CMD.sub(lambda m: _GREEK_LATEX[m.group(1)], s)
    s = _GREEK_ASCII_HEAD.sub(lambda m: _GREEK_LATEX[m.group(1)], s)
    s = re.sub(r"[\\{}_\s]", "", s)
    return unicodedata.normalize("NFKC", s)


def build_symbol_index(pass1_symbols) -> dict[str, str]:
    """normalized -> table symbol, dropping any spelling two rows share.

    An ambiguous normalisation must not auto-bind: if two table rows collapse
    to the same canonical form, binding to either would be a guess.
    """
    seen: dict[str, list[str]] = {}
    for sym in pass1_symbols:
        seen.setdefault(normalize_symbol(sym), []).append(sym)
    return {k: v[0] for k, v in seen.items() if len(v) == 1 and k}


def decode_literal_escapes(s: str) -> tuple[str, bool]:
    """Turn a literal '\\u03bc' back into the character it names.

    Under strict JSON decoding a model sometimes emits the six characters
    backslash-u-0-3-b-c instead of the character mu. It meant the character;
    it wrote a different encoding of it. This is the same situation as the
    U+FB01 ligature that made verbatim spans fail in Pass 1, and the same
    answer: make two strings that differ only in encoding compare equal.

    Returns (decoded, changed) so the substitution is recorded rather than
    applied silently.
    """
    if not s or "\\u" not in s:
        return s, False
    out = _LITERAL_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), s)
    return out, out != s


def latex_chars(tex: str) -> str:
    """Characters a reader would see, for comparison against PDF glyphs."""
    greek = {r"\mu": "μ", r"\alpha": "α", r"\beta": "β", r"\lambda": "λ",
             r"\theta": "θ", r"\rho": "ρ", r"\sigma": "σ", r"\gamma": "γ"}
    words = {r"\max": "max", r"\min": "min", r"\exp": "exp", r"\ln": "ln",
             r"\log": "log"}
    tex = tex.replace("$", " ")

    def sub(mo):
        c = mo.group(0)
        return greek.get(c, words.get(c, " "))

    tex = _LATEX_CMD.sub(sub, tex)
    tex = re.sub(r"[{}\[\]_^&\\()\s]", "", tex)
    return tex


def gate(records: list[dict], pass1_symbols: set[str],
         equation: dict, anchors: dict | None = None) -> list[GatedLaw]:
    """Deterministic checks. No LLM anywhere in here."""
    out: list[GatedLaw] = []

    context = " ".join(equation.get("text_before", []) +
                       equation.get("text_after", []))
    haystack = _norm(context)
    pdf_glyphs = "".join(equation.get("glyphs", []))

    for r in records:
        g = GatedLaw(record=r)
        failed = g.failed

        # 1. does the expression parse?
        #    Parsed against the declared symbol names only, so 'S' stays a
        #    substrate concentration instead of becoming SymPy's singleton
        #    registry.
        mapped = {m["sympy_name"]: m["pass1_symbol"] for m in r["symbol_map"]}
        try:
            g.expr = safe_sympify(r["sympy_expr"], mapped.keys())
        except Exception as exc:
            failed.append("sympy_parse")
            g.notes["parse_error"] = str(exc)[:200]

        # 2. do all symbols bind to a Pass 1 row?
        #    Binding goes through normalize_symbol() first, so K_s / Ks and
        #    \mu_{max} / \mumax / μmax are recognised as the same symbol. Only
        #    what survives normalisation unmatched is unbound, and THAT is
        #    where an OCR misread lands: 'K.' or 'K_H' matches no row.
        index = build_symbol_index(pass1_symbols)
        rebound = []
        for m in r["symbol_map"]:
            claimed = m.get("pass1_symbol")
            if claimed and claimed in pass1_symbols:
                continue
            candidates = [claimed, m.get("latex_symbol"), m.get("sympy_name")]
            hit = None
            for c in candidates:
                n = normalize_symbol(c)
                if n in index:
                    hit = index[n]
                    break
            if hit:
                mapped[m["sympy_name"]] = hit
                rebound.append(f"{m['sympy_name']}: {claimed!r} -> {hit!r}")
        if rebound:
            g.notes["bound_by_normalisation"] = rebound

        unbound = sorted(k for k, v in mapped.items() if not v)
        if unbound:
            failed.append("symbol_unbound")
            g.notes["unbound"] = unbound

        bad = sorted(v for v in mapped.values() if v and v not in pass1_symbols)
        if bad:
            failed.append("symbol_not_in_table")
            g.notes["not_in_table"] = bad
        g.notes["resolved_map"] = dict(mapped)

        # 3. is every symbol in the expression covered by symbol_map?
        if g.expr is not None:
            free = {s.name for s in g.expr.free_symbols}
            missing = sorted(free - set(mapped))
            if missing:
                failed.append("symbol_unmapped")
                g.notes["unmapped"] = missing

        # 4. does the thing it computes exist and sit on the left?
        if r["computes"]:
            if r["computes"] not in mapped:
                failed.append("computes_unmapped")
            elif g.expr is not None and getattr(g.expr, "lhs", None) is not None:
                lhs = {s.name for s in g.expr.lhs.free_symbols}
                if r["computes"] not in lhs:
                    failed.append("computes_not_lhs")

        # 5. law name must be supported by a verbatim span when claimed as prose
        src = r.get("law_name_source")
        if src == "prose":
            span = r.get("law_name_span") or ""
            if not span.strip():
                failed.append("name_span_empty")
            elif _norm(span) not in haystack:
                failed.append("name_span_support")
        elif src == "section" and r.get("law_name_span"):
            failed.append("name_span_unexpected")   # a heading is not a sentence

        # 6. OCR fidelity: do the LaTeX characters match the PDF's own glyphs?
        #    The comparison runs on the LATEX, which is what pix2tex read off
        #    the image, NOT on sympy_expr. sympy_expr is ASCII by construction
        #    ('Eq', 'mu', 'K_dot_X'), so comparing it against printed Greek
        #    reports every ASCII letter as invented and means nothing.
        #    The two sources fail differently -- pix2tex drops subscripts, the
        #    text layer loses structure -- so each covers the other.
        source_latex = equation.get("latex") or ""
        if pdf_glyphs and source_latex:
            from collections import Counter

            def keep(c):
                # letters, digits, and the operators whose loss changes the
                # model without breaking dimensions. A dropped minus turns
                # Teissier's 1 - exp(-S/K_T) into 1 - exp(S/K_T): same
                # symbols, same dimensions, opposite behaviour. Counting only
                # alphanumerics made that invisible to every gate.
                return c.isalnum() or c in "+-="

            def norm_ops(s):
                return s.replace("\u2212", "-").replace("\u2013", "-")

            lc = Counter(c for c in norm_ops(latex_chars(source_latex)) if keep(c))
            pc = Counter(c for c in norm_ops(pdf_glyphs) if keep(c))
            invented = lc - pc
            dropped = pc - lc
            if invented:
                failed.append("latex_glyph_mismatch")
                g.notes["invented_chars"] = dict(invented)
            lost_ops = {k: v for k, v in dropped.items() if k in "+-="}
            if lost_ops:
                # an operator in the PDF that the OCR lost is a FAILURE, not a
                # note: unlike a dropped equation number it changes the model
                failed.append("operator_dropped")
                g.notes["operators_dropped"] = lost_ops
            other = {k: v for k, v in dropped.items() if k not in "+-="}
            if other:
                g.notes["chars_in_pdf_not_in_ocr"] = other

        # 7. dimensional consistency, DERIVED from structure.
        #    Needs no units from Pass 1, which is just as well: on a prose
        #    paper every unit is null.
        if g.expr is not None and not failed:
            try:
                eqs, dmap, partial = derive_dimensions(g.expr)
                ok, rel = solve_dimensions(eqs, dmap, anchors)
                if not ok:
                    failed.append("dimensionally_inconsistent")
                else:
                    g.notes["dimensions"] = {k: str(v) for k, v in rel.items()}
                if partial:
                    # not a failure: an empirical power law is legitimate, but
                    # the check did not fully constrain it and that must show
                    g.notes["dimension_partial"] = partial
            except DimensionError as exc:
                failed.append("dimension_underivable")
                g.notes["dimension_error"] = str(exc)[:200]

        g.status = "flagged" if failed else "accepted"
        out.append(g)
    return out


def run_level_notes(gated: list[GatedLaw], pass1_symbols: set[str]) -> list[str]:
    """Checks about the document, not about one law. Warnings, not failures.

    A warning that is wrong costs attention. A gate failure that is wrong
    costs a correct extraction. Until a check has been wrong zero times across
    a decent number of papers, warning is the right severity.
    """
    notes = []

    used = set()
    for g in gated:
        resolved = g.notes.get("resolved_map")
        if resolved is not None:
            used.update(v for v in resolved.values() if v)
        else:
            for m in g.record["symbol_map"]:
                if m["pass1_symbol"]:
                    used.add(m["pass1_symbol"])

    orphans = sorted(pass1_symbols - used)
    if orphans:
        notes.append(
            f"QUANTITIES IN NO EQUATION: {orphans}. A quantity that appears in "
            f"no formula is usually a measurement method rather than a model "
            f"variable, or a Pass 1 hallucination. Neither can be caught by any "
            f"Pass 1 gate."
        )

    kinds = {g.record["law_kind"] for g in gated}
    if kinds and "BalanceLaw" not in kinds and "RateLaw" in kinds:
        notes.append(
            "NO BALANCE LAW: rate laws are present but nothing conserves "
            "anything. StateQuantity classifications cannot be checked from "
            "this document alone; defer them to the assembled graph."
        )
    return notes


# ----------------------------------------------------------------------
# Turtle
# ----------------------------------------------------------------------

# Pass 2's law_kind enum is unchanged (the LLM still chooses among four).
# The GRAPH has fewer: rate laws and closures are one class, because nothing
# downstream treats them differently and both can have rivals. A Definition
# (e.g. mu from two biomass readings) gets no subclass: it stays in the paper
# graph and is never merged into the shared graph.
KIND_TO_CLASS = {
    "RateLaw": "ConstitutiveLaw",
    "ClosureRelation": "ConstitutiveLaw",
    "BalanceLaw": "BalanceLaw",
    "Definition": "Law",
}


def rename_to_graph_symbols(g: GatedLaw):
    """Rewrite the formula so its symbols are formula_name(Pass 1 symbol).

    The LLM chooses ASCII names freely (mu_max, K_s). The graph needs the
    formula's symbols to point at quantity nodes, so each is renamed to the
    formula name of the Pass 1 row it was bound to. This is a bijective
    renaming: structure is untouched, so every gate result still holds.

    Returns (expr, None) or (None, reason). Only accepted laws are renamed;
    a flagged law keeps the LLM's names, since its binding is what failed.
    """
    if g.status != "accepted" or g.expr is None:
        return None, "not accepted"
    resolved = g.notes.get("resolved_map") or {}
    targets = [v for v in resolved.values() if v]
    if len(set(targets)) != len(targets):
        return None, "two formula symbols bound to one quantity"
    clash = check_formula_names(set(targets))
    if clash:
        return None, f"formula-name collision {clash}"
    sub = {sympy.Symbol(k): sympy.Symbol(formula_name(v))
           for k, v in resolved.items() if v}
    return g.expr.xreplace(sub), None


def to_turtle(gated: list[GatedLaw], doc_id: str, provenance: dict,
              equation: dict) -> str:
    paper, section = split_doc_id(doc_id)
    doc = doc_iri(doc_id)
    block = equation.get("block_id") or ""
    prov = (f'{provenance["model"]}/{provenance["decoding_tier"]}/'
            f'{provenance["prompt_version"]}')
    lines = [
        PREFIXES,
        f'{doc} a mkg:Document ; mkg:doi {json.dumps(paper)} .',
        "",
    ]

    for i, g in enumerate(gated):
        r = g.record
        lid = law_iri(doc_id, block, r["local_id"])
        if len(gated) > 1:
            # one block yielding two laws: keep them apart
            lid += f"_{i + 1}"
        resolved = g.notes.get("resolved_map") or {
            m["sympy_name"]: m["pass1_symbol"] for m in r["symbol_map"]}

        renamed, why_not = rename_to_graph_symbols(g)
        formula = sympy.sstr(renamed) if renamed is not None else r["sympy_expr"]

        lines += [
            f"{lid} a mkg:{KIND_TO_CLASS[r['law_kind']]} ;",
            f"    mkg:hasFormula {json.dumps(formula)} ;",
        ]
        if r.get("law_name"):
            lines.append(f"    rdfs:label {json.dumps(r['law_name'])} ;")

        computed_sym = resolved.get(r["computes"]) if r.get("computes") else None
        if computed_sym:
            lines.append(f"    mkg:computes {quantity_iri(doc_id, computed_sym)} ;")
        for sym in sorted({v for v in resolved.values() if v} - {computed_sym}):
            lines.append(f"    mkg:usesQuantity {quantity_iri(doc_id, sym)} ;")

        just = [f"{lid}_eqspan"]
        if r.get("law_name_span"):
            just.append(f"{lid}_namespan")
        lines.append(f"    mkg:hasJustification {' , '.join(just)} ;")
        lines.append(
            f"    mkg:hasStatus mkg:{g.status} "
            + "".join(f'; mkg:failedGate "{f}" ' for f in g.failed) + ".")
        lines.append("")

        # the equation block itself is the justification for the formula: a
        # located piece of the paper, checked by the glyph gate, not a sentence
        lines += [
            f"{lid}_eqspan a mkg:SourceSpan ;",
            f"    mkg:fromDocument {doc} ;",
            f"    mkg:locator {json.dumps(block)} ;",
            f"    mkg:latexSource {json.dumps(equation.get('latex') or '')} ;",
            f'    mkg:producedBy "{prov}" .',
            "",
        ]
        if r.get("law_name_span"):
            lines += [
                f"{lid}_namespan a mkg:SourceSpan ;",
                f"    mkg:fromDocument {doc} ;",
                f"    mkg:verbatimText {json.dumps(r['law_name_span'])} ;",
                f"    mkg:locator {json.dumps(section)} ;",
                f'    mkg:producedBy "{prov}" .',
                "",
            ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def format_table(quantities: list[dict]) -> str:
    rows = []
    for q in quantities:
        margin = q.get("class_margin", "")
        rows.append(f"  {q['symbol']}  |  {q.get('description','')}  "
                    f"|  class={q['quantity_class']} ({margin})")
    return "\n".join(rows)


def run_pass2(doc_id: str, equation: dict, quantities: list[dict],
              harness: Harness, log_path: str = "runs.jsonl",
              prompt_version: str = "p2.v1"):
    user = USER_TEMPLATE.format(
        doc_id=doc_id,
        section=equation.get("section_context") or "unknown",
        table=format_table(quantities),
        latex=equation.get("latex") or "(no LaTeX available)",
        context="\n".join(equation.get("text_before", []) +
                          equation.get("text_after", [])),
    )
    res = harness.call(system=SYSTEM_PROMPT, user=user,
                       schema=PASS2_SCHEMA, schema_name="pass2_laws",
                       prompt_version=prompt_version)
    log_call(log_path, doc_id, 2, res, user)

    if not res.ok:
        print(f"FAILED after {res.attempts} attempts", file=sys.stderr)
        for e in res.errors:
            print("  " + e, file=sys.stderr)
        return None, res

    symbols = {q["symbol"] for q in quantities}
    anchors = {}
    gated = gate(res.data["laws"], symbols, equation, anchors)
    return gated, res


def summary(gated: list[GatedLaw], pass1_symbols: set[str]) -> str:
    n = len(gated)
    acc = sum(1 for g in gated if g.status == "accepted")
    lines = [f"{n} laws, {acc} accepted, {n - acc} flagged"]
    for g in gated:
        r = g.record
        name = r.get("law_name") or "(unnamed)"
        src = r.get("law_name_source") or "-"
        mark = "OK " if g.status == "accepted" else "FLG"
        lines.append(f"  {mark} {r['local_id']:4} {name:18} [{src:7}] "
                     f"{r['law_kind']:15} {r['sympy_expr'][:52]}")
        if g.failed:
            lines.append(f"       failed: {g.failed}")
        for k in ("unbound", "not_in_table", "unmapped", "invented_chars",
                  "operators_dropped", "chars_in_pdf_not_in_ocr",
                  "bound_by_normalisation", "parse_error", "dimension_partial"):
            if k in g.notes:
                lines.append(f"       {k}: {g.notes[k]}")
        if "dimensions" in g.notes:
            lines.append(f"       dims: {g.notes['dimensions']}")
    for note in run_level_notes(gated, pass1_symbols):
        lines.append("  " + note)
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("pass2_input", help="output of marker_adapter.py")
    ap.add_argument("--quantities", required=True,
                    help="output of consolidate.py for this section")
    ap.add_argument("--block", type=int, required=True,
                    help="1-based index of the equation block to process")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", default="pass2_out.ttl")
    ap.add_argument("--log", default="runs.jsonl")
    ap.add_argument("--prompt-version", default="p2.v1")
    ap.add_argument("--anchor", action="append", default=[],
                    help="pin a symbol's dimension, e.g. --anchor S=1 "
                         "--anchor X=1. With both concentrations pinned, "
                         "Contois's Ks resolves to 0 (dimensionless) while "
                         "Monod's resolves to 1.")
    a = ap.parse_args()

    data = json.load(open(a.pass2_input, encoding="utf-8"))
    eqs = data["equations"]
    if not 1 <= a.block <= len(eqs):
        raise SystemExit(f"--block must be 1..{len(eqs)}")
    equation = eqs[a.block - 1]

    cons = json.load(open(a.quantities, encoding="utf-8"))
    quantities = cons["quantities"]

    anchors = {}
    for spec in a.anchor:
        k, _, v = spec.partition("=")
        anchors[k.strip()] = int(v)

    print(f"block {a.block}: p{equation['page_index']+1} "
          f"[{equation.get('section_context')}]")
    print(f"  latex : {(equation.get('latex') or '(none)')[:90]}")
    print(f"  table : {len(quantities)} quantities from "
          f"{cons['n_runs']} voted runs\n")

    key = a.api_key or os.environ.get("GROQ_API_KEY") or "EMPTY"
    h = Harness(base_url=a.base_url, model=a.model, api_key=key)
    gated, res = run_pass2(a.doc_id, equation, quantities, h,
                           log_path=a.log, prompt_version=a.prompt_version)
    if gated is None:
        sys.exit(1)

    # re-gate with anchors if the user supplied any
    if anchors:
        gated = gate(res.data["laws"], {q["symbol"] for q in quantities},
                     equation, anchors)

    print(f"decoding tier: {res.tier}   attempts: {res.attempts}   "
          f"{res.elapsed_s}s")
    if res.tier != "strict":
        print("WARNING: not in strict mode. This run is not evidence about the model.")
    print(summary(gated, {q["symbol"] for q in quantities}))

    open(a.out, "w", encoding="utf-8").write(
        to_turtle(gated, a.doc_id, res.provenance(), equation))
    print(f"\nwrote {a.out}")
    print("tier report:", tier_report(a.log))
