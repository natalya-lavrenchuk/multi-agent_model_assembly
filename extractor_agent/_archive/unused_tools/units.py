"""
units.py -- turn a unit string as printed in a paper into dimension exponents.

This is DETERMINISTIC. The LLM never produces dimension exponents; it only
copies the unit string it sees. This module derives the rest. That way a wrong
exponent is impossible, and a wrong unit string is catchable.

Papers write units in ways pint does not accept out of the box:
    g L-1 h-1     (negative exponents without **)
    m2 s-1        (positive exponents without **)
    -             (meaning dimensionless)
    mol/(L min)   (parenthesised denominator)

normalise() fixes those, then pint does the real work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pint

_UREG = pint.UnitRegistry()

# COUNT UNITS (decided Oct 2026, option a): biological counts get their OWN dimension,
# so the dimension check can tell cell/mmol from mmol/mmol. pint already treats
# 'molecule' as an amount of substance (1/N_A mol), which is physically right; left as is.
# Add a new count unit here when a paper needs one; an unknown word fails the parse loudly.
COUNT_UNITS = ("cell", "chain", "mRNA", "gene")
for _u in COUNT_UNITS:
    _UREG.define(f"{_u} = [{_u}]")

# pint dimension name -> our short code
_DIM_MAP = {
    "[mass]": "M",
    "[length]": "L",
    "[time]": "T",
    "[substance]": "N",
    "[temperature]": "Theta",
    "[current]": "I",
    "[luminosity]": "J",
    **{f"[{u}]": u for u in COUNT_UNITS},
}

_DIMENSIONLESS_TOKENS = {"-", "", "–", "—", "[-]", "(-)", "1",
                         "dimensionless", "none", "n/a", "unitless"}


@dataclass
class UnitParse:
    ok: bool
    original: str
    normalised: str
    dims: dict[str, int]          # e.g. {"M": 1, "L": -3}
    si_factor: float
    si_offset: float
    si_string: str
    error: str | None = None


def normalise(s: str) -> str:
    """Rewrite paper unit notation into something pint accepts."""
    s = s.strip()
    if s.lower() in _DIMENSIONLESS_TOKENS:
        return "dimensionless"

    # strip LaTeX wrapping that documentation and papers use
    s = re.sub(r"\\(?:mathrm|text|mathit|mathbf|rm)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[,;:!\s]", " ", s)          # \, \; thin spaces
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"_\{[^{}]*\}|_[A-Za-z0-9]+", "", s)  # drop subscripts: m_IV -> m
    s = s.replace("^{", "^").replace("}", "").replace("{", "")
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
    s = s.strip().strip("\\")
    if s.lower() in _DIMENSIONLESS_TOKENS:
        return "dimensionless"

    # unicode superscripts and common symbols
    sup = {"⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
           "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
           "µ": "u", "μ": "u", "·": " ", "×": "*", "−": "-", "–": "-"}
    for k, v in sup.items():
        s = s.replace(k, v)

    # "^" is fine for pint but "**" is safer after our rewrites
    s = s.replace("^", "**")

    # Hyphen between unit words, as in 'mmol/cell-h' or 'mRNA/gene-h'. Kontoravdi's
    # Nomenclature writes the same units as 'mmol/cell/h', so the hyphen joins the
    # denominator: a/b-c means a/(b*c). Without this pint reads it as a subtraction.
    while re.search(r"/([A-Za-z]+)-([A-Za-z]+)", s):
        s = re.sub(r"/([A-Za-z]+)-([A-Za-z]+)", r"/(\1*\2)", s)

    # letter(s) immediately followed by an integer -> explicit exponent.
    # matches: L-1  m2  s**-1 (left alone)  min-1  degC (no digit, left alone)
    def _exp(m: re.Match) -> str:
        unit, power = m.group(1), m.group(2)
        return f"{unit}**{power}"

    s = re.sub(r"([A-Za-z]+)\s*(?<!\*\*)(-?\d+)(?![\d.])", _exp, s)

    # collapse whitespace, then treat remaining spaces as multiplication
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?<=[\w\)])\s+(?=[\w\(])", "*", s)
    return s


def parse_unit(s: str | None) -> UnitParse:
    if s is None:
        return UnitParse(True, "", "dimensionless", {}, 1.0, 0.0, "dimensionless")

    norm = normalise(s)
    try:
        q = _UREG.Quantity(1.0, norm)
    except Exception as e:
        return UnitParse(False, s, norm, {}, 0.0, 0.0, "",
                         error=f"{type(e).__name__}: {str(e)[:120]}")

    dims: dict[str, int] = {}
    for pint_dim, power in q.dimensionality.items():
        code = _DIM_MAP.get(pint_dim)
        if code is None:
            return UnitParse(False, s, norm, {}, 0.0, 0.0, "",
                             error=f"unsupported dimension {pint_dim}")
        dims[code] = int(power)

    # affine conversion to SI: value_SI = value * factor + offset
    try:
        at_zero = _UREG.Quantity(0.0, norm).to_base_units().magnitude
        at_one = _UREG.Quantity(1.0, norm).to_base_units().magnitude
        offset = at_zero
        factor = at_one - at_zero
        si_string = str(q.to_base_units().units)
    except Exception as e:
        return UnitParse(False, s, norm, dims, 0.0, 0.0, "",
                         error=f"conversion failed: {str(e)[:120]}")

    return UnitParse(True, s, norm, dims, factor, offset, si_string)


def dims_equal(a: dict[str, int], b: dict[str, int]) -> bool:
    """Compare two dimension dicts, treating a missing key as exponent 0."""
    keys = set(a) | set(b)
    return all(a.get(k, 0) == b.get(k, 0) for k in keys)
