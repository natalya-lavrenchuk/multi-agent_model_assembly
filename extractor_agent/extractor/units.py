"""units.py -- a unit as printed ('g L^-1', 'g g^-1 h^-1', '10^-12 mmol cell^-1 h^-1') -> a pint unit:
canonical string, dimension and factor to SI. Code only, no LLM.

    python -m extractor.units            # report: how many unit strings in all papers parse, which do not

graph.py writes the result on each Variable/Parameter node (mkg:unit, mkg:dimension, mkg:siFactor) next to
the unit as printed (mkg:unitString). A unit that does not parse keeps only its printed string and is listed
by the report for a person.
Domain units pint lacks are defined here: cell (own dimension, so cells/mL is not g/L), OD (optical density),
CV (column volumes), and annotations such as 'g biomass / g glucose' are dropped from the unit and kept as a note.
"""
from __future__ import annotations
import glob
import json
import re
import sys

import pint

U = pint.UnitRegistry()
U.define('cell = [cell] = cells')
U.define('OD = [optical_density] = OD600 = od')
U.define('column_volume = [] = CV')
U.define('percent_saturation = 0.01 = percent_sat')
for _u in ('chain', 'gene', 'mRNA', 'count_ = [] = number = num'):    # counts per cell, per gene: dimensionless
    try:
        U.define(_u if '=' in _u else f'{_u} = []')
    except Exception:
        pass
SINGLE_STAR = r'(?<!\*)\*(?!\*)|\s+'                      # a product sign or a space, not the ** of a power
SUP = {'⁻': '-', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁰': '0'}
PLACEHOLDER = {'length': 'm', 'time': 's', 'mass': 'kg', 'volume': '(m*m*m)', 'number': 'num'}   # dimension words, no unit printed
DIMENSIONLESS = {'', '-', '—', '–', 'dimensionless', 'unitless', 'none', '1', '[-]', '(-)'}
WORDS = re.compile(r'\b(biomass|glucose|substrate|product|penicillin|oxygen|water|carbon|nitrogen|photons?|crystal|'
                   r'solvent|catalyst|of saturation|saturation|dry weight|dw|cdw|pcm)\b', re.I)
SPECIES = re.compile(r'\b(m?mol)\s+(?!kg|m\b|m\*|L\b|s\b|h\b)([A-Z][\w{}\-]*)')   # 'mol N', 'mmol CO_2', 'mol X_{cd}-C': what is counted


def _clean(s: str) -> str:
    t = re.sub(r'[⁻¹²³⁴⁰]+', lambda m: '^' + ''.join(SUP[c] for c in m.group(0)), s)   # m² -> m^2, s⁻¹ -> s^-1
    t = re.sub(r'(?<=[A-Za-z])-(\d)', r'^-\1', t) if re.search(r'[A-Za-z]-\d', t) and '^' not in t else t   # cm-3 -> cm^-3
    t = re.sub(r'kgcat|kg\s*cat', 'kg', t)
    t = re.sub(r'\bkg/ms(\^?2)?\b', lambda m: 'kg/(m*s' + ('^2' if m.group(1) else '') + ')', t)   # kg/(m s), not per millisecond
    t = re.sub(r'\bgL\b', 'g L', t)
    t = t.replace('°C', 'delta_degC').replace('#', 'num ')
    t = re.sub(r'\bas\s+[A-Z]\w*', '', t)                 # 'mM as N'
    t = t.strip().strip('$').replace('μ', 'u').replace('µ', 'u').replace('·', '*').replace('×', '*')
    t = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', t)
    t = re.sub(r'\\(?:,|;|!| )', ' ', t)
    t = re.sub(r'_\{[^}]*\}', '', t)                       # X_{cd}-C, annotations with subscripts
    t = re.sub(r'\^\{([^}]*)\}', r'^(\1)', t)              # ^{-1} -> ^(-1)
    t = t.replace('{', '').replace('}', '').replace('\\', '')
    t = re.sub(r'(?<=[A-Za-z])\.(?=[A-Za-z])', '*', t)     # g.L^-1 -> g*L^-1
    t = re.sub(r'(?<=\))-(?=[A-Za-z])', '*', t)
    t = re.sub(r'\bl\b', 'L', t)                           # g/l -> g/L (pint: l is litre too, keep one spelling)
    t = re.sub(r'(\d)\s*\^\s*\(?(-?\d+)\)?', r'\1**\2', t) # 10^-12 -> 10**-12
    t = re.sub(r'\^\s*\((-?[\d.]+)\)', r'**\1', t)        # h^(-1) -> h**-1
    t = re.sub(r'\^\s*(-?[\d.]+)', r'**\1', t)              # h^-1 -> h**-1
    t = re.sub(r'\bper\s+', '1/', t)
    t = re.sub(r'%', ' percent ', t)
    t = re.sub(r'-(?=[A-Za-z])', '*', t)                   # mRNA/gene-h -> gene*h
    return ' '.join(t.split())


def parse(printed: str | None) -> dict | None:
    """{'unit', 'dimension', 'si_factor', 'note'} or None if it does not parse."""
    if printed is None:
        return None
    raw = printed.strip()
    if raw.lower() in DIMENSIONLESS:
        return {'unit': 'dimensionless', 'dimension': 'dimensionless', 'si_factor': 1.0, 'note': None}
    note = None
    sp = SPECIES.findall(raw)
    if sp:                                                 # 'mol N (mol C)^-1': N and C are what is counted, not newton/coulomb
        note = ' '.join(x[1] for x in sp)
        raw = SPECIES.sub(lambda m: m.group(1), raw)
    if any(re.search(rf'\b{w}\b', raw) for w in PLACEHOLDER):   # 'length time^-1': a dimension, no unit printed
        for w, u in PLACEHOLDER.items():
            raw = re.sub(rf'\b{w}\b', u, raw)
        note = ' '.join(filter(None, [note, 'dimension only, no unit printed']))
    t = _clean(raw)
    words = WORDS.findall(t)
    if words:                                              # 'g biomass/g glucose' -> 'g/g', note 'biomass/glucose'
        note = ' '.join(filter(None, [note, ' '.join(w if isinstance(w, str) else w[0] for w in words)]))
        t = ' '.join(WORDS.sub(' ', t).split())
    if t.count('/') == 1 and '(' not in t.split('/')[1] and re.search(SINGLE_STAR, t.split('/')[1].strip()):   # 'cal/g h', 'mRNA/gene*h' = per (g h)
        num, den = t.split('/')
        parts = re.split(SINGLE_STAR, den.strip())
        t = (num.strip() or '1') + '/(' + '*'.join(parts) + ')'
    for attempt in (t, t.replace(' ', '*'), re.sub(r'\s+(?=[A-Za-z])', '*', t)):
        try:
            q = U.parse_expression(attempt)
            if not isinstance(q, U.Quantity):
                q = q * U.dimensionless
            base = q.to_base_units()
            dim = str(q.dimensionality) if q.dimensionality else 'dimensionless'
            return {'unit': f'{q.units:~}' + (f' (x{q.magnitude:g})' if q.magnitude != 1 else ''),
                    'dimension': dim, 'si_factor': float(base.magnitude), 'note': note}
        except Exception:
            continue
    return None


def report():
    seen = {}
    for f in sorted(glob.glob('papers/*/symbols.json')):
        for s, v in json.load(open(f))['symbols'].items():
            if v.get('unit'):
                seen.setdefault(v['unit'], []).append(f'{f.split("/")[1]}:{s}')
    ok = {u: parse(u) for u in seen}
    bad = [u for u, r in ok.items() if r is None]
    print(f'{len(seen)} distinct unit strings: {len(seen) - len(bad)} parse, {len(bad)} do not')
    for u in sorted(bad):
        print(f'  NOT PARSED  {u!r:40} {", ".join(seen[u][:3])}')
    if '-v' in sys.argv:
        for u, r in sorted(ok.items()):
            if r:
                print(f'  {u!r:40} -> {r["unit"]:28} {r["dimension"]:35} {r["si_factor"]:.3g}' + (f'  ({r["note"]})' if r['note'] else ''))


if __name__ == '__main__':
    report()
