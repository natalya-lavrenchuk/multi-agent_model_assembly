"""
pass1.py -- extract quantities (symbol, unit, class) from one document.

PIPELINE
    document text
      -> LLM (strict JSON)        symbol, description, unit STRING, class, span
      -> gate (deterministic)     unit parses? span real? no conflicts?
      -> accepted / flagged
      -> Turtle for the accepted ones

The LLM never emits dimension exponents. It copies the unit string it sees.
units.parse_unit() derives M/L/T/N/Theta and the SI conversion. A wrong
exponent is therefore impossible; a wrong unit string is catchable.

UNITS ARE THREE-VALUED (v2). Most papers print no units for most symbols.
    unit_string = "g L-1"  -> given, parsed, dimensions asserted
    unit_string = "-"      -> the document SAYS dimensionless, dims all zero
    unit_string = null     -> the document says nothing. NO dimension claim is
                              made and mkg:hasUnit is not asserted.
Collapsing null into dimensionless would assert a dimension the document never
stated, and Pass 2's dimensional check would then verify equations against a
fabricated premise.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass

from harness import Harness, log_call, tier_report
from units import parse_unit, dims_equal
from mkg_ids import (PREFIXES, _ident, doc_iri, quantity_iri,
                     split_doc_id)


# ----------------------------------------------------------------------
# Schema. Written for STRICT mode: every property required,
# additionalProperties false, no oneOf/anyOf. Nullable via type lists.
# ----------------------------------------------------------------------

PASS1_SCHEMA = {
    "type": "object",
    "properties": {
        "quantities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "local_id": {
                        "type": "string",
                        "description": "Short unique id within this document, e.g. q1, q2."
                    },
                    "symbol": {
                        "type": "string",
                        "description": "The symbol exactly as printed. Do not normalise or rename."
                    },
                    "description": {
                        "type": "string",
                        "description": "What the document says this quantity is, in the document's own words."
                    },
                    "unit_string": {
                        "type": ["string", "null"],
                        "description": (
                            "The unit exactly as printed, e.g. 'g L-1' or 'm2 s-1'. "
                            "Use null if the document prints no unit for this quantity -- "
                            "this is the common case and is not a failure. "
                            "Use '-' ONLY if the document explicitly states the quantity is dimensionless."
                        )
                    },
                    "quantity_class": {
                        "type": "string",
                        "enum": ["StateQuantity", "RateQuantity", "Parameter",
                                 "Constant", "ControlQuantity"],
                        "description": (
                            "How the number is obtained. Work through these questions IN "
                            "ORDER and take the first match. Do not choose by matching "
                            "words that appear in the quantity's name.\n"
                            "Q1. Does the value change during a single experiment, as time "
                            "passes or as you move through the equipment? If NO, go to Q3.\n"
                            "Q2. The value changes. Does the model track its accumulation, "
                            "so that you need its initial value and must integrate to "
                            "follow it? If yes, it is a StateQuantity. If instead it is "
                            "computed at each instant from an explicit expression of other "
                            "quantities, it is a RateQuantity.\n"
                            "Q3. The value is fixed for the whole experiment. Who supplies "
                            "the number? If the experimenter sets it and could have set it "
                            "differently, such as a feed rate, a temperature setpoint or a "
                            "column length, it is a ControlQuantity. If it is a universal "
                            "physical constant whose value you could look up in a handbook "
                            "WITHOUT knowing what system is being modelled, such as the gas "
                            "constant or Avogadro's number, it is a Constant. Otherwise it "
                            "is a Parameter: if you would have to know which organism, "
                            "material, catalyst or equipment is involved before you could "
                            "look the value up, it is a Parameter. Kinetic and transport "
                            "coefficients are Parameters.\n"
                            "Words such as constant, coefficient, rate, number, factor, "
                            "index and parameter appear inside quantity names as naming "
                            "conventions. They are not classifications. Classify by the "
                            "role the quantity plays, never by its name."
                        )
                    },
                    "class_stated": {
                        "type": "boolean",
                        "description": "True only if the document explicitly says how this quantity is obtained. False if you worked it out yourself."
                    },
                    "index_over": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sets the symbol is subscripted over, e.g. ['component'] or ['component','axial position']. Empty list if scalar."
                    },
                    "span_text": {
                        "type": "string",
                        "description": "The sentence or clause from the document that defines this quantity, copied WORD FOR WORD. Must appear verbatim in the text you were given."
                    },
                    "span_locator": {
                        "type": "string",
                        "description": "Where it came from, e.g. 'Nomenclature', 'Section 2.1', or 'text following Eq. 2'."
                    }
                },
                "required": ["local_id", "symbol", "description", "unit_string",
                             "quantity_class", "class_stated", "index_over",
                             "span_text", "span_locator"],
                "additionalProperties": False
            }
        }
    },
    "required": ["quantities"],
    "additionalProperties": False
}


SYSTEM_PROMPT = """\
You extract the physical quantities of mechanistic process models from engineering \
documents.

The text you are given was extracted from a PDF. Equations are often shattered into \
loose characters and symbols on separate lines, with no readable structure. Do not \
try to read the equations. The information you need is in the PROSE around them.

Most papers have NO nomenclature table. Symbols are usually defined inline, in \
sentences that follow an equation, in the pattern "Where, mu is the specific growth \
rate, S is the substrate concentration". Those sentences are your primary source.

Your only job is to list the physical quantities: their symbols, what they mean, \
and their units where given. You are NOT extracting equations, assumptions or \
conditions. Ignore those.

Rules:

1. Copy symbols exactly as printed. If the document writes K_s, write K_s. Never \
rename, expand or normalise a symbol. If a Greek letter is spelled out in the \
extracted text, copy what is printed.

2. Copy unit strings exactly as printed. Do not convert, do not tidy, and above all \
do not guess. Most quantities in this document will have NO printed unit. That is \
normal and expected. Use null. Inventing a plausible unit is a serious error and is \
worse than returning null. Use "-" only if the document explicitly says the quantity \
is dimensionless.

3. span_text must be copied word for word from the text you were given. It may be a \
full sentence or a clause within one. Do not paraphrase, do not shorten, do not fix \
typos or spacing. If you cannot find a verbatim passage that defines a quantity, do \
not report that quantity.

4. Report every distinct symbol once. If the same symbol carries two different \
meanings, report it twice with different local_id values.

5. class_stated is a claim about the SPAN you quoted, not about the quantity.

Set it to true ONLY if the span itself states how the value is obtained. Qualifying \
statements are ones reporting that the value was measured, fitted, estimated or \
regressed from data, integrated from a balance, computed by a named equation, taken \
from a cited source, assumed, or held fixed at a stated value.

Set it to false in every other case, including all of these:
- the span only names the quantity and says what it means
- the span gives a value or a unit but does not say where the value came from
- the quantity's NAME contains a word such as constant, coefficient, parameter, \
factor, rate, number or index. Those words describe what the quantity is called. \
They are not statements about how its value was obtained.
- you chose the class using your own engineering knowledge

Choosing the class yourself is expected and correct. It simply means class_stated is \
false. In most papers, most rows will be false.

6. Exclude: statistical measures of model fit such as R2 and RMSE, software or \
solver settings, and quantities that appear only inside a citation to another paper \
without being defined here.

Return JSON only.\
"""


USER_TEMPLATE = """\
Document: {doc_id}

The text below is a PDF extraction. Find every physical quantity that the text \
defines, working from the prose definitions rather than the equation fragments.

--- BEGIN DOCUMENT ---
{doc_text}
--- END DOCUMENT ---\
"""


# ----------------------------------------------------------------------
# Gate
# ----------------------------------------------------------------------

@dataclass
class GatedQuantity:
    record: dict
    parse: object | None          # UnitParse, or None when no unit was printed
    unit_state: str               # "given" | "dimensionless" | "not_given"
    status: str                   # "accepted" | "flagged"
    failed: list[str]


def _norm_ws(s: str) -> str:
    """Normalise for verbatim comparison.

    NFKC is applied because PDF extraction emits typographic ligatures and
    lookalike codepoints that no reader would treat as different characters:
    U+FB01 'fi' in "specific", U+00B5 MICRO SIGN vs U+03BC GREEK SMALL LETTER
    MU, en/em dashes, curly quotes. Without it the span gate rejects spans that
    ARE verbatim, on a difference that does not exist on the page.

    This does not weaken the gate. It makes two strings that differ only in
    encoding compare equal. A fabricated quotation is still caught.
    """
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def unit_state(unit_string) -> str:
    """Three-way split. null is NOT the same as dimensionless."""
    if unit_string is None:
        return "not_given"
    s = unit_string.strip()
    if s == "":
        return "not_given"
    if s in ("-", "--", "dimensionless", "[-]", "(-)"):
        return "dimensionless"
    return "given"


def gate(records: list[dict], source_text: str) -> list[GatedQuantity]:
    """Deterministic checks. Nothing here involves an LLM."""
    haystack = _norm_ws(source_text)
    out: list[GatedQuantity] = []

    seen_ids: dict[str, int] = {}
    by_symbol: dict[str, list[dict]] = {}

    for r in records:
        seen_ids[r["local_id"]] = seen_ids.get(r["local_id"], 0) + 1
        by_symbol.setdefault(r["symbol"], []).append(r)

    for r in records:
        failed: list[str] = []

        # 1. unit. Only a PRINTED unit can fail to parse. A missing unit is a
        #    fact about the paper, not an extraction error.
        st = unit_state(r["unit_string"])
        if st == "not_given":
            p = None
        else:
            p = parse_unit(r["unit_string"])
            if not p.ok:
                failed.append("unit_parse")

        # 2. is the span real? verbatim substring, whitespace-insensitive.
        #    this catches a fabricated quotation for free.
        if not r["span_text"].strip():
            failed.append("span_empty")
        elif _norm_ws(r["span_text"]) not in haystack:
            failed.append("span_support")

        # 3. duplicate local_id
        if seen_ids[r["local_id"]] > 1:
            failed.append("duplicate_id")

        # 4. same symbol, incompatible dimensions. Only comparable when BOTH
        #    sides carry a real dimension claim.
        sibs = by_symbol[r["symbol"]]
        if len(sibs) > 1 and p is not None and p.ok:
            for s in sibs:
                if s is r:
                    continue
                if unit_state(s["unit_string"]) == "not_given":
                    continue
                q = parse_unit(s["unit_string"])
                if q.ok and not dims_equal(p.dims, q.dims):
                    failed.append("symbol_conflict")
                    break
        
        # 5. repeated symbol. Rule 4 permits the same symbol twice when it
        #    carries two meanings, so this is not "reject repeats". Three
        #    branches, per the KG4DT lesson: not present, present and
        #    identical, present but DIFFERENT.
        if len(sibs) > 1:
            pos = next(i for i, s in enumerate(sibs) if s is r)
            classes = {s["quantity_class"] for s in sibs}
            if len(classes) > 1:
                # same symbol, different class, no basis to choose. Flag ALL
                # copies: silently keeping one would be a coin flip.
                failed.append("class_conflict")
            elif pos > 0 and _norm_ws(r["span_text"]) == _norm_ws(sibs[0]["span_text"]):
                # identical row, nothing distinguishing it. Keep the first,
                # flag the rest.
                failed.append("duplicate_symbol")

        out.append(GatedQuantity(
            record=r, parse=p, unit_state=st,
            status="flagged" if failed else "accepted",
            failed=failed,
        ))
    return out


# ----------------------------------------------------------------------
# Turtle output
# ----------------------------------------------------------------------

def to_turtle(gated: list[GatedQuantity], doc_id: str, provenance: dict) -> str:
    """Write ALL rows, accepted and flagged. Flagged ones stay in the graph
    but are invisible to model construction, which reads only mkg:accepted.

    IRIs come from mkg_ids: a quantity is keyed on (paper, section, symbol),
    so every run and Pass 2 name the same symbol the same way. The Document
    is the PAPER; the section goes into the span locator."""
    paper, section = split_doc_id(doc_id)
    doc = doc_iri(doc_id)
    lines = [
        PREFIXES,
        f'{doc} a mkg:Document ; mkg:doi {json.dumps(paper)} .',
        "",
    ]
    units_written: set[str] = set()
    dup_count: dict[str, int] = {}

    for g in gated:
        r, p = g.record, g.parse
        # rule 4 of the prompt allows one symbol with two meanings: the second
        # copy gets a suffix instead of silently merging into the first
        n = dup_count.get(r["symbol"], 0)
        dup_count[r["symbol"]] = n + 1
        qid = quantity_iri(doc_id, r["symbol"], dup=n)
        span_id = f"{qid}_span"

        # unit_ref is None when the document printed no unit. In that case no
        # mkg:hasUnit triple is written at all: the graph records that the
        # document made no dimension claim, rather than claiming dimensionless.
        unit_ref = None
        if g.unit_state != "not_given":
            unit_ref = "mkg:UNPARSEABLE"
            if p is not None and p.ok:
                uid = _ident(p.normalised) or "dimensionless"
                unit_ref = f"mkg:u_{uid}"
                if uid not in units_written:
                    units_written.add(uid)
                    dims = " ; ".join(
                        f"mkg:dim{k} {p.dims.get(k, 0)}"
                        for k in ("M", "L", "T", "N", "Theta")
                    )
                    lines += [
                        f"{unit_ref} a mkg:Unit ;",
                        f'    mkg:unitString {json.dumps(p.original)} ;',
                        f"    {dims} ;",
                        f"    mkg:toSIFactor {p.si_factor} ; mkg:toSIOffset {p.si_offset} .",
                        "",
                    ]

        idx = " , ".join(f"mkg:idx_{_ident(i)}" for i in r.get("index_over", []))
        lines += [
            f"{qid} a mkg:{r['quantity_class']} ;",
            f"    mkg:hasSymbol {json.dumps(r['symbol'])} ;",
            f"    mkg:hasDescription {json.dumps(r['description'])} ;",
            # Did the DOCUMENT say how this quantity is obtained, or did the
            # model infer the class? The class is asserted either way; this is
            # what tells a reviewer which rows rest on the model's judgement.
            f"    mkg:classStated {'true' if r['class_stated'] else 'false'} ;",
        ]
        if unit_ref is not None:
            lines.append(f"    mkg:hasUnit {unit_ref} ;")
        else:
            lines.append("    mkg:unitNotGiven true ;")
        if idx:
            lines.append(f"    mkg:hasIndexSet {idx} ;")
        loc = r.get("span_locator") or ""
        lines += [
            f"    mkg:hasJustification {span_id} ;",
            f"    mkg:hasStatus mkg:{g.status} " +
            ("".join(f'; mkg:failedGate "{f}" ' for f in g.failed)) + ".",
            "",
            f"{span_id} a mkg:SourceSpan ;",
            f"    mkg:fromDocument {doc} ;",
            f'    mkg:verbatimText {json.dumps(r["span_text"])} ;',
            f'    mkg:locator {json.dumps(section + (": " + loc if loc else ""))} ;',
            f'    mkg:producedBy "{provenance["model"]}/{provenance["decoding_tier"]}/'
            f'{provenance["prompt_version"]}" .',
            "",
        ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def run_pass1(doc_id: str, doc_text: str, harness: Harness,
              log_path: str = "runs.jsonl", prompt_version: str = "p1.v6"):
    user = USER_TEMPLATE.format(doc_id=doc_id, doc_text=doc_text)
    res = harness.call(
        system=SYSTEM_PROMPT, user=user,
        schema=PASS1_SCHEMA, schema_name="pass1_quantities",
        prompt_version=prompt_version,
    )
    log_call(log_path, doc_id, 1, res, user)

    if not res.ok:
        print(f"FAILED after {res.attempts} attempts", file=sys.stderr)
        for e in res.errors:
            print("  " + e, file=sys.stderr)
        return None, res

    gated = gate(res.data["quantities"], doc_text)
    return gated, res


def last_logged_call(log_path: str, doc_id: str | None = None) -> dict:
    """Most recent successful pass-1 call in the log, optionally for one doc.

    The gate is deterministic, so changing it never requires a new API call.
    Re-gating a logged run costs nothing and keeps the two results comparable:
    same model, same tier, same prompt version, same input hash.
    """
    rows = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("ok") or r.get("pass") != 1:
                continue
            if doc_id and r.get("doc_id") != doc_id:
                continue
            rows.append(r)
    if not rows:
        raise SystemExit(f"no successful pass-1 call found in {log_path}")
    return rows[-1]

def run_level_notes(gated: list[GatedQuantity]) -> list[str]:
    """Checks about the RUN, not about any single row. These are warnings,
    not gate failures: a document that genuinely carries no information about
    how quantities are obtained SHOULD produce one class for everything."""
    notes = []
    classes = {g.record["quantity_class"] for g in gated}
    if len(gated) >= 4 and len(classes) == 1:
        only = next(iter(classes))
        notes.append(
            f"CLASS COLLAPSE: all {len(gated)} rows classed "
            f"{classes.pop()}. Either the source carries no distinguishing "
            f"information, or this run degenerated. Check against other runs "
            f"on the same input."
        )
    syms = [g.record["symbol"] for g in gated]
    if len(syms) != len(set(syms)):
        dupes = sorted({s for s in syms if syms.count(s) > 1})
        notes.append(f"REPEATED SYMBOLS: {dupes}")
    if "RateQuantity" in classes and "Parameter" not in classes:
        notes.append(
            "RATE WITHOUT PARAMETERS: a rate quantity is present but nothing is "
            "classed Parameter. Rate laws are constitutive and normally carry "
            "fitted coefficients specific to the system. Check whether fixed "
            "quantities were misclassified as Constant."
        )
    return notes

def summary(gated: list[GatedQuantity]) -> str:
    n = len(gated)
    acc = sum(1 for g in gated if g.status == "accepted")
    counts: dict[str, int] = {}
    for g in gated:
        for f in g.failed:
            counts[f] = counts.get(f, 0) + 1

    ustates: dict[str, int] = {}
    for g in gated:
        ustates[g.unit_state] = ustates.get(g.unit_state, 0) + 1

    lines = [f"{n} quantities, {acc} accepted, {n - acc} flagged"]
    lines.append("  units: " + ", ".join(
        f"{k}={v}" for k, v in sorted(ustates.items())))
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {v}")
    for g in gated:
        if g.failed:
            u = g.record["unit_string"]
            lines.append(f"  FLAG {g.record['symbol']:12s} unit={u!r:16s} {g.failed}")
    for note in run_level_notes(gated):
        lines.append("  " + note)
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("textfile")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", default="pass1_out.ttl")
    ap.add_argument("--log", default="runs.jsonl")
    ap.add_argument("--prompt-version", default="p1.v6",
                    help="Recorded in the log and in mkg:producedBy. Bump it "
                         "whenever the prompt text changes.")
    ap.add_argument("--regate", action="store_true",
                    help="Re-run the deterministic gate on the last logged call "
                         "for this doc-id instead of calling the API. Costs no "
                         "tokens. Use after changing gate logic.")
    a = ap.parse_args()

    import os
    text = open(a.textfile, encoding="utf-8").read()
    print(f"input: {a.textfile}  {len(text):,} chars  ~{len(text.split()):,} words")

    if a.regate:
        rec = last_logged_call(a.log, a.doc_id)
        gated = gate(rec["data"]["quantities"], text)
        prov = {
            "model": rec["model"],
            "decoding_tier": rec["tier"],
            "prompt_version": rec["prompt_version"],
        }
        print(f"REGATE of call at {rec['timestamp']}  "
              f"model={rec['model']}  tier={rec['tier']}  "
              f"prompt={rec['prompt_version']}")
        if rec.get("input_chars") != len(text):
            print(f"WARNING: logged input was {rec.get('input_chars'):,} chars, "
                  f"this file is {len(text):,}. Spans are being checked against "
                  f"text the model did not see.")
        print(summary(gated))
    else:
        key = a.api_key or os.environ.get("GROQ_API_KEY") or "EMPTY"
        h = Harness(base_url=a.base_url, model=a.model, api_key=key)
        gated, res = run_pass1(a.doc_id, text, h, log_path=a.log,
                               prompt_version=a.prompt_version)
        if gated is None:
            sys.exit(1)
        prov = res.provenance()
        print(f"decoding tier: {res.tier}   attempts: {res.attempts}   "
              f"{res.elapsed_s}s")
        if res.tier != "strict":
            print("WARNING: not in strict mode. This run is not evidence about the model.")
        print(summary(gated))

    open(a.out, "w", encoding="utf-8").write(to_turtle(gated, a.doc_id, prov))
    print(f"\nwrote {a.out}")
    print("tier report:", tier_report(a.log))
