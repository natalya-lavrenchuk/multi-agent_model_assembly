"""
eqfind.py -- locate DISPLAY equations in a born-digital PDF and crop them.

No hardcoded coordinates, margins, column counts or numbering conventions.
Everything that could differ between publishers is measured from the document
itself:

  body font      the font carrying the most characters in the document
  line height    median vertical gap between text baselines
  math spans     spans whose font is NOT the body font and whose text is short

A cluster of math spans is a DISPLAY equation (as opposed to an inline symbol
in a sentence) when ANY of three independent tests fires:

  stacked         numerator above bar above denominator spans more than one
                  line height. Inline math sits on one baseline.
  fraction bar    a thin wide filled rectangle was drawn through it.
  alone on line   no body-font prose shares its baseline. This is the only
                  test that catches single-line display math such as
                  "mu = a + b ln(s)", which has neither of the above.

Each test was added after the previous set MISSED a real equation. The set is
therefore fitted to the documents seen so far and should be re-checked on a
paper from a different publisher before it is trusted.

Crops whose glyph multiset matches are collapsed: papers restate equations in
graphical abstracts and summaries, and the multiset is the same wherever it
sits on the page.

Output: one PNG per equation plus equations.json with bbox, page, the glyphs
actually present in the crop, and the nearest equation number if there is one.
The glyph list is what gates the OCR later: every symbol the OCR claims must
be a character the PDF really contains in that box.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter

import pymupdf

EQNUM = re.compile(r'^\(\s*(\d{1,3})\s*\)$')


# ----------------------------------------------------------------------
# Document-level measurements
# ----------------------------------------------------------------------

def measure_document(doc) -> dict:
    """Body font and median line height, measured, never assumed."""
    font_chars: Counter = Counter()
    heights: list[float] = []

    for page in doc:
        prev_y = None
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for s in line["spans"]:
                    t = s["text"].strip()
                    if t:
                        font_chars[s["font"]] += len(t)
                y0 = line["bbox"][1]
                if prev_y is not None:
                    gap = y0 - prev_y
                    if 4 < gap < 40:
                        heights.append(gap)
                prev_y = y0

    body_font = font_chars.most_common(1)[0][0] if font_chars else ""
    line_h = statistics.median(heights) if heights else 12.0
    return {"body_font": body_font, "line_height": line_h,
            "fonts": font_chars.most_common(8)}


# ----------------------------------------------------------------------
# Span collection
# ----------------------------------------------------------------------

def math_spans(page, body_font: str, max_len: int) -> list[dict]:
    """Spans that are plausibly math: not the body font, and short.

    Math is set glyph by glyph, so spans are 1-3 characters. A long span in a
    non-body font is a heading or an italicised species name, not an equation.
    """
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            for s in line["spans"]:
                t = s["text"].strip()
                if not t or len(t) > max_len:
                    continue
                if s["font"] == body_font:
                    continue
                out.append({"text": t, "font": s["font"],
                            "bbox": tuple(s["bbox"])})
    return out


def body_spans(page, body_font: str) -> list[dict]:
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            for s in line["spans"]:
                t = s["text"].strip()
                if t and s["font"] == body_font:
                    out.append({"text": t, "bbox": tuple(s["bbox"])})
    return out


def fraction_bars(page) -> list[pymupdf.Rect]:
    """Thin wide filled rectangles. Scaled to the page, not to a fixed pt."""
    bars = []
    for dr in page.get_drawings():
        r = dr["rect"]
        if r.height <= 2.5 and r.width >= 6:
            bars.append(r)
    return bars


def is_alone_on_line(page, group, body_font: str, line_h: float) -> bool:
    """Third display test: does the cluster have a line to itself?

    Stacking and fraction bars both describe the SHAPE of an equation, so
    both miss single-line display math such as 'mu = a + b ln(s)'. What makes
    that display rather than inline is position: it sits on its own line with
    no prose beside it. Inline math always shares a baseline with a sentence.
    An equation number at the margin is not prose and is ignored.

    COLUMN-AWARE. In a two-column paper the other column's prose overlaps
    vertically with an equation while being nowhere near it, so vertical
    overlap alone is not enough. Prose must also be horizontally CLOSE.
    Measured separation on a two-column Elsevier paper: inline math has prose
    within ~3 pt, a column gutter is ~18 pt, and line height is ~10 pt. So a
    threshold of about one line height separates the two cases without any
    page geometry being assumed.
    """
    x0, y0, x1, y1 = bbox_of(group)
    mid = (y0 + y1) / 2
    near = line_h * 1.2

    for s in body_spans(page, body_font):
        sx0, sy0, sx1, sy1 = s["bbox"]
        if not (sy0 - line_h * 0.4 <= mid <= sy1 + line_h * 0.4):
            continue
        if EQNUM.match(s["text"].strip()):
            continue
        gap = max(sx0 - x1, x0 - sx1, 0.0)   # 0 if they overlap horizontally
        if gap <= near:
            return False
    return True


# ----------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------

def cluster(spans: list[dict], gap_x: float, gap_y: float) -> list[list[dict]]:
    """Single-link clustering on bbox proximity. Order-independent."""
    remaining = list(spans)
    clusters: list[list[dict]] = []

    while remaining:
        seed = remaining.pop()
        group = [seed]
        changed = True
        while changed:
            changed = False
            gx0 = min(s["bbox"][0] for s in group) - gap_x
            gy0 = min(s["bbox"][1] for s in group) - gap_y
            gx1 = max(s["bbox"][2] for s in group) + gap_x
            gy1 = max(s["bbox"][3] for s in group) + gap_y
            still = []
            for s in remaining:
                x0, y0, x1, y1 = s["bbox"]
                if x1 >= gx0 and x0 <= gx1 and y1 >= gy0 and y0 <= gy1:
                    group.append(s)
                    changed = True
                else:
                    still.append(s)
            remaining = still
        clusters.append(group)
    return clusters


def bbox_of(group) -> tuple[float, float, float, float]:
    return (min(s["bbox"][0] for s in group), min(s["bbox"][1] for s in group),
            max(s["bbox"][2] for s in group), max(s["bbox"][3] for s in group))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def vertical_slot(span_bbox, bars) -> str:
    """Which side of a fraction bar this span sits on.

    A plain character bag cannot tell Monod-with-biomass from Contois:

        eq4   mu = mu_max * X * S / (Ks + S)      X ABOVE the bar
        eq9   mu = mu_max * S / (Ks * X + S)      X BELOW the bar

    Identical characters, different model. Since those are the rival pair the
    whole project turns on, the dedup key must keep structure. The fraction
    bar is already detected for the display test, so tagging each span by its
    side of the nearest bar costs nothing and needs no parsing.
    """
    x0, y0, x1, y1 = span_bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    best = None
    for r in bars:
        if r.x0 - 1 <= cx <= r.x1 + 1:
            d = abs(r.y0 - cy)
            if best is None or d < best[0]:
                best = (d, r.y0)
    if best is None:
        return "flat"
    return "num" if cy < best[1] else "den"


def _char_bag(slots: list[tuple[str, str]]) -> tuple:
    """Sorted (slot, character) counts, ignoring whitespace.

    Span BOUNDARIES are not stable: the same equation is emitted as
    ['K','s'] in the body and ['Ks'] in a graphical abstract, and raising
    --max-span-len turns ['mu','max','S'] into ['mumaxS']. Comparing spans
    therefore misses real duplicates. Characters are stable.

    The slot tag from vertical_slot() is carried alongside each character so
    that two equations with the same characters in different positions do NOT
    compare equal.
    """
    from collections import Counter as _C
    bag: _C = _C()
    for slot, text in slots:
        for c in text:
            if not c.isspace():
                bag[(slot, c)] += 1
    return tuple(sorted(bag.items()))


def count_equals(glyphs: list[str]) -> int:
    """How many '=' the cluster contains.

    A well-formed display equation has exactly one. Zero means a FRAGMENT:
    an expression quoted mid-sentence, such as '(1 + S/Ki)'. Two or more
    means the clusterer MERGED neighbouring equations, which happens in
    graphical abstracts and summary tables where equations sit close
    together. Both are structural, not layout-specific.
    """
    return "".join(glyphs).count("=")


def deduplicate(found: list[dict]) -> tuple[list[dict], list[dict]]:
    """Collapse crops whose STRUCTURE-TAGGED character bag is identical.

    Papers restate equations: a graphical abstract, a summary table, a
    repeated definition. Those are the same equation typeset twice, and the
    tagged characters are identical regardless of where the copy sits on the
    page or how the typesetter split it into spans. This is
    layout-independent, unlike 'skip page 2', which would be a rule about one
    document.

    Ordering within a slot is ignored on purpose: PDF reading order shuffles
    superscripts and denominators, so the same equation can emit the same
    characters in a different sequence.

    The copy WITH an equation number wins, since a number makes the crop
    traceable and gradeable. Ties keep the earlier page.
    """
    buckets: dict[tuple, list[dict]] = {}
    for e in found:
        key = _char_bag([tuple(s) for s in e["glyph_slots"]])
        buckets.setdefault(key, []).append(e)

    kept, dropped = [], []
    for group in buckets.values():
        group.sort(key=lambda e: (e["eq_number"] is None, e["page"]))
        winner = group[0]
        winner["duplicate_of_count"] = len(group) - 1
        kept.append(winner)
        for e in group[1:]:
            e["duplicate_of"] = {"page": winner["page"],
                                 "eq_number": winner["eq_number"]}
            dropped.append(e)

    kept.sort(key=lambda e: (e["page"], e["bbox"][1]))
    return kept, dropped


def find_equations(path: str, min_spans: int, max_len: int,
                   stack_factor: float) -> tuple[list[dict], dict]:
    doc = pymupdf.open(path)
    m = measure_document(doc)
    body_font, line_h = m["body_font"], m["line_height"]

    found = []
    for pno, page in enumerate(doc, start=1):
        ms = math_spans(page, body_font, max_len)
        if not ms:
            continue
        bars = fraction_bars(page)
        # gaps scaled to the measured line height, so a 9pt paper and a 12pt
        # paper both work without retuning
        groups = cluster(ms, gap_x=line_h * 0.55, gap_y=line_h * 0.35)

        for g in groups:
            if len(g) < min_spans:
                continue
            x0, y0, x1, y1 = bbox_of(g)
            spread = (y1 - y0) / line_h
            rect = pymupdf.Rect(x0, y0, x1, y1)
            has_bar = any(r.intersects(rect) for r in bars)
            alone = is_alone_on_line(page, g, body_font, line_h)

            # THREE independent display tests. A cluster is display math if
            # ANY of them fires, because each catches a case the others miss:
            #   stacked   fractions, sums, anything with a numerator
            #   has_bar   a drawn fraction rule
            #   alone     single-line display math with neither of the above
            # Each was added after a miss. See the module docstring.
            stacked = spread >= stack_factor
            if not (stacked or has_bar or alone):
                continue

            ordered = sorted(g, key=lambda s: (round(s["bbox"][1], 1),
                                               s["bbox"][0]))
            glyphs = [s["text"] for s in ordered]
            slots = [[vertical_slot(s["bbox"], bars), s["text"]]
                     for s in ordered]
            neq = count_equals(glyphs)

            found.append({
                "page": pno, "bbox": [x0, y0, x1, y1],
                "n_spans": len(g), "y_spread_lines": round(spread, 2),
                "stacked": stacked,
                "has_fraction_bar": has_bar,
                "alone_on_line": alone,
                "n_equals": neq,
                "shape": ("fragment" if neq == 0 else
                          "equation" if neq == 1 else "merged"),
                "glyphs": glyphs,
                "glyph_slots": slots,
                "fonts": sorted({s["font"] for s in g}),
            })

    # attach the nearest equation number, if the paper uses them.
    # metadata only: nothing above depends on it.
    for e in found:
        page = doc[e["page"] - 1]
        _, y0, _, y1 = e["bbox"]
        best = None
        for s in body_spans(page, body_font):
            mo = EQNUM.match(s["text"])
            if not mo:
                continue
            sy = (s["bbox"][1] + s["bbox"][3]) / 2
            if y0 - line_h <= sy <= y1 + line_h:
                d = abs(sy - (y0 + y1) / 2)
                if best is None or d < best[0]:
                    best = (d, mo.group(1))
        e["eq_number"] = best[1] if best else None

    kept, dropped = deduplicate(found)
    return kept, dropped, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--outdir", default="equations")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--pad", type=float, default=4.0,
                    help="points of whitespace around each crop")
    ap.add_argument("--min-spans", type=int, default=5,
                    help="fewest math spans for a cluster to be an equation")
    ap.add_argument("--max-span-len", type=int, default=3,
                    help="longest span still treated as math")
    ap.add_argument("--stack-factor", type=float, default=1.3,
                    help="y-extent in line heights above which a cluster is "
                         "DISPLAY rather than inline math")
    ap.add_argument("--no-render", action="store_true",
                    help="detect and report only, write no images")
    ap.add_argument("--shapes", default="equation",
                    help="comma-separated shapes to keep: equation (one '='), "
                         "fragment (none), merged (two or more). "
                         "Use 'all' to keep everything.")
    a = ap.parse_args()

    import os
    os.makedirs(a.outdir, exist_ok=True)

    eqs, dropped, m = find_equations(a.pdf, a.min_spans, a.max_span_len,
                                     a.stack_factor)

    from collections import Counter as _C
    shapes = _C(e["shape"] for e in eqs)
    if a.shapes != "all":
        want = {s.strip() for s in a.shapes.split(",")}
        eqs = [e for e in eqs if e["shape"] in want]

    print(f"{a.pdf}")
    print(f"  body font   : {m['body_font']}")
    print(f"  line height : {m['line_height']:.1f} pt")
    print(f"  top fonts   : {[f for f, _ in m['fonts'][:5]]}")
    print(f"  shapes      : {dict(shapes)}   keeping: {a.shapes}")
    print(f"  equations   : {len(eqs)}  ({len(dropped)} duplicates collapsed)")
    print(f"  numbered    : {sum(1 for e in eqs if e['eq_number'])}\n")

    doc = pymupdf.open(a.pdf)
    for i, e in enumerate(eqs, start=1):
        tag = f"eq{e['eq_number']}" if e["eq_number"] else f"p{e['page']}_{i}"
        e["image"] = f"{a.outdir}/{tag}.png"
        if not a.no_render:
            x0, y0, x1, y1 = e["bbox"]
            clip = pymupdf.Rect(x0 - a.pad, y0 - a.pad, x1 + a.pad, y1 + a.pad)
            doc[e["page"] - 1].get_pixmap(clip=clip, dpi=a.dpi).save(e["image"])
        why = "".join([
            "S" if e["stacked"] else "-",
            "B" if e["has_fraction_bar"] else "-",
            "A" if e["alone_on_line"] else "-",
        ])
        dup = f" x{e['duplicate_of_count']}" if e["duplicate_of_count"] else ""
        print(f"  {tag:10} p{e['page']} spans={e['n_spans']:3} "
              f"why={why} {e['shape']:8}{dup:4} | {' '.join(e['glyphs'])[:50]}")

    print("\n  why: S=stacked  B=fraction bar  A=alone on line")

    with open(f"{a.outdir}/equations.json", "w", encoding="utf-8") as f:
        json.dump({"pdf": a.pdf, "measured": m, "equations": eqs,
                   "duplicates_dropped": dropped}, f,
                  indent=2, ensure_ascii=False)
    print(f"wrote {a.outdir}/equations.json")


if __name__ == "__main__":
    main()
