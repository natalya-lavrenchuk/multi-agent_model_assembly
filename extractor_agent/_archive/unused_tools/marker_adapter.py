"""
marker_adapter.py -- build Pass 2's input from Marker JSON + pix2tex LaTeX + PDF.

Pass 2 needs, per equation, THREE INDEPENDENT READINGS of the same region so
that disagreement between them is the gate:

    latex         from pix2tex, which reads the IMAGE. Recovers structure
                  (\\frac, superscripts) that the PDF never encoded. Its known
                  failure is dropping subscripts: K_s -> 'K.', 'R.'.
    glyph_slots   from pymupdf, which reads the TEXT LAYER. Exact characters
                  (so it has the 's' pix2tex dropped), but no structure except
                  what fraction-bar geometry recovers. Fails completely when
                  the PDF's ToUnicode map is broken (Springer: '=' as '\u00bc').
    text_before / from Marker's neighbouring Text blocks. This is where the
    text_after    law's NAME, its citation and its scope conditions live:
                  "The Contois model can be defined by equation 9."

No single source is trusted. The gate is agreement.

SECTION TITLES ARE A FALLBACK, NOT A PRIMARY NAME SOURCE.
Marker resolves a section hierarchy, and in a REVIEW paper each model happens
to get its own heading. That is real information and should not be thrown
away. But a primary paper puts eight equations under one 'Model development'
heading, where the title would name them all identically.

So Pass 2 records law_name_source alongside the name:
    "prose"    extracted from text_before/text_after WITH a verbatim span.
               Gated the same way Pass 1 gates spans.
    "section"  taken from the enclosing heading. No span, because a heading
               is not a sentence about this equation. Weaker evidence, kept
               because it is usually right and always checkable by a human.
    null       neither available. An anonymous law is honest; the assembler
               names it later against the graph.

The distinction is the point. A name with a span and a name from a heading
must not look identical in the graph.
"""

from __future__ import annotations

import argparse
import html
import json
import re


# ----------------------------------------------------------------------
# Marker JSON
# ----------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")


def strip_html(s: str) -> str:
    """Tags out, entities decoded. Marker emits Greek as '&#956;', and an
    undecoded entity would fail every verbatim span check downstream."""
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", s or ""))).strip()


def index_sections(doc: dict) -> dict[str, str]:
    """Map every SectionHeader's id to its title text."""
    out: dict[str, str] = {}

    def walk(n):
        if isinstance(n, dict):
            if n.get("block_type") == "SectionHeader":
                title = strip_html(n.get("html", ""))
                if n.get("id") and title:
                    out[n["id"]] = title
            for c in (n.get("children") or []):
                walk(c)
        elif isinstance(n, list):
            for c in n:
                walk(c)

    walk(doc)
    return out


def deepest_section(hierarchy: dict | None, titles: dict[str, str]) -> str | None:
    """Title of the most specific enclosing section.

    Levels are string keys ('1', '2', '3'). The highest number is the deepest
    heading, which is the one that names the law.
    """
    if not hierarchy:
        return None
    try:
        key = max(hierarchy, key=lambda k: int(k))
    except ValueError:
        key = sorted(hierarchy)[-1]
    return titles.get(hierarchy[key])


def collect_equations(doc: dict, titles: dict[str, str],
                      n_context: int = 2) -> list[dict]:
    """Equation blocks with their neighbouring Text blocks, in reading order.

    The surrounding prose is the payload, not decoration. It carries:
      - the law's name          "The Contois model can be defined by eq 9"
      - the citation            "(Contois, 1959)"
      - scope conditions        "cannot describe the lag and death phase"
    Pass 2 extracts the name from here with a span. Pass 4 will need the same
    blocks for scope.

    Pages are the direct children of Document and are iterated explicitly. A
    counter threaded through recursive calls does not propagate back up, which
    silently tags every block with page 1.
    """
    flat: list[dict] = []
    for pi, page in enumerate(doc.get("children") or []):
        page_blocks: list[dict] = []

        def walk(n):
            if isinstance(n, dict):
                bt = n.get("block_type")
                if bt in ("Equation", "ChemicalBlock", "Text", "SectionHeader",
                          "Caption", "Footnote"):
                    page_blocks.append(n)
                for c in (n.get("children") or []):
                    walk(c)
            elif isinstance(n, list):
                for c in n:
                    walk(c)

        walk(page)
        for b in page_blocks:
            flat.append({"page_index": pi, "page_bbox": page.get("bbox"),
                         "block": b})

    def prose(i: int, step: int, k: int) -> list[str]:
        """Up to k Text blocks walking outward from position i."""
        out, j = [], i + step
        while 0 <= j < len(flat) and len(out) < k:
            b = flat[j]["block"]
            bt = b.get("block_type")
            if bt in ("Equation", "ChemicalBlock"):
                break                      # stop at the next equation
            if bt == "Text":
                t = strip_html(b.get("html", ""))
                if t:
                    out.append(t)
            j += step
        return out[::-1] if step < 0 else out

    eqs = []
    for i, item in enumerate(flat):
        b = item["block"]
        if b.get("block_type") not in ("Equation", "ChemicalBlock"):
            continue
        eqs.append({
            "block_id": b.get("id"),
            "block_type": b.get("block_type"),
            "page_index": item["page_index"],
            "bbox_marker": b.get("bbox"),
            "page_bbox_marker": item["page_bbox"],
            # context only -- never used as the law's name, see module docstring
            "section_context": deepest_section(b.get("section_hierarchy"), titles),
            "text_layer": strip_html(b.get("html", "")),
            "html": b.get("html", ""),
            "text_before": prose(i, -1, n_context),
            "text_after": prose(i, +1, n_context),
        })
    return eqs


# ----------------------------------------------------------------------
# PDF glyphs, for the deterministic side of the gate
# ----------------------------------------------------------------------

def attach_glyphs(eqs: list[dict], pdf_path: str) -> None:
    """Characters and fraction-bar slots from the PDF's own text layer.

    Marker's bbox may be in a different coordinate space from the PDF's
    points, so it is rescaled by the ratio of page boxes. If the ratio is 1.0
    the rescale is a no-op.
    """
    import pymupdf
    from core.eqfind import fraction_bars, vertical_slot

    doc = pymupdf.open(pdf_path)

    for e in eqs:
        pi = e["page_index"]
        if pi >= doc.page_count:
            e["glyph_slots"], e["bbox_pdf"] = [], None
            continue

        page = doc[pi]
        pr = page.rect
        pb = e.get("page_bbox_marker")
        if pb and (pb[2] - pb[0]) and (pb[3] - pb[1]):
            sx = pr.width / (pb[2] - pb[0])
            sy = pr.height / (pb[3] - pb[1])
        else:
            sx = sy = 1.0

        x0, y0, x1, y1 = e["bbox_marker"]
        rect = pymupdf.Rect(x0 * sx - 2, y0 * sy - 2, x1 * sx + 2, y1 * sy + 2)
        e["bbox_pdf"] = [rect.x0, rect.y0, rect.x1, rect.y1]

        bars = [r for r in fraction_bars(page) if r.intersects(rect)]
        slots = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for s in line["spans"]:
                    t = s["text"].strip()
                    if not t:
                        continue
                    sb = s["bbox"]
                    cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
                    if rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1:
                        slots.append([vertical_slot(sb, bars), t])
        e["glyph_slots"] = slots
        e["glyphs"] = [t for _, t in slots]


# ----------------------------------------------------------------------
# pix2tex output
# ----------------------------------------------------------------------

_MATH_TAG = re.compile(r"<math[^>]*>(.*?)</math>", re.S | re.I)
_EQ_NUMBER = re.compile(r"(\\q?quad|\\,|\\;|\s)*\(\s*\d{1,3}\s*\)\s*$")


def latex_from_marker_html(html_str: str) -> str | None:
    """LaTeX out of a Marker Equation block, when Marker's VLM read it.

    In balanced mode Marker puts the recognised LaTeX inside <math> tags. If
    there is no <math> tag, the VLM did not run on this block and the html is
    just the raw text layer ('mu muS K max s'), which is NOT LaTeX. That case
    returns None rather than passing glyph soup to Pass 2 as if it were an
    equation. Blackman (eq 5) came back this way.

    The trailing equation number is removed: '\\quad (9)' is layout, not
    mathematics, and left in it becomes a stray symbol for SymPy.
    """
    if not html_str:
        return None
    m = _MATH_TAG.findall(html_str)
    if not m:
        return None
    tex = html.unescape(" ".join(m)).strip()
    tex = _EQ_NUMBER.sub("", tex).strip()
    return tex or None


def attach_latex_from_marker(eqs: list[dict]) -> None:
    for e in eqs:
        e["latex"] = latex_from_marker_html(e.get("html", ""))
        e["latex_source"] = "marker" if e["latex"] else None


def attach_latex(eqs: list[dict], latex_path: str | None) -> None:
    """Merge pix2tex LaTeX, matched BY ORDER.

    Order-matching is fragile: it assumes the LaTeX file was produced from
    the same Marker JSON, walking pages in the same sequence. The count is
    checked and a mismatch is fatal rather than silently misaligning
    equations, which is the failure that produced 17 blocks all tagged
    'page 1' earlier.
    """
    if not latex_path:
        for e in eqs:
            e["latex"] = None
        return

    data = json.load(open(latex_path, encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("equations") or data.get("blocks") or []

    if len(data) != len(eqs):
        raise SystemExit(
            f"LaTeX file has {len(data)} entries, Marker JSON has {len(eqs)} "
            f"equations. Refusing to match by order. Regenerate the LaTeX from "
            f"this Marker JSON, or add block_id to the LaTeX entries."
        )

    for e, d in zip(eqs, data):
        e["latex"] = d.get("latex") if isinstance(d, dict) else str(d)


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("marker_json")
    ap.add_argument("pdf")
    ap.add_argument("--latex", default=None,
                    help="pix2tex output, JSON list in the same order as the "
                         "Marker equation blocks")
    ap.add_argument("--latex-from-marker", action="store_true",
                    help="take LaTeX from Marker's own <math> tags (balanced "
                         "mode). Use this instead of --latex.")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--context", type=int, default=4,
                    help="how many neighbouring Text blocks to attach on each "
                         "side. A law's name is often stated a paragraph or "
                         "two away from its equation, not adjacent to it.")
    ap.add_argument("--out", default="pass2_input.json")
    a = ap.parse_args()

    doc = json.load(open(a.marker_json, encoding="utf-8"))
    titles = index_sections(doc)
    eqs = collect_equations(doc, titles, n_context=a.context)
    attach_glyphs(eqs, a.pdf)
    if a.latex_from_marker:
        attach_latex_from_marker(eqs)
    else:
        attach_latex(eqs, a.latex)

    print(f"{len(titles)} section headers, {len(eqs)} equation blocks\n")
    for i, e in enumerate(eqs, 1):
        print(f"  {i:3} p{e['page_index']+1} {e['block_type']:13} "
              f"[{str(e['section_context'])[:24]}]")
        if e.get("latex"):
            print(f"       tex   : {e['latex'][:70]}")
        elif e["block_type"] == "Equation":
            print(f"       tex   : NONE -- VLM did not read this block")
        for j, t in enumerate(e["text_before"]):
            print(f"       bef{j} : ...{t[-80:]}")
        for j, t in enumerate(e["text_after"]):
            print(f"       aft{j} : {t[:80]}...")
        if not e["glyph_slots"]:
            print(f"       WARNING: no glyphs in bbox -- check bbox scaling")
        if not e["text_before"] and not e["text_after"]:
            print(f"       WARNING: no surrounding prose -- law_name will be null")

    json.dump({"doc_id": a.doc_id, "pdf": a.pdf, "equations": eqs},
              open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
