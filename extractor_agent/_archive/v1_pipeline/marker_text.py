"""
marker_text.py -- turn Marker JSON into per-section text files for Pass 1.

REPLACES hand-picked line ranges. Previously sections were sliced by reading
line numbers out of a pdf2text dump by eye, which does not scale past one
paper and silently truncates when a window clips a definition list. Marker
gives headings, so sections are cut where the document says they are.

WHY PER SECTION, NOT THE WHOLE PAPER
    Muloiwa is ~11,400 words, about 16k tokens, against a per-minute cap of
    8,000. One call cannot hold it. A model section is 700-1,500 words, about
    3k tokens with the prompt, which fits with headroom for a retry in the
    same minute.

WHAT IS INCLUDED
    Text blocks only. Equation blocks are EXCLUDED: their text layer is glyph
    soup ('mu muS KS max s'), and feeding it to Pass 1 invites the model to
    read structure that is not there. Pass 2 reads equations, from the image.
    Inline math inside a Text block is kept, because that is where 'Where, mu
    specific growth rate' lives and that list is Pass 1's primary source.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re

_TAG = re.compile(r"<[^>]+>")
_SLUG = re.compile(r"[^a-z0-9]+")


def strip_html(s: str) -> str:
    """Tags out, entities decoded.

    Entity decoding is not cosmetic: Marker emits Greek as '&#956;', and an
    undecoded entity would make every span containing 'mu' fail the verbatim
    check in Pass 1's gate.
    """
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", s or ""))).strip()


def slug(s: str) -> str:
    return _SLUG.sub("-", s.lower()).strip("-")[:40] or "untitled"


def flatten(doc: dict) -> list[dict]:
    """Every block in reading order, with its page index."""
    out = []
    for pi, page in enumerate(doc.get("children") or []):
        def walk(n):
            if isinstance(n, dict):
                if n.get("block_type"):
                    out.append({"page": pi, "b": n})
                for c in (n.get("children") or []):
                    walk(c)
            elif isinstance(n, list):
                for c in n:
                    walk(c)
        walk(page)
    return out


def split_sections(doc: dict, include_equations: bool = False) -> list[dict]:
    """Cut at every SectionHeader. Text before the first header is 'front'."""
    sections = [{"title": None, "page": 0, "paras": []}]

    for item in flatten(doc):
        b, pi = item["b"], item["page"]
        bt = b.get("block_type")

        if bt == "SectionHeader":
            title = strip_html(b.get("html", ""))
            if title:
                sections.append({"title": title, "page": pi, "paras": []})
            continue

        keep = ("Text",) if not include_equations else ("Text", "Equation")
        if bt in keep:
            t = strip_html(b.get("html", ""))
            if t:
                sections[-1]["paras"].append(t)

    return [s for s in sections if s["paras"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("marker_json")
    ap.add_argument("--outdir", default="sections")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--min-words", type=int, default=40,
                    help="skip sections shorter than this (headers, captions)")
    ap.add_argument("--max-words", type=int, default=2000,
                    help="warn above this: the call may exceed the token cap")
    ap.add_argument("--include-equations", action="store_true",
                    help="also write Equation blocks. Off by default: their "
                         "text layer is glyph soup and Pass 2 reads equations "
                         "from the image instead.")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    doc = json.load(open(a.marker_json, encoding="utf-8"))
    sections = split_sections(doc, a.include_equations)

    manifest = []
    print(f"{len(sections)} sections with text\n")
    for i, s in enumerate(sections):
        text = "\n\n".join(s["paras"])
        words = len(text.split())
        title = s["title"] or "front-matter"
        name = f"{i:02d}-{slug(title)}"
        path = os.path.join(a.outdir, name + ".txt")

        flag = ""
        if words < a.min_words:
            flag = "  SKIP (too short)"
        elif words > a.max_words:
            flag = "  WARNING: may exceed token cap, consider splitting"

        print(f"  {name:44} p{s['page']+1:<3} {words:5} words{flag}")
        if words < a.min_words:
            continue

        open(path, "w", encoding="utf-8").write(text)
        manifest.append({
            "file": path,
            "section": title,
            "page": s["page"] + 1,
            "words": words,
            "sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
            "doc_id": f"{a.doc_id}#{slug(title)}",
        })

    mpath = os.path.join(a.outdir, "sections.json")
    json.dump({"doc_id": a.doc_id, "source": a.marker_json,
               "sections": manifest}, open(mpath, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote {len(manifest)} files + {mpath}")
    print(f"estimated cost: ~{sum(m['words'] for m in manifest) * 1.4 / 1000:.0f}k "
          f"tokens for one run over all sections")


if __name__ == "__main__":
    main()
