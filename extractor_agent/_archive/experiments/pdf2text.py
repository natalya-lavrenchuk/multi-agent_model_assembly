"""
pdf2text.py -- turn a paper PDF into the plain text the extractor reads.

    pip install pymupdf          (best; falls back to pypdf or pdfplumber)
    python pdf2text.py paper.pdf -o paper.txt

Why this and not copy-paste: the span gate checks that every quoted sentence
appears VERBATIM in the text you gave the model. So the text file has to be
the exact thing the model saw. Copy-pasting from a browser silently changes
whitespace and ligatures, and then real spans start failing the gate.

What it does:
  - extracts text page by page
  - marks page boundaries so span_locator can say "p. 4"
  - joins words broken across line ends by a hyphen
  - optionally drops the references section, which is noise for extraction
"""

from __future__ import annotations

import argparse
import re
import sys


BACKENDS = ("pymupdf", "pypdf", "pdfplumber")


def pick_backend() -> str | None:
    """Use whichever PDF library is installed. pymupdf gives the best text."""
    for name in BACKENDS:
        try:
            if name == "pymupdf":
                import pymupdf  # noqa: F401
            elif name == "pypdf":
                import pypdf  # noqa: F401
            else:
                import pdfplumber  # noqa: F401
            return name
        except ImportError:
            continue
    return None


def extract_pages(path: str, backend: str) -> list[str]:
    """Return one string per page."""
    if backend == "pymupdf":
        import pymupdf
        with pymupdf.open(path) as doc:
            return [p.get_text() or "" for p in doc]

    if backend == "pypdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        return [(p.extract_text() or "") for p in reader.pages]

    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return [(p.extract_text() or "") for p in pdf.pages]


def clean(text: str) -> str:
    # join hyphenated words split across lines: "disper-\nsion" -> "dispersion"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip trailing spaces
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text


def drop_references(text: str) -> tuple[str, bool]:
    """Cut everything from the references heading onward."""
    pat = re.compile(r"^\s*(References|Bibliography|REFERENCES)\s*$",
                     re.MULTILINE)
    matches = list(pat.finditer(text))
    if not matches:
        return text, False
    # use the LAST match, in case "References" appears in a TOC
    return text[:matches[-1].start()], True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--keep-refs", action="store_true",
                    help="keep the references section (dropped by default)")
    ap.add_argument("--pages", default=None,
                    help="page range, e.g. 1-6 (1-based, inclusive)")
    a = ap.parse_args()

    backend = pick_backend()
    if backend is None:
        print("No PDF library found. Install one of these:", file=sys.stderr)
        print("    pip install pymupdf        <- recommended, no compiler needed",
              file=sys.stderr)
        print("    pip install pypdf          <- pure python, also fine",
              file=sys.stderr)
        return 1

    lo, hi = 1, 10**6
    if a.pages:
        parts = a.pages.split("-")
        lo = int(parts[0])
        hi = int(parts[1]) if len(parts) > 1 else lo

    pages = extract_pages(a.pdf, backend)
    total = len(pages)
    chunks = [f"\n[page {i}]\n{t}"
              for i, t in enumerate(pages, start=1) if lo <= i <= hi]
    text = clean("\n".join(chunks))

    dropped = False
    if not a.keep_refs:
        text, dropped = drop_references(text)

    out = a.out or (a.pdf.rsplit(".", 1)[0] + ".txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

    words = len(text.split())
    print(f"{a.pdf}: {total} pages   (extracted with {backend})")
    print(f"wrote {out}   {len(text):,} characters   ~{words:,} words")
    if dropped:
        print("dropped the references section (use --keep-refs to keep it)")
    if words > 12000:
        print("\nNOTE: that is a long document. If the model times out or "
              "misses things,\n      use --pages to feed it the modelling "
              "section only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
