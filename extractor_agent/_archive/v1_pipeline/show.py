"""
show.py -- print the last Pass 1 run in a form you can actually read.

    python show.py cadet_config.txt
    python show.py cadet_config.txt --flagged      # only the problems
    python show.py cadet_config.txt --json         # raw model output

Reads runs.jsonl (written by pass1.py), takes the most recent successful run,
re-applies the gate against the source text, and prints a table.
"""

import argparse
import json
import sys

from pass1 import gate
from units import parse_unit


def load_last_run(log_path: str):
    last = None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # skip a truncated or malformed line
            if rec.get("ok") and rec.get("data"):
                last = rec
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("textfile", help="the same document you ran pass1.py on")
    ap.add_argument("--log", default="runs.jsonl")
    ap.add_argument("--flagged", action="store_true", help="show only flagged rows")
    ap.add_argument("--json", action="store_true", help="dump the raw model output")
    a = ap.parse_args()

    run = load_last_run(a.log)
    if run is None:
        print(f"No successful run found in {a.log}")
        return 1

    if a.json:
        print(json.dumps(run["data"], indent=2, ensure_ascii=False))
        return 0

    source = open(a.textfile, encoding="utf-8").read()
    gated = gate(run["data"]["quantities"], source)

    print(f"document : {run['doc_id']}")
    print(f"run      : {run['timestamp']}   model={run['model']}   "
          f"tier={run['tier']}   {run['elapsed_s']}s")
    print()

    rows = [g for g in gated if (g.failed or not a.flagged)]

    hdr = f"{'':2s} {'SYMBOL':<22} {'UNIT AS PRINTED':<28} {'DIMENSIONS':<22} {'CLASS':<16} SRC"
    print(hdr)
    print("-" * len(hdr))

    for g in rows:
        r, p = g.record, g.parse
        mark = "  " if g.status == "accepted" else "!!"
        unit = r["unit_string"] if r["unit_string"] is not None else "(none given)"
        if p is not None and p.ok:
            dims = " ".join(f"{k}{v:+d}" for k, v in sorted(p.dims.items())) or "dimensionless"
        else:
            dims = "-- will not parse --"
        src = "stated" if r["class_stated"] else "guessed"
        print(f"{mark} {r['symbol']:<22} {unit[:28]:<28} {dims:<22} "
              f"{r['quantity_class']:<16} {src}")
        if g.failed:
            print(f"     -> {', '.join(g.failed)}")
            print(f"     -> says: {r['description'][:90]}")
            print(f"     -> span: {r['span_text'][:90]!r}")

    n, acc = len(gated), sum(1 for g in gated if g.status == "accepted")
    print()
    print(f"{n} quantities   {acc} accepted   {n - acc} flagged")

    guessed = sum(1 for g in gated if not g.record["class_stated"])
    print(f"{guessed}/{n} classes were guessed by the model, not stated in the document")

    if not a.flagged and any(g.failed for g in gated):
        print("\n(run with --flagged to see only the problems)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
