"""
cons_to_turtle.py -- write the VOTED Pass 1 table (consolidate.py output) as
paper-graph Turtle.

Why this exists: pass1.py writes Turtle for ONE run. Pass 2 binds against the
voted table. So the quantities Pass 2's laws point at were never in any graph.
This writes exactly the table Pass 2 saw, with the same IRIs Pass 2 uses.

No tokens. The deterministic Pass 1 gate is re-run against the section text,
so the voted rows face the same checks a single run does.

--min-presence drops rows that appeared in too few runs (OD entered on 1 of 5
runs of the Monod section). Default 3 of 5: a majority.

Usage:
  python cons_to_turtle.py cons_monod.json sections/monod.txt \
      --doc-id "10.1016/j.sajce.2020.07.006#monod-model" --out q_monod.ttl
"""

from __future__ import annotations

import argparse
import json

from pass1 import gate, to_turtle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cons_json")
    ap.add_argument("section_text")
    ap.add_argument("--doc-id", default=None,
                    help="defaults to the doc_id recorded in the cons file")
    ap.add_argument("--min-presence", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cons = json.load(open(a.cons_json, encoding="utf-8"))
    doc_id = a.doc_id or cons["doc_id"]
    text = open(a.section_text, encoding="utf-8").read()

    kept, dropped = [], []
    for q in cons["quantities"]:
        if q.get("runs_present", cons["n_runs"]) < a.min_presence:
            dropped.append(f"{q['symbol']} ({q['runs_present']}/{cons['n_runs']})")
            continue
        kept.append({
            "local_id": q["local_id"],
            "symbol": q["symbol"],
            "description": q["description"],
            "unit_string": q["unit_string"],
            "quantity_class": q["quantity_class"],
            "class_stated": q["class_stated"],
            "index_over": q.get("index_over", []),
            "span_text": q["span_text"],
            "span_locator": q.get("span_locator", ""),
        })

    gated = gate(kept, text)
    prov = {"model": cons.get("model", "unknown"),
            "decoding_tier": f"voted{cons['n_runs']}",
            "prompt_version": cons.get("prompt_version", "unknown")}
    open(a.out, "w", encoding="utf-8").write(to_turtle(gated, doc_id, prov))

    acc = sum(g.status == "accepted" for g in gated)
    print(f"{len(gated)} quantities written, {acc} accepted -> {a.out}")
    for g in gated:
        if g.failed:
            print(f"  FLAG {g.record['symbol']}: {g.failed}")
    if dropped:
        print(f"  dropped below min-presence {a.min_presence}: {dropped}")


if __name__ == "__main__":
    main()
