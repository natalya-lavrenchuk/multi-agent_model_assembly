"""
consolidate.py -- turn N Pass 1 runs of the same input into one symbol table.

WHY VOTE
    Pass 1's quantity_class is an unanchored judgement: class_stated is false
    on essentially every row, meaning the document never says how a quantity
    is obtained, so the class is the model's own guess. Six prompt versions
    did not make that guess stable. Any SINGLE run is therefore a coin flip on
    the rows that vary, and picking one run hides that.

    Voting does not make the class correct. It makes the instability VISIBLE
    and recorded: every consolidated row carries how many runs agreed.

WHAT IT DOES NOT FIX
    A stably wrong row. In prompt version v3, mu_max and Ks were classed
    Constant in 7 runs out of 7. A vote ratifies that unanimously. Voting
    catches noise, never bias. The structural checks in Pass 2 are what catch
    bias.

VOTES ARE ON SYMBOL, NOT local_id
    local_id is assigned per run (q1, q2, ...) and the same symbol lands on
    different ids in different runs.
"""

from __future__ import annotations

import collections
import json


def load_runs(log_path: str, doc_id: str,
              prompt_version: str | None = None,
              input_sha1: str | None = None) -> list[dict]:
    """Successful Pass 1 calls for one document.

    Filtering on prompt_version AND input_sha1 matters: mixing versions or
    inputs produces a vote across runs that were not answering the same
    question, which is how an earlier series became uninterpretable.
    """
    out = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not (r.get("ok") and r.get("pass") == 1):
                continue
            if r.get("doc_id") != doc_id:
                continue
            if prompt_version and r.get("prompt_version") != prompt_version:
                continue
            if input_sha1 and r.get("input_sha1") != input_sha1:
                continue
            out.append(r)
    return out


def consolidate(runs: list[dict]) -> dict:
    """Majority vote per symbol across runs.

    Returns a symbol table plus the evidence for every decision, so a reader
    can see which rows were unanimous and which were 3-2.
    """
    if not runs:
        raise ValueError("no runs to consolidate")

    n = len(runs)
    seen: dict[str, dict] = {}

    for r in runs:
        for q in r["data"]["quantities"]:
            s = q["symbol"]
            e = seen.setdefault(s, {
                "symbol": s,
                "runs_present": 0,
                "classes": collections.Counter(),
                "stated": collections.Counter(),
                "units": collections.Counter(),
                "descriptions": [],
                "spans": [],
            })
            e["runs_present"] += 1
            e["classes"][q["quantity_class"]] += 1
            e["stated"][bool(q["class_stated"])] += 1
            e["units"][q["unit_string"]] += 1
            if q.get("description"):
                e["descriptions"].append(q["description"])
            if q.get("span_text"):
                e["spans"].append(q["span_text"])

    table = []
    for i, (s, e) in enumerate(sorted(seen.items()), start=1):
        cls, cls_votes = e["classes"].most_common(1)[0]
        unit, _ = e["units"].most_common(1)[0]
        stated, stated_votes = e["stated"].most_common(1)[0]

        # longest span wins: a longer verbatim quotation is stronger evidence
        # and is what the span gate will be checked against.
        span = max(e["spans"], key=len) if e["spans"] else ""
        desc = max(e["descriptions"], key=len) if e["descriptions"] else ""

        table.append({
            "local_id": f"c{i}",
            "symbol": s,
            "description": desc,
            "unit_string": unit,
            "quantity_class": cls,
            "class_stated": stated,
            "span_text": span,
            # evidence, so a 3-2 decision never looks like a 5-0 one
            "n_runs": n,
            "runs_present": e["runs_present"],
            "class_votes": dict(e["classes"]),
            "class_margin": f"{cls_votes}/{e['runs_present']}",
            "unanimous": len(e["classes"]) == 1,
            "always_present": e["runs_present"] == n,
        })

    return {
        "n_runs": n,
        "doc_id": runs[0]["doc_id"],
        "prompt_version": runs[0]["prompt_version"],
        "input_sha1": runs[0]["input_sha1"],
        "model": runs[0]["model"],
        "quantities": table,
    }


def report(cons: dict) -> str:
    lines = [f"{cons['doc_id']}  n={cons['n_runs']}  "
             f"{cons['prompt_version']}  sha={cons['input_sha1']}"]
    for q in cons["quantities"]:
        marks = []
        if not q["unanimous"]:
            marks.append(f"SPLIT {q['class_votes']}")
        if not q["always_present"]:
            marks.append(f"only in {q['runs_present']}/{q['n_runs']} runs")
        lines.append(f"  {q['local_id']:4} {q['symbol']:8} "
                     f"{q['quantity_class']:15} {q['class_margin']:6} "
                     + ("  ".join(marks)))
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id")
    ap.add_argument("--log", default="runs.jsonl")
    ap.add_argument("--prompt-version", default=None)
    ap.add_argument("--input-sha1", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = load_runs(a.log, a.doc_id, a.prompt_version, a.input_sha1)
    if not runs:
        raise SystemExit(f"no runs for {a.doc_id}")
    shas = {r["input_sha1"] for r in runs}
    vers = {r["prompt_version"] for r in runs}
    if len(shas) > 1 or len(vers) > 1:
        raise SystemExit(
            f"runs are not comparable: inputs={shas} versions={vers}. "
            f"Pass --input-sha1 and --prompt-version to pick one set.")

    cons = consolidate(runs)
    print(report(cons))
    if a.out:
        json.dump(cons, open(a.out, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        print(f"\nwrote {a.out}")
