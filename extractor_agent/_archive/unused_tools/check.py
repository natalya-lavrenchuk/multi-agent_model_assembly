"""
check.py -- run this BEFORE pass1.py to prove the connection works.

    python check.py

It answers three questions in order:
  1. can we reach the API at all?
  2. is the key accepted and the model name valid?
  3. does the server actually support strict JSON schema mode?

Each one has a different fix, so it tells you which one failed.
"""

import os
import sys

from core.harness import Harness

BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY")

TINY_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string"},
        "unit_string": {"type": "string"},
    },
    "required": ["symbol", "unit_string"],
    "additionalProperties": False,
}


def main() -> int:
    print(f"base_url : {BASE_URL}")
    print(f"model    : {MODEL}")
    if not KEY:
        print("\nFAIL: no API key found.")
        print("  Set it in THIS shell, then re-run:")
        print("    export GROQ_API_KEY=gsk_...        (mac/linux)")
        print('    $env:GROQ_API_KEY="gsk_..."        (windows powershell)')
        return 1
    print(f"key      : {KEY[:7]}...{KEY[-4:]} ({len(KEY)} chars)")

    h = Harness(base_url=BASE_URL, model=MODEL, api_key=KEY, max_attempts=1)
    r = h.call(
        system="You extract nomenclature. Return JSON only.",
        user='From this line, extract the symbol and its unit:\n'
             '"D_ax  axial dispersion coefficient, m2 s-1"',
        schema=TINY_SCHEMA,
        schema_name="tiny",
    )

    print(f"\nresult   : ok={r.ok}  tier={r.tier}  attempts={r.attempts}  "
          f"{r.elapsed_s}s")
    if r.data:
        print(f"data     : {r.data}")
    for e in r.errors:
        print(f"  {e}")

    if not r.ok:
        print("\nFAIL. Nothing above is a model problem. Fix the connection first.")
        return 1

    if r.tier == "strict":
        print("\nPASS. Strict schema mode works. You are ready to run pass1.py.")
        return 0

    print(f"\nPARTIAL. Connection works but you landed in '{r.tier}' mode, "
          "not 'strict'.")
    print("  The model was NOT constrained by the schema, so extraction results")
    print("  from this setup are not evidence about the model.")
    print("  Try a different model name, or accept it and record the tier.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
