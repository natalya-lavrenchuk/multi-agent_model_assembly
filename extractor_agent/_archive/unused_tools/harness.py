"""
harness.py -- the wrapper around every LLM call in the extractor.

WHAT THIS SOLVES
----------------
When you ask a model for JSON there are three ways it can come back:

  strict  - the server forced the model to follow your schema exactly.
  loose   - the server made it valid JSON but ignored your schema.
  text    - you got prose and have to dig the JSON out yourself.

Servers silently drop from strict to loose to text. If you don't record which
one you got, a bad extraction looks like the model being bad, when actually
you were never in strict mode. That mistake costs days.

This harness always records what it actually got. Every result carries a
`tier` field. If you ever see `tier="text"` in your results, that run is not
evidence about the model.

USAGE
-----
    h = Harness(base_url="http://localhost:8000/v1", model="gpt-oss-120b")
    result = h.call(
        system="You extract nomenclature from papers.",
        user=paper_text,
        schema=my_json_schema,
        schema_name="pass1_nomenclature",
    )
    if result.ok:
        print(result.data, result.tier)
"""

from __future__ import annotations

import json
import re
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any

import jsonschema
import urllib.request
import urllib.error


# ----------------------------------------------------------------------
# Plain-language diagnosis of the errors that are not the model's fault
# ----------------------------------------------------------------------

def _diagnose(code: int, body: str) -> str:
    b = body.lower()
    if "1010" in b or "browser" in b:
        return ("HINT: Cloudflare blocked the request on its browser signature "
                "(error 1010). This is NOT your API key and NOT the model. "
                "The User-Agent header is missing or looks automated.")
    if code == 401 or "invalid api key" in b or "unauthorized" in b:
        return ("HINT: the API key was rejected. Check GROQ_API_KEY is set in "
                "THIS shell (echo $GROQ_API_KEY) and has no quotes or spaces.")
    if code == 403:
        return ("HINT: 403 without a Cloudflare code usually means the key is "
                "valid but not allowed to use this model. Check the model name.")
    return f"HINT: HTTP {code}, see body above."


# ----------------------------------------------------------------------
# Result object
# ----------------------------------------------------------------------

@dataclass
class CallResult:
    ok: bool
    data: Any | None                 # the parsed JSON, if we got any
    tier: str                        # "strict" | "loose" | "text" | "failed"
    attempts: int
    errors: list[str] = field(default_factory=list)
    raw: str = ""                    # the raw model output, always kept
    model: str = ""
    prompt_version: str = ""
    elapsed_s: float = 0.0

    def provenance(self) -> dict:
        """The block that goes into every candidate record."""
        return {
            "model": self.model,
            "decoding_tier": self.tier,
            "prompt_version": self.prompt_version,
        }


# ----------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------

class Harness:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        timeout: int = 600,
        max_attempts: int = 3,
        max_rate_waits: int = 6,
        user_agent: str = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0 Safari/537.36"),
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.max_rate_waits = max_rate_waits
        self.user_agent = user_agent

    # -- low-level HTTP -------------------------------------------------

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                # Cloudflare blocks the default Python-urllib agent (error 1010).
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    # -- the three tiers ------------------------------------------------

    def _payload(self, system: str, user: str, tier: str,
                 schema: dict, schema_name: str) -> dict:
        p = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if tier == "strict":
            # vLLM / OpenAI structured output. The server enforces the schema.
            p["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            }
        elif tier == "loose":
            # Valid JSON guaranteed, schema NOT enforced.
            p["response_format"] = {"type": "json_object"}
        # tier == "text": no response_format at all.
        return p

    # -- JSON recovery from prose ---------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Any | None:
        """Last resort: pull the first balanced {...} out of prose."""
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
                      flags=re.MULTILINE)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        if start == -1:
            return None
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    # -- the main entry point -------------------------------------------

    def call(
        self,
        system: str,
        user: str,
        schema: dict,
        schema_name: str = "record",
        prompt_version: str = "v1",
    ) -> CallResult:
        t0 = time.time()
        errors: list[str] = []
        attempts = 0
        waits = 0
        raw = ""

        for tier in ("strict", "loose", "text"):
            tries = 0
            while tries < self.max_attempts:
                tries += 1
                attempts += 1
                try:
                    resp = self._post(
                        self._payload(system, user, tier, schema, schema_name)
                    )
                except urllib.error.HTTPError as e:
                    body = e.read().decode("utf-8", "replace")[:300]
                    errors.append(f"[{tier}] HTTP {e.code}: {body}")
                    # Auth or edge-blocked. Retrying and changing tier cannot help.
                    if e.code in (401, 403):
                        hint = _diagnose(e.code, body)
                        return CallResult(
                            ok=False, data=None, tier="blocked",
                            attempts=attempts, errors=errors + [hint],
                            raw="", model=self.model,
                            prompt_version=prompt_version,
                            elapsed_s=round(time.time() - t0, 2),
                        )
                    # A 400 usually means this tier is unsupported. Move on.
                    if e.code == 400:
                        break
                    # 413: the prompt is too big. Retrying or changing tier cannot help.
                    if e.code == 413:
                        return CallResult(
                            ok=False, data=None, tier="failed", attempts=attempts,
                            errors=errors + ["HINT: request too large; send fewer symbols per call"],
                            raw="", model=self.model, prompt_version=prompt_version,
                            elapsed_s=round(time.time() - t0, 2),
                        )
                    # 429: rate limit. Wait as long as the server asks and retry the SAME tier.
                    # Falling back to a looser tier here would silently degrade decoding for a
                    # reason that has nothing to do with the schema.
                    if e.code == 429 and waits < self.max_rate_waits:
                        waits += 1
                        m = re.search(r"try again in ([\d.]+)(ms|s)", body)
                        wait = (float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)) if m else 20.0
                        time.sleep(min(wait + 1.0, 65.0))
                        tries -= 1             # a rate-limit wait does not use up an attempt
                        continue
                    time.sleep(2)
                    continue
                except Exception as e:
                    errors.append(f"[{tier}] {type(e).__name__}: {e}")
                    time.sleep(2)
                    continue

                raw = resp["choices"][0]["message"]["content"] or ""
                data = self._extract_json(raw)
                if data is None:
                    errors.append(f"[{tier}] no JSON in output")
                    continue

                try:
                    jsonschema.validate(data, schema)
                except jsonschema.ValidationError as e:
                    path = "/".join(str(p) for p in e.absolute_path) or "<root>"
                    errors.append(f"[{tier}] schema: {path}: {e.message[:160]}")
                    continue

                return CallResult(
                    ok=True, data=data, tier=tier, attempts=attempts,
                    errors=errors, raw=raw, model=self.model,
                    prompt_version=prompt_version,
                    elapsed_s=round(time.time() - t0, 2),
                )

        return CallResult(
            ok=False, data=None, tier="failed", attempts=attempts,
            errors=errors, raw=raw, model=self.model,
            prompt_version=prompt_version,
            elapsed_s=round(time.time() - t0, 2),
        )


# ----------------------------------------------------------------------
# Run logging -- one JSONL line per call, so nothing is lost
# ----------------------------------------------------------------------

def log_call(path: str, doc_id: str, pass_no: int, result: CallResult,
             user_prompt: str) -> None:
    rec = asdict(result)
    rec["doc_id"] = doc_id
    rec["pass"] = pass_no
    rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    # keep a fingerprint of the input so you can tell runs apart
    rec["input_sha1"] = hashlib.sha1(user_prompt.encode()).hexdigest()[:12]
    rec["input_chars"] = len(user_prompt)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def tier_report(log_path: str) -> dict[str, int]:
    """Read the log back and count tiers. Run this after EVERY experiment."""
    counts: dict[str, int] = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["tier"]
            counts[t] = counts.get(t, 0) + 1
    return counts
