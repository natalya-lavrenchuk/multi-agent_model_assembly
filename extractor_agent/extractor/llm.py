"""llm.py -- every LLM call of the pipeline goes through ask().

    LLM_BASE_URL   default https://api.openai.com/v1 (any OpenAI-compatible server)
    LLM_MODEL      default gpt-6-luna
    LLM_MAX_TOKENS default 32000 (answer limit, reasoning included)
    LLM_EFFORT     reasoning effort for every step (low | medium | high); default high
    LLM_API_KEY    the key for LLM_BASE_URL (OPENAI_API_KEY, then GROQ_API_KEY, if it is not set)

    Back to Groq / gpt-oss: LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_MODEL=openai/gpt-oss-120b

ask() sends one request with the JSON schema enforced by the server (strict).
cached_ask() wraps it for the extraction steps:
    cache     the answer is saved in papers/<id>/llm_cache/<hash>.json, keyed on everything that goes
              into the request (model, effort, prompt, message, schema). The same call again costs
              nothing; change the prompt or the paper text and only the calls that changed are paid.
    dry run   DRY_RUN = True (set by --dry-run): print the size of each call that would be made,
              send nothing, return None.
    usage     tokens per call (from the server), added to USAGE for the run summary.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1') + '/chat/completions'
MODEL = os.environ.get('LLM_MODEL', 'gpt-6-luna')
MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', 32000))   # free Groq tier: 5000 (prompt + this must fit 8000 a minute)
EFFORT = os.environ.get('LLM_EFFORT', 'high')                # every step, unless a call keeps its own (keep_effort)
DRY_RUN = False
USAGE = {'calls': 0, 'cached': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'dry_run_tokens_estimate': 0}
LAST_USAGE: dict = {}


class JSONFailed(Exception):
    """The server could not produce JSON matching the schema, usually because the answer (with its
    reasoning) ran out of tokens. The caller splits the batch and asks again."""


def cached_ask(paper_dir: str, system: str, user: str, schema: dict, name: str, effort: str = 'low',
               label: str = '', refresh: bool = False, max_tokens: int | None = None,
               keep_effort: bool = False) -> dict | None:
    """ask() with the on-disk cache and the dry run. label: what the call is, for the printout."""
    if not keep_effort:                                    # e.g. LLM_EFFORT=high: more thinking on every step
        effort = EFFORT or effort
    key = hashlib.sha256(json.dumps([MODEL, effort, system, user, schema], ensure_ascii=False).encode()).hexdigest()[:16]
    path = f'{paper_dir}/llm_cache/{name}_{key}.json'
    if os.path.exists(path) and not refresh:
        USAGE['cached'] += 1
        print(f'  {label}: cached')
        return json.load(open(path))['answer']
    est = (len(system) + len(user)) // 4                   # ~4 characters per token
    if DRY_RUN:
        USAGE['dry_run_tokens_estimate'] += est
        print(f'  {label}: would send ~{est} input tokens')
        return None
    answer = ask(system, user, schema, name, effort, max_tokens)
    USAGE['calls'] += 1
    USAGE['prompt_tokens'] += LAST_USAGE.get('prompt_tokens', 0)
    USAGE['completion_tokens'] += LAST_USAGE.get('completion_tokens', 0)
    print(f"  {label}: {LAST_USAGE.get('prompt_tokens', '?')} in, {LAST_USAGE.get('completion_tokens', '?')} out "
          f"(effort {effort}, answer limit {max_tokens or MAX_TOKENS})")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({'model': MODEL, 'effort': effort, 'usage': LAST_USAGE, 'user': user, 'answer': answer},
              open(path, 'w'), indent=1, ensure_ascii=False)
    return answer


def usage_line() -> str:
    u = USAGE
    if DRY_RUN:
        return f"dry run: {u['cached']} calls cached, about {u['dry_run_tokens_estimate']} input tokens to send (plus the output)"
    return (f"LLM: {u['calls']} calls ({u['prompt_tokens']} in, {u['completion_tokens']} out tokens), "
            f"{u['cached']} answers from the cache")


def api_key() -> str | None:
    return os.environ.get('LLM_API_KEY') or os.environ.get('OPENAI_API_KEY') or os.environ.get('GROQ_API_KEY')


def ask(system: str, user: str, schema: dict, name: str, effort: str = 'low', max_tokens: int | None = None) -> dict:
    """One call with the schema enforced by the server (strict). No silent fallback to looser
    decoding: if strict mode fails, the run stops. Rate limits: wait as long as the server asks."""
    global LAST_USAGE
    key = api_key()
    if not key:
        raise SystemExit('set LLM_API_KEY first (the key for ' + URL + ')')
    budget = max_tokens or MAX_TOKENS
    style = {'limit_key': 'max_tokens', 'temperature': True}   # OpenAI's reasoning models want max_completion_tokens

    def make(b):
        body = {'model': MODEL, style['limit_key']: b, 'reasoning_effort': effort,
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                'response_format': {'type': 'json_schema', 'json_schema': {'name': name, 'schema': schema, 'strict': True}}}
        if style['temperature']:
            body['temperature'] = 0
        return json.dumps(body)
    body = make(budget)
    last_requested = None
    for _ in range(8):
        req = urllib.request.Request(URL, data=body.encode(), method='POST', headers={
            'Content-Type': 'application/json', 'Authorization': f'Bearer {key}',
            'User-Agent': 'Mozilla/5.0'})          # Cloudflare blocks the default urllib agent
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                reply = json.loads(r.read())
                LAST_USAGE = reply.get('usage', {})
                choice = reply['choices'][0]
                try:
                    return json.loads(choice['message'].get('content') or '')
                except json.JSONDecodeError:                    # cut off (finish_reason 'length') or refused
                    raise JSONFailed(f"no complete JSON (finish_reason {choice.get('finish_reason')})")
        except urllib.error.HTTPError as e:
            msg = e.read().decode('utf-8', 'replace')
            if e.code == 400 and 'json_validate_failed' in msg:
                raise JSONFailed(msg[:300])
            if e.code == 400 and "'max_tokens'" in msg and style['limit_key'] == 'max_tokens':
                style['limit_key'] = 'max_completion_tokens'      # provider wants the newer parameter name
                body = make(budget)
                continue
            if e.code == 400 and "'temperature'" in msg and style['temperature']:
                style['temperature'] = False                      # reasoning models only allow the default
                body = make(budget)
                continue
            if e.code == 400 and ('length' in msg or 'max_completion_tokens' in msg) and 'incomplete' in msg:
                raise JSONFailed(msg[:300])
            too_big = re.search(r'Limit (\d+), Requested (\d+)', msg) if e.code == 413 else None
            if too_big:                            # free tier: prompt + answer budget over the per-minute limit
                requested = int(too_big.group(2))
                if requested == last_requested or budget - (requested - int(too_big.group(1)) + 100) < 1500:
                    raise SystemExit(f'this request is larger than the plan allows per minute ({too_big.group(1)} '
                                     f'tokens): a paid plan, or a smaller prompt, is needed. {msg[:160]}')
                last_requested = requested
                budget -= requested - int(too_big.group(1)) + 100
                if budget < 1500:
                    raise SystemExit(f'the prompt alone is too large for this plan\'s per-minute limit: {msg[:200]}')
                print(f'  request over the per-minute limit: answer budget lowered to {budget} tokens')
                body = make(budget)
                continue
            if e.code != 429:
                raise SystemExit(f'HTTP {e.code}: {msg[:300]}')
            wait = _wait_seconds(msg)
            if wait > 120:                         # a daily (not per-minute) limit: waiting here is pointless
                raise SystemExit(f'rate limit, try again in about {wait / 60:.0f} min: {msg[:200]}')
            time.sleep(wait + 1)
        except (urllib.error.URLError, TimeoutError) as e:   # network hiccup or a slow reply: retry
            print(f'  network error ({e}), retrying')
            time.sleep(10)
    raise SystemExit('still failing after 8 attempts')


def _wait_seconds(msg: str) -> float:
    """'try again in 1h2m3.5s' / '7.1s' / '850ms' -> seconds."""
    m = re.search(r'try again in ((?:[\d.]+(?:h|ms|m|s))+)', msg)
    if not m:
        return 20.0
    unit = {'h': 3600, 'm': 60, 's': 1, 'ms': 0.001}
    return sum(float(v) * unit[u] for v, u in re.findall(r'([\d.]+)(h|ms|m|s)', m.group(1)))
