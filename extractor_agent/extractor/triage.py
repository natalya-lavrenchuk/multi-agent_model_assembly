"""triage.py -- review by exception: which flags a person must look at, which are only for information.

The pipeline marks anything it is not sure of. Most of that is "the paper did not say, so it was
inferred" (a description written from the equations, a unit worked out from them): worth knowing, not
worth a person's time one by one. Those are FYI and count as accepted. A person checks only what can
be wrong in a way that matters:
    symbols  the class contradicts its own reason; several candidate values and none chosen; no answer
    laws     a possible misprint; a blocking assumption (it limits where the law may be reused)
Nothing is hidden: FYI items stay on the review page, collapsed.
"""
from __future__ import annotations
import json
import os
import re

CHECK = [
    r'^class \w+ but the reason says',          # symbols: the LLM's reason contradicts its own class
    r'^value: not clear',                       # symbols: several candidate values
    r'^no answer from the LLM',
    r'^possible misprint',                      # laws
    r'^possible OCR misread',
    r'^blocking assumption',
]


def tier(reason: str) -> str:
    return 'check' if any(re.search(p, reason) for p in CHECK) else 'fyi'


def must_check(item: dict) -> list[str]:
    return [r for r in item.get('review') or [] if tier(r) == 'check']


def status(item: dict) -> str:
    """'accepted' unless something needs a person (FYI reasons alone do not)."""
    if item.get('status') == 'accepted':
        return 'accepted'
    return 'pending' if must_check(item) else 'accepted'


def summary(paper_dir: str) -> dict:
    """Counts after the reviewer's corrections: an item marked resolved no longer counts as 'check'."""
    from extractor.graph import corrected                 # here, not at the top: graph.py imports triage
    syms, laws = corrected(paper_dir)
    syms = {k: {**v, 'status': 'accepted'} if v.get('status') == 'accepted' else v for k, v in syms.items()}
    unresolved = lambda v: bool(must_check(v)) and v.get('status') != 'accepted'
    items = list(syms.values()) + list(laws.values())
    check = sum(unresolved(v) for v in items)
    resolved = sum(bool(must_check(v)) and v.get('status') == 'accepted' for v in items)
    fyi = sum(bool(v.get('review')) and not must_check(v) for v in items)
    return {'check': check, 'resolved': resolved, 'fyi': fyi}
