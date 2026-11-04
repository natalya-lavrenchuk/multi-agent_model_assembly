# v0 graph: LaTeX storage, distinct nodes per law

Built and checked by `extractor/graph.py` (equations and tables read by `extractor/paper.py`). Code only, no LLM, no SymPy.

## What changed from the first version

| | first version (archived, git `ade31c4`) | v0 (now) |
|---|---|---|
| equation stored as | SymPy string (`mkg:formula`), from a rule-based converter | LaTeX as read (`mkg:latex`); `mkg:formula` empty |
| Variable / Parameter nodes | one per symbol per Sup, shared by every law that uses it | one per symbol **per law**; the same symbol in two laws is two nodes |
| IRI of μ in eq 9 | `kg:var/mu` | `kg:law/eq9/var/mu` |
| computes | decided by elimination across laws (eq 1 computes V, so eq 2 computes X_v) | only when the left side has exactly one symbol; otherwise left open and flagged |
| symbols found by | SymPy, after conversion | `extractor/paper.py`, a lister on the LaTeX string (not a parser) |

The converter and everything that depended on it are in `_archive/sympy_converter/`.

## Kontoravdi result

| | pipeline (Paddle) | reference (hand-checked LaTeX) |
|---|---:|---:|
| Laws | 28 | 28 |
| Variable nodes | 96 | 93 |
| Parameter nodes | 39 | 38 |
| distinct symbols | 66 | 63 |
| computes edges | 25 | 25 |
| ODE / algebraic | 15 / 13 | 16 / 12 |

Parameter type uses paper-level evidence, but the node still belongs to one law. A symbol is a Parameter when it is never on a left side and a table row gives it a number. By symbol this gives the paper's 30 parameters, the same set as before. There are more nodes than parameters because, for example, K_A sits in 5 laws, so it has 5 nodes.

**Pipeline vs reference, read back from the two Turtle files:**

- 25 of 28 laws have identical computes, uses and hasParameter sets. ν is counted as v: Paddle reads the printed X_v as X_ν. Without that normalisation, 19 of 28 match.
- The 3 laws that differ are the 3 reading errors:
  - **eq 6**: `K_{Ilac}` read as `K I_{lac}`, one name split into K and I_lac. Flagged "split name?" from the table row `KI_{lac}`.
  - **eq 8**: the `d` of d/dt dropped, so d and t become symbols. Flagged "derivative marker missing".
  - **eq 24**: `[H_2L]` read as `[|H_2L]` (a stray bar). Not flagged on its own. The pipeline graph has `[H_2L]` used in 3 laws, but its "computing" law names `[|H_2L]`, so `[H_2L]` shows up as unclassified.
- The stored LaTeX string is never identical. Paddle spaces subscripts (`K_{g l c}`), so a string comparison of LaTeX is not a usable check. The symbol sets are.

## Schema validation

All terms are defined, and every edge connects the right classes. One rule fails, as expected: "every Law computes exactly one Variable" fails for **eqs 2, 3 and 8** (reference and pipeline alike).

Their left sides are d(V·X_v)/dt, d(V·X_t)/dt and d(V·[GLC])/dt. Deciding which symbol the law computes needs eq 1, which computes V. With distinct nodes a law cannot see eq 1, so the choice is left open rather than guessed.

This is the concrete cost of distinct nodes: the first version resolved all three by elimination.

## What v0 does not do (same as before)

- Meanings and units of symbols.
- Assumptions in prose.
- Roles.
- Law type.
- Parameter values.
- Linking equal symbols across laws.
