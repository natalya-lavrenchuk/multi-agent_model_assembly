> **Superseded.** This describes the first version (shared nodes, rule-based LaTeX-to-SymPy converter), git `ade31c4`. Its code is in `_archive/sympy_converter/`. Current version: `results/v0_distinct_graph.md`.

# Step 1 results: ontology and extractor, Kontoravdi model

Paper: Kontoravdi, Pistikopoulos & Mantalaris (2010), *Comput. Chem. Eng.* 34, 1192–1198, doi:10.1016/j.compchemeng.2010.03.012.
The model is Kontoravdi et al. (2005), restated in Appendix A of the 2010 paper as 28 equations.
State: symbols are taken as the equations write them. Meanings and units of symbols are not extracted yet.

---

## 1. Ontology structure

![ontology](../figures/ontology.png)

Five classes and no others (`schema/kg_schema.ttl`).

| Class | What it is |
|---|---|
| Sup | one published model (here: Kontoravdi 2005) |
| Law | one printed equation |
| Variable | a quantity a law computes or uses; one node per symbol per Sup, shared by all its laws |
| Parameter | a constant carried by a law; `fixed` marks physical constants, so Level 3 never estimates them |
| Metadata | structured fields about a Law or a Sup |

| Edge | From → To | Rule |
|---|---|---|
| includes | Sup → Law | |
| computes | Law → Variable | exactly one per law; two laws computing the same Variable are rivals |
| uses | Law → Variable | any number |
| hasParameter | Law → Parameter | |
| hasMetadata | Sup or Law → Metadata | |
| sameAs | Law → Law | proposed, for the same law printed in two papers |

Metadata fields: source (DOI, page, quote), form (ODE / algebraic; DAE on the Sup), conditions and assumptions, status. Defined but not filled yet: roles, law type, index.

## 2. How the ontology is populated

![pipeline](../figures/pipeline.png)

| Field | Filled by | LLM? |
|---|---|---|
| Law, one per equation | PaddleOCR-VL reads the page (`colab/paddleocr_vl_extract.ipynb`) | no (OCR model) |
| Law.formula | `stages/latex_to_sympy.py`, rule-based LaTeX → SymPy | no |
| computes / uses | `stages/structure_check.py`: the left-hand side; if it holds two symbols, drop the one another law already computes (eq 2: d(V·X_v)/dt → X_v, since eq 1 computes V) | no |
| Variable vs Parameter | never computed **and** has a numeric value in a table row of the paper → Parameter; never computed, no value → Variable (candidate input, flagged) | no |
| Metadata.form | SymPy: a time derivative → ODE, otherwise algebraic | no |
| Metadata.conditions | printed next to the equation (eq 7: `n > 1`), split off by the converter | no |
| Metadata.source | DOI from the PDF, page and LaTeX from the reader | no |
| Metadata.status | `flagged` if a structural check fires on the law, else `pending` | no |

Commands (from `extractor_agent/`):

```
python -m stages.build_graph papers/kontoravdi_2010 --sup kontoravdi_2005 --doi 10.1016/j.compchemeng.2010.03.012 --title "..."
python -m stages.build_graph papers/kontoravdi_2010 --gold ...      # reference graph from the hand-checked equations
python -m stages.compare_graphs papers/kontoravdi_2010
python -m stages.make_figures papers/kontoravdi_2010 figures
```

## 3. Extractor results

### 3.1 Reading the equations

| Reader | Equations read exactly as printed (of 28) |
|---|---|
| Marker 2 | 22 |
| PaddleOCR-VL | 26 |

Paddle's two misses: eq 8 (dropped the `d` of d/dt) and eq 6 (`KI_lac` written as K × I_lac). Scored as SymPy expressions against `gold_laws.json`, so spacing differences do not count as errors.

### 3.2 LaTeX → SymPy

| Converter | Result on the 28 printed equations |
|---|---|
| SymPy's built-in LaTeX parser | 19 parse, most of them silently wrong (`F_{in}` → F·i·n, `[GLC]` → G·L·C) |
| `latex_to_sympy.py` (rule-based) | 28 / 28 correct |

### 3.3 The populated graph

![example](../figures/example_eq4_5.png)

| | Pipeline (from Paddle) | Reference (hand-checked equations) |
|---|---|---|
| Laws | 28 (15 ODE, 13 algebraic) | 28 (16 ODE, 12 algebraic) |
| Variables | 34 | 33 |
| Parameters | 30 | 30 (the paper states 30) |
| computes / uses / hasParameter edges | 27 / 67 / 39 | 28 / 65 / 38 |
| Schema validation | passes | passes |
| Laws flagged | eq 6, eq 8 | none |

Pipeline vs reference: 26 of 28 formulas identical, 27 of 28 computes edges identical. Both differences are the two reading errors, and both are flagged:

- **eq 8:** without the `d` the law is no longer an ODE, nothing is computed, and a stray symbol `d` appears. Flag: no single computed variable.
- **eq 6:** `I_lac`, `I_amm` match the table's `KI_lac`, `KI_amm` only partially. Flag: the reader probably split one name in two. Consequence if unreviewed: eq 6's `K` would be shared with the mRNA decay constant `K` of eqs 17–18.

Remaining flags (also in the reference graph): `F_in`, `F_out`, `[GLC]_in`, `[GLN]_in` and `λ` are never computed and have no table value. The first four are inputs. λ is a parameter that the paper never defines or gives a value for.

![sup graph](../figures/sup_graph_pipeline.png)

### 3.4 Problems in the paper itself

Found while building the reference. Confirmed in the PDF's own text layer, so not reading errors. Not yet detected automatically: most need units, which come with the postponed variable step.

1. Eq 5 has `[GLC]` in both factors, the second with `K_gln`; almost certainly `[GLN]` was meant.
2. Eqs 10, 13, 15, 27 write d([C])/dt on the left but amount-per-time terms (V, F) on the right; eq 8 writes the correct d(V[GLC])/dt.
3. γ₁ and γ₂ have their units swapped between the Nomenclature and Table 1. Eq 27, (γ₂ − γ₁μ), shows the Nomenclature is right.
4. K_A: cell/molecule/h in the Nomenclature, cell/molecule-L in Table 1.
5. μ_max: 5.8×10⁻² in Table 1 (fed-batch), 5.4×10⁻² in Table 2.
6. λ (eq 28): no definition, no value anywhere in the paper.
7. K_lysis: fitted in Table 2, but appears in none of the 28 equations.

## 4. Not done yet

- Symbol meanings and units (the LLM-assisted variable step is built and tested on its code-only parts, then paused).
- Law metadata from the prose: assumptions, roles, law type.
- Dimensional consistency check (needs units).
- Assembler: merging Sups from several papers. Only one paper has gone through the new pipeline.

## 5. Files

| File | Content |
|---|---|
| `papers/kontoravdi_2010/sup_graph_pipeline.ttl` | the Sup graph built from the reader output |
| `papers/kontoravdi_2010/sup_graph_reference.ttl` | the same, built from the hand-checked equations |
| `papers/kontoravdi_2010/*_report.json` | counts, flags, schema validation |
| `papers/kontoravdi_2010/graph_comparison.json` | pipeline vs reference, law by law |
| `papers/kontoravdi_2010/gold_laws.json` | the 28 equations as printed, with reader errors and paper errors noted |
| `figures/` | all figures above (PNG and SVG, regenerated by `make_figures.py`) |
