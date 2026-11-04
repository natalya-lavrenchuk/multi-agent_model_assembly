# Issues log

## Fixed on 21 Oct (before running batch 2; code-only tests, no API calls)
- numbering.py (new): numbers by page geometry, not OCR reading order; (2a)/(A1)/[3]/full-width kept
  as printed; condition lines ('d = 0,...', 'if [AMM] <= ...') join the equation above. Reproduces the
  checked numbering of all 10 earlier papers. Applied to the 7 new papers (paddle.orig.json = backup).
  nagy_2008: second (13) set to 15 by hand (OCR misread).
- split.py (new): a printed block with several equations becomes one law per equation, ids '6.1'...,
  'part_of': 6 (xing eq 6, nagy eq 11 matrix form, vandenbussche [3], asprey 16). No approved model of
  the earlier papers is affected.
- Unnumbered equations get ids 'p5-2' instead of being dropped (vandenbussche numbers 2 of 33).
- Table equations (issues 1, 2): all reach the laws step; the note under the equation is cut off.
- In-place updates (issue 3): read as 'updates' (metadata mkg:updates), not rivals of the law that
  computes the quantity; check message when one quantity is updated several times.
- Boundary rules 4-6 in the sup prompt; free-floating laws: 'free_laws' in paper.json, written as Laws
  with no Sup (mkg:freeLaw, fit context on the law).
- Issue 7: blocking only for stated assumptions that are process conditions (new process_condition field).
- Issue 8: misprint flag needs a named term and a reason; standard forms are not misprints.
- Issue 9: saturation concentrations / solubilities / equilibrium constants are parameters.
- Values per model: symbols.json 'values_by_model' (rival models, free laws); graph picks the Sup's own.
  graph.py: exact floats (1.4e-1). A value with several candidates is no longer written (seidel beta_13).

## Batch 2 boundaries (Luna, 21 Oct): 7 new papers + muloiwa, goldrick
Fixed in sup.py (code only; reran from the cache, no cost):
- A stated equation list overrode the LLM's even when it covered only part of the model (owhondah 2R
  lists the balances 5-8 but not the kinetics; bolton's sentence lists single-mechanism laws). Now the
  stated list is used only if it does not leave quantities uncomputed that the LLM's list computes.
- Code added back equations the LLM had excluded with a reason (nagy A1 and 23: analytical method).
  Now it never overrides an explicit exclusion; a check asks "input, or should eq N be in?".
- In-place updates were counted as rivals in the per-model check (goldrick 18, 27 vs 11). Fixed.
Open, for review or decision:
- owhondah: 30 Sups (every structure x kinetics combination). The paper fits them all (Tables 3, 4: rRMSE
  only) but gives parameter values only for 6 (Table 5: GW and FW x 1R, 2R, 3R). DECIDED 21 Oct: one Sup per
  combination with reported values (6), unused candidate laws (Monod 17, Tessier 20) as free laws; general rule
  added to the sup prompt.
  Also: the kinetic laws (eqs 15-20) are generic templates (r, S, X) applied to each reaction (r1 with
  S1, X1; r2 with S2, X2); nothing in the graph binds a template to a reaction yet.
- vandenbussche: Luna kept the rate laws with the full site balance (p5-1, p5-2, eq 2) and excluded [3],
  the reduced form the paper fits (Table 1 belongs to [3]). Reviewer correction: [3.1], [3.2] plus the
  constants' definitions.
- goldrick: A-3 (temperature) and A-10 (pH) are written mu_X ~ mu_X,max f(...), so they read as rivals of
  each other, not as factors. Paper notation is ambiguous; review.
- nagy (Delta C, b), bolton (t), osberghaus (D_ax): flagged "input, or missing equation?"; all look like
  inputs/parameters.
- muloiwa: proposal is in paper_proposed_gpt-6-luna.json (the hand-written paper.json counts as approved).
Fixed after the first owhondah rerun:
- The boundary step never showed tables, so Luna could not see owhondah's Table 5 and guessed the 6 models
  (4 returned, 2 of them wrong). Now tables about fitted models are in the message (title once, short rows,
  max ~5000 characters).
- Code added the 1R balances (eqs 1, 2) to the 2R models because eq 9 names X1, S1. Now code never adds an
  equation the LLM put in another model of the same paper; a check asks instead.
KNOWN LIMITATION (for the thesis): models written in matrix form (owhondah 2R/3R: dxi/dt = K r(xi) + D(xi_in - xi),
with xi, r, K defined in one block) are read as one vector law; the pipeline cannot yet turn them into one
law per species, and the generic kinetic templates (r, S, X) are not bound to reactions (r1 with S1, X1).

## owhondah_2016: DROPPED (21 Oct), moved to _to_delete/. Matrix-form models not worth the work for one paper.

## xing_2010 extraction review (21 Oct), fixed in code:
- KI_amm / KD_lac split into K * I_amm: names of 2-3 capitals + subscript now come from tables, the
  nomenclature and \mathrm{KI} in the equations (prose_names).
- 'gln' read as \ln: inside subscripts ln/log/lg are letters (m_{g\ln} -> m_{gln}); eq 8 and eq 6.3 now share m_gln.
- Table rows with a value are always among a symbol's snippets (mu_max, k_d, yields were in Table 2 but not shown).
- Subscripts that differ only in case match (Y_{X_V/glc} vs Y_{Xv/glc}).
- value_number (plain number in the given unit) added to symbols; graph uses it; _number drops '± err'.
- Regression: symbol sets of the 7 earlier papers unchanged.
LLM miss (review item): law conditions in the text before eq 8 (mu = 0 in the death phase; m_glc = m_gln = 0 for d <= 6) not recorded.

## Boundaries approved 21 Oct (command line, no LLM): asprey, nagy (eq 10 removed), bolton, osberghaus,
vandenbussche (3.1, 3.2, p5-5, p5-6, p5-8), muloiwa (free laws; hand version kept as paper_hand.json).
Earlier versions: paper.before_review.json. goldrick ON HOLD (A-3, A-10 temperature/pH factors).
- preflight.py (new, code only) + run_batch.sh (one command for all approved papers).
- Reader: '\mathrm{e x p}' / spaced function names read as the function (was e*x*p). No regressions.
Review items (not code): vandenbussche p5-8 kappa(i) is indexed over parameter groups i; K* in [3] vs k1* in
p5-5 (paper's own inconsistent case). goldrick: mu_X/mu_x, mu_P/mu_p, C_{P_w}/C_{pw} case variants to check.

## Batch run 21 Oct 19:55 (asprey, nagy, bolton, osberghaus, vandenbussche, muloiwa, wang, anane, birol): fixed in code
- graph.py crashed sorting ids '11.1' with 12 (nagy): natural sort for every id form.
- Reader: function notation on the left 'C^{*}(T) = ...' computes C* (not T); 'lg(K_P1)' kept as K_P1.
- Reader: a label before the law 'M_I(theta,u,t); r = ...' dropped (asprey; junk symbols M_I, theta, u gone).
- Reader: 'A/A_0 = ...' computes A, A_0 is used (bolton: 24 schema errors gone).
- graph.py: a role naming a symbol the law no longer has is skipped (seidel crash).
- Values: table rows were cut at 260 characters including their long title, so the value columns never
  reached the LLM (anane: 0 of 20 values; bolton). Title now shortened, not the row.
- Values per data set/condition: symbols 'values_by_case' (osberghaus per protein/pH); graph mkg:valuesByCase.
- All approved papers build with 0 schema problems except seidel eq 23 (pre-existing: gamma*/gamma_0).
Review items (paper or LLM, not code): misprint flags in muloiwa (4, 5, 7, 16), anane (16-21, many look like
unit misreads by the LLM), birol (3, 15, 17); vandenbussche parameters are fitted as groups (Table 1), so
single constants have no value; nagy gives ln k_b and -ln k_g (transformed); asprey mode 'semi-continuous'
(is fed-batch).

## Review by exception (21 Oct, no rerun needed): triage.py
Must check: class contradicting its reason, several candidate values, no LLM answer, possible misprint,
blocking assumption, preflight (unreadable equation, symbols differing only in case). Everything else
(description/unit/class inferred) is FYI: counts as accepted, shown collapsed on the review page. graph.py
writes the triage status; app.py's column is "To check". Kontoravdi: 39 -> 5 to check.

## PLANNED (after the demo): parameter values when the paper does not print them
Order of sources, each stored with its basis:
1. reported value (done); 2. derived from the paper's own equations (seidel k_i(T) from beta_1-13;
vandenbussche groups A, B), computed by code at a stated temperature, basis 'derived'; 3. literature values
the paper cites (xing Table 3 'Values in literature', osberghaus literature table), basis 'cited';
4. range across papers in the KG after linking (same concept, several Sups), basis 'kg range';
5. LLM prior range from domain knowledge, only if nothing else: a range, unit-checked, basis 'llm prior',
never written as mkg:value, always shown in review. Also: experiment conditions in methods text
(ho, bolton: area, R_m, viscosity) are missed by the snippet search.

## 22 Oct: shared KG v1 built (19 papers, 27 Sups + 15 free laws, 252 laws, ~19k triples, 2 schema problems)
- Fixed: PDE/ODE with several derivatives on the left computes the state written first (ng eq 1 C, nopens eq 1 n).
- Fixed: 'h(x,t)_{agg}' read as h_{agg}(x,t) (nopens). nopens symbols.json predates this: rerun its symbols
  step for descriptions of h_agg, h_break.
- Left: nopens eq 7 (D[4,3] notation), seidel eq 23 (gamma*/gamma_0): computes undecided.
- Sup-level assumptions are all 'blocking' when quoted (cheema: 6); apply the process-condition test there
  too, and allow editing severity on the review page.

# Earlier log

## Batch 1: Luna on anane_2017, birol_2002, wang_2019 (extraction), muloiwa_2020, goldrick_2015 (boundary)

### Must fix (general, will recur)
1. Equations printed inside a table are dropped from the laws step. wang_2019 eqs 1-6 are `table_formula`
   (Table 1, "Full Set of ODEs"); the laws message contained only eq 6, so the LLM saw no LaTeX for eqs 1-5
   and wrote "equation absent" for each. Likely laws._position maps all six to the same table block.
2. Table equations carry stray text from neighbouring cells: wang eq 1 ends "... -m X_cd   Q_n",
   eq 4 ends "... m_chl". Probably the table-to-formula step joins other columns.
3. In-place updates written with ≈ (goldrick eqs 18, 19, 22, 24, 26-28, A-3, A-10):
   "mu_X ≈ 0.5 mu_X (1 - tanh(...))" means "multiply mu_X by this factor". OCR is right; the lister refuses
   them and the check tells the user to fix the LaTeX, which is the wrong advice. Need a reading as a
   modifier of mu_X (base rate times factors).

### Rules to decide (boundary)
4. DECIDED (21 Oct): a feed policy (anane eq 1) is part of the Sup, as the law that defines an input.
   Controllers stay out. Generic initial conditions (anane eq 6) stay out.
   Was: Operating policies and initial conditions: anane eq 1 (exponential feed F(t)) and eq 6 (x(t0) = x0) were
   added by Luna; the hand boundary excludes them. Controllers are excluded (birol eq 8, goldrick A-21).
   Feed policy: in the Sup as an input definition, or out like the controller? Generic initial conditions: out.
5. DECIDED (21 Oct, agreed with the PhD students): rate expressions compared without a model around them
   are free-floating Laws, not Sups. To build: sup.py outcome "laws only" (no Sup); graph.py writes Law
   nodes with their Metadata and Parameters but no Sup/includes; the fit context (system, data, source)
   has to sit on the Law's Metadata since there is no Sup to carry it. Owhondah is mixed (full models
   with alternative kinetics): look at it when it arrives.
   Was: Papers that compare rival rate expressions without balances (muloiwa: 15 Sups, one rate law each;
   owhondah will be similar). One Sup per rate law (current rule), or another structure?
6. DECIDED (21 Oct): follow the paper's list ([H+] is an input, pH controlled); keep the check visible in review.
   Was: birol: eq 3 uses [H+], eq 5 computes it, the paper's own list leaves eq 5 out (code check flagged it).
   Decide: [H+] as input (pH controlled) or include eq 5.

### Calibration
7. Blocking assumptions too liberal: birol has 8 "blocking", including freezing/boiling temperature of water.
8. Misprint flags need checking against the PDFs: birol eqs 10 (K_op X + C_L, may be the normal Contois
   form: false alarm?), 15, 16, 17 (F/s_f); anane 16-19; wang 16.
9. P/V: saturation concentrations (anane DOT*, birol C_L*) classed variable; by the rule (property of the
   material/system) they are parameters. wang q_n (Luna parameter) and S_NO3_0 (Luna variable) look right.

### Working as intended
- birol boundary: Luna = hand boundary exactly.
- Values: wang 9 symbols flagged "unclear" (Table 2 gives mean and per-experiment optimum): real ambiguity.
  birol K_ox, K_op flagged (table gives a no-limitation variant).
- Operating mode: anane "batch, fed-batch", birol "fed-batch", wang "not stated" (not verified).

- 2026-10-22: preflight removed (paper-specific debugging aid, not for users). Uploading OCR runs numbering only. preflight.py and preflight.json files moved to _to_delete/preflight/.

- 2026-10-22: Asprey Sup renamed Canoid -> Contois (paper typo; quotes keep the printed word). Model-level assumptions now follow the laws.py rule: sup.py asks process_condition and drops assumptions that only name the model; existing papers classified by hand (backups in _to_delete/before_sup_assumptions/).
- 2026-10-22: roles kept only on constitutive laws (laws.py drops them for balances/definitions, graph.py and review.py check the final law type). Existing laws.json filtered: 715 -> 578 roles. Prompt unchanged so cached answers stay valid.
- 2026-10-26: Sup metadata made free-form. supmeta.py reads the whole paper (was: title, abstract, 18 keyword sentences) and returns a description paragraph plus facts {topic chosen by the LLM, text, quote}; quotes checked by code. Fixed fields system/unit_operation/operating_mode/fitted_on/phenomena removed (schema: mkg:hasFact, mkg:topic). Old sup_meta.json files read as facts until rerun. Backups in _to_delete/before_flexible_supmeta/.
- 2026-10-26: every LLM step (sup, symbols, laws, supmeta) now gets the whole paper first (paper.with_paper), then its own material. Existing papers: rerun symbols, laws, supmeta (not sup: boundaries stay approved). supmeta prompt: no individual parameter values in facts, no placeholder absences.
- 2026-10-26: blocking rule for law assumptions = quote found + process condition (stated-as-assumption no longer required; it depended on wording and varied between runs). Recomputed from cache, no LLM calls.
- 2026-10-26: law type (balance/constitutive/definition) removed from laws.py, graph (mkg:lawRole), review page and schema (PhD feedback: an LLM judgement the system does not need). Roles kept only for algebraic laws that are not a printed value (code decides from the equation).
- 2026-10-26: review page: Sup card shows Variables and Parameters of its laws (was States and Inputs); law chips and symbol table use one colour per class (variable, parameter) instead of a separate colour for computed symbols.
- 2026-10-26: symbols.py adds estimation (parameters), measured and initial_value (variables), each kept only with a quote found in the paper (else not said); graph: mkg:estimation, mkg:measured, mkg:initialValue. supmeta prompt: topic examples (bioprocess-biased) replaced by field-neutral questions; the LLM names topics.
- 2026-10-26: supmeta prompt asks for broad, reused topics (Anane without examples gave 28 topics for 32 facts).
- 2026-10-26: supmeta: reference list included (earlier models named by author and year), questions on model kind and size, standard names for kinetic/transport forms.
- 2026-10-26: supmeta: broad topics but one idea per fact, keep details; 25-45 facts (round 2 merged Anane facts into 22 compound ones and lost detail).
- 2026-10-26: symbols.py adds uncertainty (parameters, numbers checked against the paper) and same_as (same quantity written differently; batches now see the full symbol list). Graph: mkg:valueUncertainty, mkg:sameQuantityAs.
- 2026-10-27: laws prompt: mechanism/kinetic-form statements are not assumptions; sharper process_condition test (how the process is built, fed or run vs modelling choices). Full run gave 61 blocking, many of them mechanism statements (Bolton, Nagy, Wang ...).
- 2026-10-27: Sup-level assumptions re-checked by hand: Cheema heat losses -> warning; Ho (2) and Seidel (2) mechanism statements removed. sup.py prompt gets the same sharpened process-condition wording as laws.py (new papers only).
- 2026-10-27: units.py: printed units -> pint unit, dimension, SI factor (165 of 171 distinct strings parse; the 6 left are not units). graph.py writes mkg:unit, mkg:dimension, mkg:siFactor, mkg:unitNote. demo/: Anane PDF + OCR files and reset_demo.sh for the Thursday demo (rehearse once, reset, replay from saved answers).
- 2026-10-27: Bolton: each Sup keeps only its closed forms (no variable computed twice); cake-standard gets eq 22.
- 2026-10-27: review pass prep: OCR errors fixed in paddle.json (Anane 18, Nopens 3 and 7, Seidel 7, 9, C-16; checked against PDF images); Seidel corrigendum applied to eqs 22 and 28; graph writes mkg:possibleMisprint on law metadata so misprint opinions survive acceptance.
- 2026-10-27: fixes.py (new step after laws): for laws flagged as possible misprints, the LLM proposes a corrected form only when the paper shows the intended one; code checks it parses, uses only model symbols, quote in paper. Stored in fixes.json; graph writes mkg:latexCorrected next to mkg:latex (status proposed/accepted); review page has a "use the corrected form" box. Added to the app pipeline and run_batch.sh.
- 2026-10-27: fixes.py run on Kontoravdi, Muloiwa, Birol, Zentou, Wang: 11 of 12 corrections right; Birol eq 15 corrected by hand ((T-T0)/(Tv-T0), the paper says the exponent 5 is large enough, so it stays finite). Reader-names bug in fixes.py fixed.

## 27 Oct — PDF crop of the equation on the review page; hand fixes survive reruns
- Review page shows each flagged equation cut from the PDF (app `/crop/<paper>/<eq>.png`, from the OCR bbox; none for table equations) above the rendered LaTeX, with a choice: OCR error / paper error / fine. OCR error opens a box for the right LaTeX, which replaces `mkg:latex` in the graph (`mkg:ocrFixNote`); fine or OCR hides the proposed correction. Code only, no LLM calls.
- `fixes.py` now keeps entries with `basis: human` when it rewrites `fixes.json` (the first all-paper run overwrote the Birol 15 hand fix). Birol 15 restored to (T-T0)/(Tv-T0); Muloiwa 12 hand-set to capital S (OCR read s).
- Checked the 19 proposed corrections against the crops: VdB p5-5 is a case-only rename (choose fine); Cheema 6 unclear (leave unticked); the rest look right.
- Misprint card redesigned (same day): the card compares the PDF crop with **the equation the model will use** (changes highlighted), says why they differ and why it was flagged, and asks only which equation to use: corrected / paper's version / edit. The cause is worked out, not asked: corrected form used → paper misprint; "the paper already shows the corrected version" or an edit the paper prints → reading corrected (mkg:latex replaced); paper's version → false alarm. New `mkg:reviewOutcome` once resolved. Picking an option marks the law resolved. Old answers (misprint_kind/use_fix/latex_fix) still read.
- OCR misreads fixed by hand earlier (in paddle.json) are now review items too: `fixes.json` has a `misreads` section (what the OCR read, what the PDF shows, the note), 15 equations in Anane, Cheema, Kontoravdi, Nopens, Seidel; `fixes.py` keeps it on reruns. Until the reviewer decides, the graph holds what the OCR read (mkg:latex and the symbol links read from it), and the law is a "possible OCR misread (found during review, not by the pipeline)" check item. Accepting the corrected reading replaces it; the OCR version is then not kept in the graph (mkg:ocrFixNote + mkg:reviewOutcome "reading corrected" record that it happened). Symbol links follow a reading, never a misprint correction (corrected forms like d(V[GLN])/dt would break "computes"). The answers given earlier for Seidel 7, 9, C-16 and Nopens 3, 7 were cleared so these are decided in the app. Backups: _to_delete/before_misread_entries_27oct/.
- All review items resolved in the 19 papers (backups: _to_delete/corrections_before_accept_all_27oct/), except Cheema 6 and Muloiwa 4, 5 (misprints with no fix we trust): 16 paper misprints, 15 OCR misreads (reading corrected), false alarms, blocking assumptions and 3 value items accepted.
- 28 Oct: a new paper (no symbols.json yet) showed no step buttons: the 'only what needs checking' filter looked for the 'no symbol needs checking' row, which only exists once symbols are extracted, and the script stopped before drawing the step panel. Fixed the check; the step panel now runs in its own script, so an error drawing the cards cannot remove the buttons again, and any such error shows as a red line at the top of the page.
- 28 Oct: review card reworded as a proposal ('Proposed change' / 'Accept the change' / 'Keep the paper's version'); the already-in-paper box says when to tick it. fixes.py check now flags only symbols the correction adds that neither the printed equation nor the model has (t, z, x, r and integration variables excluded); 'python -c "from extractor.fixes import recheck"' redoes it without LLM calls. Only Vanden Bussche p5-5 (k1* -> K1*) remains flagged.

## 28 Oct — Sup map (linking, design A)
- New `extractor/supmap.py`: one embedding per Sup (label + description + facts, read from the shared graph; LLM_EMBED_MODEL, default text-embedding-3-small; cached in kg/embeddings/ by model + text), cosine similarity, 5 nearest Sups per Sup (`mkg:near` -> blank node with mkg:sup, mkg:similarity, mkg:rank), average-linkage clusters with the count chosen by silhouette, named by one cached LLM call per cluster (`mkg:mapCluster`), 2D classical MDS picture kg/sup_map.png. numpy + matplotlib only.
- Runs inside "build shared KG" in the app (skipped with a log line if there is no key); also `python -m extractor.supmap` (`--offline`: word counts instead of embeddings, to test the code).
- Closeness of what the papers say, not interchangeability of equations. Free-floating laws (Muloiwa) have no Sup, so they are not on the map. Rival models from one paper (Bolton, Seidel, Asprey, Zentou, Osberghaus) sit together by construction.
- Quantity-concept linking (design C: law outputs mapped to shared concepts, rivals by code) is proposed for the 29 Oct meeting; not built.
- Map in the ontology: mkg:Group (derived class: members by code, name by an LLM), Sup mkg:inGroup Group, Sup mkg:closeTo its 2 nearest Sups (plain edges, drawn by GraphDB without a query); mkg:mapCluster removed. Scores stay on mkg:near.
- Links (`extractor/links.py`, runs in "Build shared KG"): pairs = each Sup's 5 nearest Sups from other papers + every same-paper pair; one LLM call per pair (effort high) on both Sups' names, descriptions and facts -> relation (fixed list), summary, shared, differences (aspect/A/B); 'not meaningfully related' stores nothing. No review: written straight into the graph with mkg:provenance (model + date). Link node .../kg/link/<a>__<b> (no new class) + plain A mkg:relatedTo B. `cites` by code from the facts (author + year). Test map: 108 pairs, ~400k input tokens; cites found only Bolton -> Ho.
- Links changed to free text (29 Oct decision, before the meeting): related yes/no + a short paragraph (how they relate, shared, differences; several relations allowed), mkg:linkDescription on the link node; fixed relation list, summary, shared, differences removed. Pairs judged not related before are not asked again (--all to recheck). Old version kept in _to_delete/links_fixed_labels_v1.* . Exact rival queries: law-level linking (design C), later.
- 29 Oct: Kontoravdi title fixed: hybridoma (HFN7.1 in this paper, 14-4-4S for the 2005 model), not CHO; the hand-written title in paper.json was wrong, the Sup facts were right.
- Hand-kept links: kg/link_overrides.json (a, b as Sup ids, related, description, provenance) replace the LLM for that pair at no cost; related false removes a pair. First entry: Nagy ~ Nopens, the earlier fixed-label answer joined into a paragraph.
- Links reworked (29 Oct): text names the models (no A/B, no sides); one link node per pair, both Sups -> mkg:hasLink; removed mkg:closeTo, mkg:near, mkg:relatedTo, mkg:sideA, mkg:sideB (neighbours stay in kg/sup_map.json). Finished pairs kept in kg/links.json with a hash of both Sups' texts: a build asks only new or changed pairs; --redo asks all. Kontoravdi Sup renamed kontoravdi_2005 -> kontoravdi_2010 (paper.json sup + title, sup_meta.json key).
