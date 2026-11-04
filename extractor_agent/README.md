# extractor_agent

Paper -> one Sup subgraph in Turtle, following `schema/kg_schema.ttl` (Sup, Law, Variable, Parameter, Metadata).
MSc thesis: *A multi-agent LLM system for automated mechanistic model development in process industry 6.0* (UniPD).

Current version: v0. Every law has its own nodes, and equations are stored as LaTeX: [`results/v0_distinct_graph.md`](results/v0_distinct_graph.md).
Why there is no SymPy translation: [`results/sympy_parser_eval.md`](results/sympy_parser_eval.md).

## Run

    # 1. on Colab: colab/paddleocr_vl_extract.ipynb  ->  papers/<id>/paddle.json + paddle_blocks.json
    # 2. here, once per new paper, before any LLM step: equation numbers from the OCR layout
    #    ((2a), (A1), [3], numbers below wide formulas, 'd = 0,1,...' conditions); prints what to check
    python -m extractor.numbering papers/<id>
    # 3. then:
    export LLM_API_KEY=...      # default model gpt-6-luna on OpenAI, effort high (extractor/llm.py)
    python -m extractor.sup papers/kontoravdi_2010         # LLM: which equations are the model + Sup metadata -> paper.json (proposed)
    python -m extractor.sup papers/kontoravdi_2010 --approve   # after checking paper.json
    python -m extractor.symbols papers/kontoravdi_2010     # LLM: description, unit, value, Parameter/Variable per symbol
    python -m extractor.laws papers/kontoravdi_2010        # LLM: law description, type, assumptions, roles
    python -m extractor.supmeta papers/kontoravdi_2010     # LLM: Sup description (system, equipment, mode, phenomena)
    python -m extractor.review papers/kontoravdi_2010      # review.html: check, correct, Export corrections -> corrections.json
    python -m extractor.graph papers/kontoravdi_2010
    python -m extractor.assemble papers/kontoravdi_2010 papers/anane_2017   # one shared graph -> kg/

Or run it all from the browser: `python -m extractor.app`, then open http://127.0.0.1:8765. Every paper with its stage; a paper's page is the review page with buttons for the next step (propose boundary, approve and continue, rebuild graph, estimate tokens). A boundary is approved automatically only when the paper lists the model's equations itself and code finds nothing to check.

Add `--dry-run` to `symbols` or `laws` to see how many tokens a run would send, without sending anything. Every LLM answer is saved in `papers/<id>/llm_cache/`; running again only pays for the calls whose input changed (`--refresh` ignores the cache).

`symbols` and `laws` replace `classify` (kept for the earlier results). Every LLM field says where it came from: stated (quoted, the quote checked against the paper) or inferred (the model's judgement), and inferred or blocking items are listed under `review`.

`classify` decides, for every symbol no law computes, Parameter or Variable. Each answer is either stated (quoted from the paper, quote checked) or inferred (convention, sent to review). It writes `classification.json`. Add `--hide-tables` to test it without the tables. When the paper has tables, the table rule is the answer key.

`graph` builds the pipeline graph and, if `gold_laws.json` exists, the reference graph. It checks both against the schema and compares them. The pipeline graph uses `symbols.json` and `laws.json` if present, else `classification.json`. Without it, and always for the reference graph, it uses the table rule: Parameter = has a number in a table row.

## Files

    extractor/
      paper.py     reads the OCR output: equations (symbols per side, ODE/algebraic, side condition)
                   and the text as snippets (sentences, list lines, table rows)
      sup.py       LLM: the model's equations (boundary), description, model assumptions, complete model?
      symbols.py   LLM: per symbol description, unit, value, Parameter or Variable
      laws.py      LLM: per law description, law type, law-specific assumptions, roles (one call per section)
      supmeta.py   LLM: the Sup's description, from the extracted laws and symbols (fields for linking Sups)
      app.py       local web app: the review pages plus buttons that run the next step
      review.py    review.html per paper: Sup, laws, symbols; edits exported as corrections.json
      llm.py       the LLM call: strict JSON schema, on-disk cache, dry run, token count
      classify.py  LLM: Parameter or Variable only (earlier step, replaced by symbols.py)
      assemble.py  KG assembler, step 1: joins papers' Sup graphs into kg/shared_graph.ttl (nothing merged yet)
      graph.py     builds the Sup graph, checks it against the schema, compares pipeline vs reference
    studies/
      sympy_parser_eval.py   side study: SymPy's own parse_latex on the 28 equations
                             (needs antlr4-python3-runtime==4.11.* and lark)
    schema/kg_schema.ttl     the 5-class schema
    papers/<id>/
      paper.json             sup id, doi, title, and optionally 'equations': which printed equations are
                             the model (written by hand, one per paper)
      paper.pdf              not in the repository (copyright)
      paddle.json, paddle_blocks.json     OCR output (from Colab)
      gold_laws.json         hand-checked equations, for the reference graph
      symbols.json, laws.json   LLM outputs (symbols.py, laws.py); llm_cache/ holds every raw answer
      classification.json    LLM classification (from classify.py, earlier step)
      sup_graph_*.ttl, *_report.json, graph_comparison.json, sympy_parser_eval.json   outputs
    results/                 write-ups
    colab/                   paddleocr_vl_extract.ipynb (used), marker_extract.ipynb (earlier reader)
    _archive/                not used now, kept for reference:
      sympy_converter/       first Step-1 version (converter, elimination, shared nodes, figures)
      unused_tools/          LLM harness, unit parser, table/text-layer candidates, Marker adapters, PDF helpers
      v1_pipeline/           the Pass 1 / Pass 2 pipeline and its outputs

Python 3.9+ and rdflib; classify needs a Groq API key (no extra packages). The study also needs sympy.
