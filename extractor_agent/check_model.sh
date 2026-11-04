#!/bin/bash
# Which model and server the pipeline will call, from this shell
python3 -c "from extractor import llm; print('model:', llm.MODEL, '| server:', llm.URL, '| effort:', llm.EFFORT, '| key set:', bool(llm.api_key()))"
