#!/bin/bash
# Runs the agents' test suite (no LLM, no network). Pass -m "not slow" to skip the fit test.
export PATH=/opt/ipopt/bin:$PATH IPOPT=${IPOPT:-/opt/ipopt/bin/ipopt}
cd "$(dirname "$0")/.."
python -m pytest tests -v -p no:cacheprovider "$@"
