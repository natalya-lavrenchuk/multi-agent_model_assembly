#!/bin/bash
# Setup script for the claude.ai/code environment (paste into the environment's setup script box).
# Same package versions as the Mac (.venv, Python 3.9). ipopt 3.14 from conda-forge, as on the Mac.
set -e
pip install --quiet "numpy==1.26.4" "scipy==1.13.1" "sympy==1.14.0" "pyomo==6.9.5" "rdflib==7.6.0" \
    "pint==0.24.4" "pandas==2.3.3" "networkx==3.2.1" "langgraph==0.6.11" pytest
if [ ! -x /opt/ipopt/bin/ipopt ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local bin/micromamba
  /usr/local/bin/micromamba create -y -q -p /opt/ipopt -c conda-forge "ipopt=3.14"
fi
/opt/ipopt/bin/ipopt --version | head -1
