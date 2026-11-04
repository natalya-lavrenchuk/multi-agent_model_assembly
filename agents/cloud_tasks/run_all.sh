#!/bin/bash
# Runs the long jobs one after another (nproc=4 -> at most one job at a time, per BRIEF.md).
export PATH=/opt/ipopt/bin:$PATH IPOPT=/opt/ipopt/bin/ipopt
cd "$(dirname "$0")/.."
date -u +"stats 101-130 start %FT%TZ"
python stats_study.py run --seeds 101-130 > runs/stats/run_101_130.log 2>&1
date -u +"design 101-112 start %FT%TZ"
python cloud_tasks/design_study.py run --seeds 101-112 > runs/design_study/run_101_112.log 2>&1
date -u +"stats 131-200 start %FT%TZ"
python stats_study.py run --seeds 131-200 > runs/stats/run_131_200.log 2>&1
date -u +"design 113-125 start %FT%TZ"
python cloud_tasks/design_study.py run --seeds 113-125 > runs/design_study/run_113_125.log 2>&1
date -u +"design 101-112 --hr start %FT%TZ"
python cloud_tasks/design_study.py run --seeds 101-112 --hr > runs/design_study/run_101_112_hr.log 2>&1
python stats_study.py report > runs/stats/report.txt 2>runs/stats/report.err
python cloud_tasks/design_study.py report > runs/design_study/report.txt 2>runs/design_study/report.err
date -u +"all done %FT%TZ"
