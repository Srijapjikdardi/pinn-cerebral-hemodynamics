#!/usr/bin/env bash
# Reproduce the reliability study from scratch. Needs data/cow_v1.h5 (python data/generate_cow.py --n 1500 --out data/cow_v1.h5 --seed 1).
# CPU only; each 5-member ensemble takes 25-45 min on one core. Peak memory about 4 GB per job: run at most two at once on 8 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
D=data/cow_v1.h5
run() { python training/reliability_eval.py --data $D --members 5 --epochs 100 "$@"; }

for sp in variant complexity; do
  run --split $sp --seed-base 100 --out outputs/rel_${sp}.npz                         # data-only MLP
  run --split $sp --seed-base 200 --out outputs/rel_${sp}_s200.npz                    # independent second ensemble
  run --split $sp --seed-base 400 --phys-weight 0.03 --out outputs/relphys_${sp}.npz  # soft physics loss
  run --split $sp --seed-base 500 --project --out outputs/relproj_${sp}.npz           # hard junction projection
  run --split $sp --seed-base 600 --arch graph --width 128 --out outputs/relgraph_${sp}.npz   # vessel-graph surrogate
done

for sp in variant complexity; do
  python reliability/metrics.py outputs/rel_${sp}.npz
  python reliability/robustness.py outputs/rel_${sp}.npz outputs/rel_${sp}_s200.npz
  python reliability/cluster_bootstrap.py outputs/rel_${sp}.npz outputs/rel_${sp}_s200.npz
  python reliability/fallback.py outputs/rel_${sp}.npz $D
  python reliability/goodhart.py outputs/rel_${sp}.npz outputs/relproj_${sp}.npz outputs/relgraph_${sp}.npz
done
