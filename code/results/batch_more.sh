#!/bin/bash
# Rollout go/no-go test on more instances and settings (30 days each).
cd "$(dirname "$0")/.."
run() { out="results/rollout_$1.txt"; shift; python run_rollout.py "$@" --seeds 30 --workers 5 > "$out" 2>&1; }
(
  run R75_R3_ugv1   R75_R3  --ugv 1
  run R75_R3_ugv2   R75_R3  --ugv 2
  run C75_R5_dod25  C75_R5  --ugv 1 --dod 0.25
  run C75_R5_dod75  C75_R5  --ugv 1 --dod 0.75
  run R100_R4_ugv1  R100_R4 --ugv 1
) &
(
  run RC75_R4_ugv1  RC75_R4 --ugv 1
  run RC75_R4_ugv2  RC75_R4 --ugv 2
  run C75_R5_E45    C75_R5  --ugv 1 --scale 0.1 --battery 45
  run C75_R3_ugv1   C75_R3  --ugv 1
  run RC100_R5_ugv1 RC100_R5 --ugv 1
) &
wait
echo all done
