#!/usr/bin/env bash
# Supplemental Guide Labs scoring on the R2 checkpoint (VM1): the R2 eval
# chain ran pre-W7/W8 code, so flip_gated/calibrated modes and the
# attribution stage are owed. Needs the GPU free (run after R8 training
# or its eval chain, one job per card). New output files: append-only
# policy respected (no --overwrite). Also runs the W3 band-width sweep
# (0.02/0.05/0.10/0.20; band15 exists from the chain) with per-concept
# coverage/displacement now reported per mode.
set -euo pipefail
cd ~/odyssey
git merge-base --is-ancestor 8161e198eec100e50cefce7a7169fefb29773587 HEAD \
  || { echo "COMMIT FLOOR FAILED: checkout predates 8161e19"; exit 1; }
PY=~/odyssey/.venv/bin/python
RUN=~/runs/full_run_v10
DATA=~/data/mimiciv_3.1_v1/data
setsid nohup bash -c "
  $PY -m odyssey.inference.interventions --run-dir $RUN \
    --held-out-shard-dir $DATA/held_out \
    --output-json $RUN/interventions_guidelabs.json \
    --max-shards 4 --num-lanes 64 --chunk-size 512 --uncertain-band 0.15 \
    --modes none truth flip flip_gated truth_calibrated flip_calibrated \
    --calibrated-tau 1.0 --dump-per-subject > ~/r2_guidelabs.log 2>&1
  for BAND in 0.02 0.05 0.10 0.20; do
    TAG=\$(echo \$BAND | sed 's/0\\.//')
    $PY -m odyssey.inference.interventions --run-dir $RUN \
      --held-out-shard-dir $DATA/held_out \
      --output-json $RUN/interventions_band\$TAG.json \
      --max-shards 4 --num-lanes 64 --chunk-size 512 --uncertain-band \$BAND \
      --modes none truth flip --dump-per-subject >> ~/r2_guidelabs.log 2>&1
  done
  $PY -m odyssey.inference.concept_attribution --run-dir $RUN \
    --held-out-shard-dir $DATA/held_out \
    --output-json $RUN/attribution.json \
    --max-shards 4 --num-lanes 64 --chunk-size 512 >> ~/r2_guidelabs.log 2>&1
  $PY ~/odyssey/scripts/intervention_cis.py \
    --per-subject $RUN/interventions_guidelabs_per_subject.json \
    --output-json $RUN/intervention_cis.json >> ~/r2_guidelabs.log 2>&1
  echo GUIDELABS_DONE >> ~/r2_guidelabs.log
" > /dev/null 2>&1 < /dev/null &
disown
echo "R2 Guide Labs scoring launched (log: ~/r2_guidelabs.log)"
