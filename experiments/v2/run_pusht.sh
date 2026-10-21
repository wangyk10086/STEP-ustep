#!/bin/bash
# Push-T (state) follow-up runs with the official DP checkpoint (Internet Archive copy).
set -x
source /workspace/venv/bin/activate
export MUJOCO_GL=egl
W=/workspace; cd $W/STEP
mkdir -p $W/eval data
until grep -q SETUP_DONE $W/logs/setup.log 2>/dev/null; do sleep 15; done
until grep -q "^DONE" $W/logs/pdl_pusht.log 2>/dev/null; do sleep 15; done
until grep -q "^DONE" $W/logs/pdl_pushtzip.log 2>/dev/null; do sleep 15; done
[ -d data/pusht ] || (cd data && unzip -q -o $W/data_dl/pusht.zip)
ls data/pusht
P=$W/STEP/checkpoints/pusht_lowdim/step_varhead.ckpt
DP=$W/ckpt/pusht_dp.ckpt
DEV=$(python -c "import torch;print('cuda:0' if torch.cuda.is_available() else 'cpu')")
echo DEVICE $DEV
# 1) offline solver error / predicted t0*
python experiments/v2/solver_error.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --device $DEV \
  --n_contexts 512 --out $W/eval/solver_pusht.json > $W/logs/solver_pusht.log 2>&1
echo STAGE solver_pusht; date
# 2) qualitative rollouts with video (same initial states for DDIM-1 / STEP-1 / ours-1)
python experiments/v2/sim_rollouts.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP \
  --modality lowdim --t0 90 --n 1 --n_test 16 --out $W/eval/sim_pusht > $W/logs/sim_pusht.log 2>&1
echo STAGE sim_pusht; date
# 3) latency (meaningful on GPU only)
if [ "$DEV" != cpu ]; then
  python bench_latency.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --steps 1 2 4 --deltas 1.0 \
    --n_calls 300 --out $W/eval/latency_pusht.json > $W/logs/latency.log 2>&1
  echo STAGE latency; date
fi
echo PUSHT_ALL_DONE
