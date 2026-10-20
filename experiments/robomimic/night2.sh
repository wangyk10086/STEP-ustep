#!/bin/bash
# start-level curve on the harder tasks: lower t0 with 2 steps (and 4 steps)
set -x
export MUJOCO_GL=egl
export N_ENVS=28
cd /workspace/STEP
while ! grep -q '^NIGHT_ALL_DONE' /workspace/night.log; do sleep 60; done
E=/workspace/eval
for T in transport tool_hang; do
  P=/workspace/runs/${T}_pred/checkpoints/latest.ckpt; DP=/workspace/ckpt/${T}_dp.ckpt
  K=$(python -c "import json;print(json.load(open('$E/calib_$T.json'))['kappa'])")
  python experiments/fixed_start_sweep.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP \
    --grid 50:2 65:2 75:2 75:4 --kappa $K --eval_seeds 100000 100100 100200 --out_root $E/${T}_fixed_low
  echo "LOWT_DONE $T"; date
done
echo NIGHT2_DONE; date
