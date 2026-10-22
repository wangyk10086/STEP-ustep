#!/bin/bash
set -x
W=/workspace; cd $W/STEP; source $W/venv/bin/activate
export MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 N_ENVS=17
SEEDS="100000 100100"
sweep() { python experiments/fixed_start_sweep.py --action_predictor_ckpt $W/runs/$1_pred/checkpoints/latest.ckpt --diffusion_policy_ckpt $W/ckpt/$1_dp.ckpt --grid $3 --n_test 50 --eval_seeds $SEEDS --out_root $W/eval/$1_$2 > $W/logs/sweep_$1_$2.log 2>&1; }
sweep transport fixed2 "85:1 90:1 85:2 95:2"; echo STAGE sweep_transport2; date
T=tool_hang
for H in none data; do
  [ -s $W/eval/solver_${T}_$H.json ] || python experiments/v2/solver_error.py --action_predictor_ckpt $W/runs/${T}_pred/checkpoints/latest.ckpt --diffusion_policy_ckpt $W/ckpt/${T}_dp.ckpt --history $H --n_contexts 512 --out $W/eval/solver_${T}_$H.json > $W/logs/solver_${T}_$H.log 2>&1
done
echo STAGE solver_tool_hang; date
sweep tool_hang fixed "60:1 70:1 80:1 90:1 85:2 95:2"; echo STAGE sweep_tool_hang; date
mkdir -p $W/bundle/predictors
for T in square can transport tool_hang; do cp $W/runs/${T}_pred/checkpoints/latest.ckpt $W/bundle/predictors/${T}_lowdim_abs_pred.ckpt; done
cd $W && tar czf rm_results.tar.gz eval logs bundle *.sh && ls -la rm_results.tar.gz
echo ALL_DONE; date
for i in $(seq 1 360); do [ -f $W/FETCHED ] && break; sleep 30; done
runpodctl stop pod $RUNPOD_POD_ID
