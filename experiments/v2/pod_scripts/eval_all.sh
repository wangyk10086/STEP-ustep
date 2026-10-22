#!/bin/bash
# evaluation branch (restarted after installing av); predictor training keeps running separately
set -x
W=/workspace; cd $W/STEP; source $W/venv/bin/activate
export MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 N_ENVS=18
python -c "import av, gym; print('DEPS_OK', av.__version__)" || pip install -q --only-binary :all: "av==12.3.0"
python -c "import av, gym; print('DEPS_OK', av.__version__)" || exit 1
TASKS="square can transport tool_hang"; SEEDS="100000 100100"
GRID="70:1 80:1 85:1 90:1 95:1 80:2 85:2 90:2 95:2"
for T in $TASKS; do
  until grep -q "^TRAINED $T" $W/logs/train_all.log 2>/dev/null; do sleep 30; done
  P=$W/runs/${T}_pred/checkpoints/latest.ckpt; DP=$W/ckpt/${T}_dp.ckpt
  for H in none data; do
    [ -s $W/eval/solver_${T}_$H.json ] || python experiments/v2/solver_error.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --history $H --n_contexts 512 --out $W/eval/solver_${T}_$H.json > $W/logs/solver_${T}_$H.log 2>&1
  done
  echo STAGE solver_$T; date
  if [ $T = square ]; then
    rm -rf $W/eval/sim_square
    python experiments/v2/sim_rollouts.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --modality lowdim --t0 90 --n 1 --n_test 12 --out $W/eval/sim_square > $W/logs/sim_square.log 2>&1
    echo STAGE sim_square; date
  fi
  rm -rf $W/eval/${T}_fixed
  python experiments/fixed_start_sweep.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --grid $GRID --n_test 50 --eval_seeds $SEEDS --out_root $W/eval/${T}_fixed > $W/logs/sweep_$T.log 2>&1
  echo STAGE sweep_$T; date
done
mkdir -p $W/bundle/predictors
for T in $TASKS; do cp $W/runs/${T}_pred/checkpoints/latest.ckpt $W/bundle/predictors/${T}_lowdim_abs_pred.ckpt 2>/dev/null; done
cd $W && tar czf rm_results.tar.gz eval logs bundle
ls -la $W/rm_results.tar.gz
echo ALL_DONE; date
for i in $(seq 1 360); do [ -f $W/FETCHED ] && break; sleep 30; done
runpodctl stop pod $RUNPOD_POD_ID
