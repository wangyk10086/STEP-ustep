#!/bin/bash
# RoboMimic (PH, low-dim, absolute actions) follow-up with the OFFICIAL Diffusion Policy checkpoints:
#   per task: train STEP predictor -> offline solver error / predicted t0* (no rollouts)
#             -> closed-loop fixed-start sweep around the prediction; Square also gets comparison videos.
# Ends by packing /workspace/rm_results.tar.gz, then stops the pod (after the bundle is fetched or 3 h).
#   nohup bash experiments/v2/run_robomimic.sh > /workspace/logs/rm.log 2>&1 &
set -x
W=/workspace; cd $W/STEP
source $W/venv/bin/activate
export MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 N_ENVS=${N_ENVS:-18}
mkdir -p $W/eval $W/runs $W/logs $W/ckpt
TASKS=${TASKS:-"square can transport tool_hang"}
SEEDS="100000 100100"
GRID="80:1 85:1 90:1 95:1 85:2 90:2 95:2"
BASE=https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim

# ---- data + official checkpoints ----
if [ ! -f $W/rm_extract/.ok ]; then
  # dl.sh (parallel aria2c download of the zip and the four checkpoints) writes ZIP_DONE when finished
  [ -f $W/dl.sh ] || bash $(dirname $0)/dl_robomimic.sh > $W/logs/dl.log 2>&1
  until grep -q ZIP_DONE $W/logs/zip.log 2>/dev/null; do sleep 20; done
  python - <<'EOF'
import zipfile
z = zipfile.ZipFile('/workspace/rm_lowdim.zip')
for t in ['square', 'can', 'transport', 'tool_hang']:
    for n in z.namelist():
        if f'/{t}/ph/' in n and n.endswith('.hdf5'):
            z.extract(n, '/workspace/rm_extract'); print(n)
EOF
  touch $W/rm_extract/.ok
fi
for T in $TASKS; do
  mkdir -p data/robomimic/datasets/$T/ph
  D=$(find $W/rm_extract -path "*/$T/ph" -type d | head -1)
  ln -sf $D/low_dim.hdf5 data/robomimic/datasets/$T/ph/low_dim.hdf5
  ln -sf $D/low_dim_abs.hdf5 data/robomimic/datasets/$T/ph/low_dim_abs.hdf5
  if [ ! -s $W/ckpt/${T}_dp.ckpt ]; then
    F=$(curl -s $BASE/${T}_ph/diffusion_policy_cnn/train_0/checkpoints/ | grep -o 'href="epoch[^"]*ckpt' | head -1 | sed 's/href="//')
    echo "$T ckpt: $F" | tee -a $W/eval/ckpt_sources.txt
    wget -q -O $W/ckpt/${T}_dp.ckpt "$BASE/${T}_ph/diffusion_policy_cnn/train_0/checkpoints/$F" && echo "${T}_CKPT_OK"
  fi
done
ls -la $W/ckpt data/robomimic/datasets/*/ph/
echo STAGE data; date

epochs() { case $1 in square|can) echo 300;; transport) echo 100;; tool_hang) echo 70;; esac; }

# ---- branch A: predictors, sequentially ----
train() {
  [ -f $W/runs/$1_pred/checkpoints/latest.ckpt ] && { echo "TRAINED $1"; return; }
  python train.py --config-name=train_action_predictor_lowdim_workspace task=$1_lowdim_abs logging.mode=disabled \
    '~task.env_runner' policy.model.predict_var=True training.num_epochs=$(epochs $1) training.rollout_every=100000 \
    training.checkpoint_every=25 training.val_every=5 dataloader.num_workers=8 val_dataloader.num_workers=4 \
    exp_name=$1_pred hydra.run.dir=$W/runs/$1_pred > $W/logs/train_$1.log 2>&1
  echo "TRAINED $1"; date
}
( for T in $TASKS; do train $T; done ) > $W/logs/train_all.log 2>&1 &

# ---- branch B: evaluate each task once its predictor exists ----
for T in $TASKS; do
  until grep -q "^TRAINED $T" $W/logs/train_all.log 2>/dev/null; do sleep 30; done
  P=$W/runs/${T}_pred/checkpoints/latest.ckpt; DP=$W/ckpt/${T}_dp.ckpt
  [ -f $P ] || { echo "NO_PRED $T"; continue; }
  for H in none data; do
    python experiments/v2/solver_error.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP \
      --history $H --n_contexts 512 --out $W/eval/solver_${T}_$H.json > $W/logs/solver_${T}_$H.log 2>&1
  done
  echo STAGE solver_$T; date
  python experiments/fixed_start_sweep.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP \
    --grid $GRID --n_test 50 --eval_seeds $SEEDS --out_root $W/eval/${T}_fixed > $W/logs/sweep_$T.log 2>&1
  echo STAGE sweep_$T; date
  if [ $T = square ]; then
    python experiments/v2/sim_rollouts.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP \
      --modality lowdim --t0 90 --n 1 --n_test 12 --out $W/eval/sim_square > $W/logs/sim_square.log 2>&1
    echo STAGE sim_square; date
  fi
done
wait

# ---- bundle: results, logs, videos, predictors (small) ----
mkdir -p $W/bundle/predictors
for T in $TASKS; do cp $W/runs/${T}_pred/checkpoints/latest.ckpt $W/bundle/predictors/${T}_lowdim_abs_pred.ckpt 2>/dev/null; done
cd $W && tar czf rm_results.tar.gz eval logs bundle --exclude='*.zarr*'
ls -la $W/rm_results.tar.gz
echo ALL_DONE; date

# ---- stop the pod once the bundle has been fetched (flag file), or after 3 h ----
for i in $(seq 1 360); do [ -f $W/FETCHED ] && break; sleep 30; done
runpodctl stop pod $RUNPOD_POD_ID
