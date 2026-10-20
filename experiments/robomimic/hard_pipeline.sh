#!/bin/bash
# Square / Transport / ToolHang (PH, lowdim, abs action): train predictors (branch A) while evaluating (branch B)
set -x
export MUJOCO_GL=egl
export N_ENVS=28
cd /workspace/STEP
BASE=https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim
# ---- data + checkpoints ----
python - <<'EOF'
import zipfile
z=zipfile.ZipFile('/workspace/rm_lowdim.zip')
for t in ['square','transport','tool_hang']:
    names=[n for n in z.namelist() if f'/{t}/ph/' in n]
    print(names)
    for n in names: z.extract(n,'/workspace/rm_extract')
EOF
for T in square transport tool_hang; do
  mkdir -p data/robomimic/datasets/$T/ph
  D=$(find /workspace/rm_extract -path "*/$T/ph" -type d | head -1)
  ln -sf $D/low_dim.hdf5 data/robomimic/datasets/$T/ph/low_dim.hdf5
  ln -sf $D/low_dim_abs.hdf5 data/robomimic/datasets/$T/ph/low_dim_abs.hdf5
  F=$(curl -s $BASE/${T}_ph/diffusion_policy_cnn/train_0/checkpoints/ | grep -o 'href="epoch[^"]*ckpt' | head -1 | sed 's/href="//')
  echo "$T ckpt: $F"
  wget -q -O /workspace/ckpt/${T}_dp.ckpt "$BASE/${T}_ph/diffusion_policy_cnn/train_0/checkpoints/$F" && echo "${T}_CKPT_OK"
done
ls -la /workspace/ckpt data/robomimic/datasets/*/ph/

# ---- branch A: train predictors sequentially (no rollouts during training) ----
train() {  # task epochs
  [ -f /workspace/runs/$1_pred/checkpoints/latest.ckpt ] && return
  python train.py --config-name=train_action_predictor_lowdim_workspace task=$1_lowdim_abs logging.mode=disabled \
    '~task.env_runner' policy.model.predict_var=True training.num_epochs=$2 training.rollout_every=100000 \
    training.checkpoint_every=25 dataloader.num_workers=8 val_dataloader.num_workers=4 \
    exp_name=$1_pred hydra.run.dir=/workspace/runs/$1_pred > /workspace/train_$1.log 2>&1
  echo "TRAINED $1"; date
}
( train square 300; train transport 100; train tool_hang 70 ) &

# ---- branch B: evaluate each task once its predictor exists ----
evaltask() {  # task steps_list
  T=$1; PRED=/workspace/runs/${T}_pred/checkpoints/latest.ckpt; DP=/workspace/ckpt/${T}_dp.ckpt
  while ! grep -q "^TRAINED $T" /workspace/hard_pipeline.log; do sleep 30; done
  SEEDS="100000 100100 100200"
  python calibrate_uwarm.py --action_predictor_ckpt $PRED --modality lowdim --max_batches 50 \
    --deltas 0.01 0.03 0.1 0.3 1 --out /workspace/eval/calib_$T.json
  K=$(python -c "import json;print(json.load(open('/workspace/eval/calib_$T.json'))['kappa'])")
  echo "KAPPA $T $K"
  python eval_combined_inference.py --action_predictor_ckpt $PRED --diffusion_policy_ckpt $DP --modality lowdim \
    --n_test 50 --n_test_vis 0 --steps_list $2 --use_ddim --eval_seeds $SEEDS \
    --ustep_deltas 0.03 0.3 1 --ustep_kappa $K --ustep_n_max 4 --output_dir /workspace/eval/${T}_main
  echo "MAIN_DONE $T"; date
  python experiments/fixed_start_sweep.py --action_predictor_ckpt $PRED --diffusion_policy_ckpt $DP \
    --grid 95:1 90:1 95:2 85:2 95:4 --kappa $K --eval_seeds $SEEDS --out_root /workspace/eval/${T}_fixed
  echo "FIXED_DONE $T"; date
}
evaltask square "1 2 4"
evaltask transport "2 4"
evaltask tool_hang "2 4"
wait
echo HARD_PIPELINE_DONE; date
