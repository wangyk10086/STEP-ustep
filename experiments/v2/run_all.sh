#!/bin/bash
# Follow-up experiments on a fresh pod. Run after setup_v2.sh.
#   A) Push-T: qualitative rollouts (DDIM-1 / STEP-1 / ours-1), latency, offline solver error
#   B) RoboMimic lowdim (abs actions): retrain predictors, offline solver error -> predicted t0*
#   C) Square rollouts with video
# Every result lands in /workspace/eval (persistent volume); STAGE lines mark progress.
set -x
export MUJOCO_GL=egl
export N_ENVS=${N_ENVS:-24}
W=/workspace; cd $W/STEP
mkdir -p $W/eval $W/runs
until grep -q SETUP_DONE $W/logs/setup.log 2>/dev/null; do sleep 20; done
until grep -q CKPT_ALL_DONE $W/logs/ckpt.log 2>/dev/null; do sleep 20; done
P_PUSHT=$W/STEP/checkpoints/pusht_lowdim/step_varhead.ckpt

# ---------------- A) Push-T ----------------
( wget -q -O $W/data_dl/pusht.zip https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip && \
  mkdir -p data && cd data && unzip -q -o $W/data_dl/pusht.zip && echo PUSHT_DATA_OK ) > $W/logs/pusht_data.log 2>&1 &
python experiments/v2/sim_rollouts.py --action_predictor_ckpt $P_PUSHT --diffusion_policy_ckpt $W/ckpt/pusht_dp.ckpt \
  --modality lowdim --t0 90 --n 1 --n_test 16 --out $W/eval/sim_pusht > $W/logs/sim_pusht.log 2>&1
echo STAGE sim_pusht; date
python bench_latency.py --action_predictor_ckpt $P_PUSHT --diffusion_policy_ckpt $W/ckpt/pusht_dp.ckpt \
  --steps 1 2 4 --deltas 1.0 --n_calls 300 --out $W/eval/latency_pusht.json > $W/logs/latency.log 2>&1
echo STAGE latency; date

# ---------------- B) RoboMimic data + predictors ----------------
until grep -q DATA_OK $W/logs/data.log 2>/dev/null; do sleep 20; done
python - <<'EOF'
import zipfile
z = zipfile.ZipFile('/workspace/data_dl/rm_lowdim.zip')
for t in ['can', 'square', 'transport', 'tool_hang']:
    for n in z.namelist():
        if f'/{t}/ph/' in n and n.endswith('.hdf5'):
            z.extract(n, '/workspace/rm_extract'); print(n)
EOF
for T in can square transport tool_hang; do
  mkdir -p data/robomimic/datasets/$T/ph
  D=$(find $W/rm_extract -path "*/$T/ph" -type d | head -1)
  ln -sf $D/low_dim.hdf5 data/robomimic/datasets/$T/ph/low_dim.hdf5
  ln -sf $D/low_dim_abs.hdf5 data/robomimic/datasets/$T/ph/low_dim_abs.hdf5
done
ls -la data/robomimic/datasets/*/ph/
echo STAGE rm_data; date

until grep -q PUSHT_DATA_OK $W/logs/pusht_data.log 2>/dev/null; do sleep 10; done
python experiments/v2/solver_error.py --action_predictor_ckpt $P_PUSHT --diffusion_policy_ckpt $W/ckpt/pusht_dp.ckpt \
  --out $W/eval/solver_pusht.json > $W/logs/solver_pusht.log 2>&1
echo STAGE solver_pusht; date

train() {  # task epochs
  [ -f $W/runs/$1_pred/checkpoints/latest.ckpt ] && return
  python train.py --config-name=train_action_predictor_lowdim_workspace task=$1_lowdim_abs logging.mode=disabled \
    '~task.env_runner' policy.model.predict_var=True training.num_epochs=$2 training.rollout_every=100000 \
    training.checkpoint_every=50 training.val_every=10 dataloader.num_workers=8 val_dataloader.num_workers=4 \
    hydra.run.dir=$W/runs/$1_pred > $W/logs/train_$1.log 2>&1
  echo "TRAINED $1"; date
}
for spec in square:300 can:300 transport:100 tool_hang:70; do
  T=${spec%:*}; E=${spec#*:}
  train $T $E
  python experiments/v2/solver_error.py --action_predictor_ckpt $W/runs/${T}_pred/checkpoints/latest.ckpt \
    --diffusion_policy_ckpt $W/ckpt/${T}_dp.ckpt --out $W/eval/solver_$T.json > $W/logs/solver_$T.log 2>&1
  echo STAGE solver_$T; date
  if [ $T = square ]; then
    python experiments/v2/sim_rollouts.py --action_predictor_ckpt $W/runs/square_pred/checkpoints/latest.ckpt \
      --diffusion_policy_ckpt $W/ckpt/square_dp.ckpt --modality lowdim --t0 90 --n 1 --n_test 12 \
      --out $W/eval/sim_square > $W/logs/sim_square.log 2>&1
    echo STAGE sim_square; date
  fi
done
echo ALL_DONE; date
