#!/bin/bash
set -x
export MUJOCO_GL=egl
export N_ENVS=56
cd /workspace/STEP
until grep -q DATA_OK /workspace/data.log 2>/dev/null && grep -q SETUP_DONE /workspace/setup.log 2>/dev/null; do sleep 15; done
H5=$(find /workspace/rm_extract -name low_dim.hdf5 | head -1)
mkdir -p data/robomimic/datasets/can/ph
ln -sf $H5 data/robomimic/datasets/can/ph/low_dim.hdf5
ls -la data/robomimic/datasets/can/ph/
# allow overriding number of parallel envs in eval
python - <<'EOF'
p='eval_combined_inference.py'; s=open(p).read()
k="    env_runner_cfg.n_test_vis = n_test_vis\n"
if 'N_ENVS' not in s:
    s=s.replace(k, k+"    if os.environ.get('N_ENVS'):\n        env_runner_cfg.n_envs = int(os.environ['N_ENVS'])\n",1)
    open(p,'w').write(s)
print('patched', 'N_ENVS' in s)
EOF
grep -n "^import os\|^import os," eval_combined_inference.py | head -2
DP=/workspace/ckpt/can_dp.ckpt
SEEDS="100000 100100 100200"
# 1) STEP predictor + variance head (joint; NLL uses detached residual, so the mean is the STEP predictor)
python train.py --config-name=train_action_predictor_lowdim_workspace task=can_lowdim logging.mode=disabled \
  policy.model.predict_var=True training.num_epochs=300 training.rollout_every=100000 training.checkpoint_every=50 \
  dataloader.num_workers=8 val_dataloader.num_workers=4 exp_name=can_pred hydra.run.dir=/workspace/runs/can_pred \
  > /workspace/train_can.log 2>&1
PRED=/workspace/runs/can_pred/checkpoints/latest.ckpt
ls -la $PRED || exit 1
echo TRAIN_DONE; date
python calibrate_uwarm.py --action_predictor_ckpt $PRED --modality lowdim --max_batches 50 \
  --deltas 0.003 0.01 0.03 0.1 0.3 1 --out /workspace/eval/calib_can.json
K=$(python -c "import json;print(json.load(open('/workspace/eval/calib_can.json'))['kappa'])")
echo KAPPA=$K
# 2) main: STEP / DDIM baselines + U-STEP (n_max=4)
python eval_combined_inference.py --action_predictor_ckpt $PRED --diffusion_policy_ckpt $DP --modality lowdim \
  --n_test 50 --n_test_vis 0 --steps_list 1 2 4 --use_ddim --eval_seeds $SEEDS \
  --ustep_deltas 0.003 0.01 0.03 0.1 0.3 1 --ustep_kappa $K --ustep_n_max 4 --output_dir /workspace/eval/can_main
echo MAIN_DONE; date
# 3) fixed-start (no per-sample adaptivity) at matched NFE
python experiments/fixed_start_sweep.py --action_predictor_ckpt $PRED --diffusion_policy_ckpt $DP \
  --grid 95:1 90:1 95:2 85:2 95:3 85:3 95:4 85:4 75:4 --kappa $K --eval_seeds $SEEDS --out_root /workspace/eval/can_fixed
echo FIXED_DONE; date
# 4) fidelity (offline)
python eval_distribution_fidelity.py --action_predictor_ckpt $PRED --diffusion_policy_ckpt $DP --modality lowdim \
  --history none --n_contexts 96 --k_samples 32 --deltas 0.01 0.03 0.1 --kappa $K --n_max 4 \
  --step_steps 1 2 4 --ddim_steps 2 4 --fixed 95:1 95:2 95:3 95:4 --out /workspace/eval/can_fidelity_none.json
echo PIPELINE_DONE; date
