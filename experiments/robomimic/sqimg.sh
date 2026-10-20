#!/bin/bash
set -x
export MUJOCO_GL=egl
export N_ENVS=28
cd /workspace/STEP
E=/workspace/eval
D=https://diffusion-policy.cs.columbia.edu/data
CD=$(dirname $(readlink -f data/robomimic/datasets/square/ph/low_dim_abs.hdf5))
# 1) render (runs alongside Can image eval)
python /workspace/prender.py $CD/low_dim_abs.hdf5 $CD/image_abs.hdf5 8 > /workspace/render_square.log 2>&1
python - $CD/image_abs.hdf5 <<'EOF'
import h5py, json, sys
with h5py.File(sys.argv[1], 'a') as f:
    m = json.loads(f['data'].attrs['env_args']); k = m['env_kwargs']
    k.update(dict(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                  camera_names=['agentview', 'robot0_eye_in_hand'], camera_heights=84, camera_widths=84, camera_depths=False))
    f['data'].attrs['env_args'] = json.dumps(m); print('env_args patched')
EOF
ln -sf $CD/image_abs.hdf5 data/robomimic/datasets/square/ph/image_abs.hdf5
F=$(curl -s $D/experiments/image/square_ph/diffusion_policy_cnn/train_0/checkpoints/ | grep -o 'href="epoch[^"]*ckpt' | head -1 | sed 's/href="//')
[ -f /workspace/ckpt/square_img_dp.ckpt ] || wget -q -O /workspace/ckpt/square_img_dp.ckpt "$D/experiments/image/square_ph/diffusion_policy_cnn/train_0/checkpoints/$F"
echo RENDER_DONE; date
# 2) train after the Can image eval releases the GPU/CPU
while ! grep -q '^CANIMG_DONE' /workspace/canimg_eval.log; do sleep 30; done
python train.py --config-name=train_action_predictor_image_workspace task=square_image_abs logging.mode=disabled \
  '~task.env_runner' policy.model.predict_var=True training.num_epochs=50 training.rollout_every=100000 \
  training.checkpoint_every=50 training.val_every=10 dataloader.num_workers=12 val_dataloader.num_workers=4 \
  hydra.run.dir=/workspace/runs/square_img_pred > /workspace/train_square_img_pred.log 2>&1
P=/workspace/runs/square_img_pred/checkpoints/latest.ckpt; DP=/workspace/ckpt/square_img_dp.ckpt
ls -la $P || exit 1
echo TRAIN_DONE; date
python calibrate_uwarm.py --action_predictor_ckpt $P --modality image --max_batches 30 --batch_size 64 \
  --deltas 0.01 0.03 0.1 --out $E/calib_square_img.json
K=$(python -c "import json;print(json.load(open('$E/calib_square_img.json'))['kappa'])"); echo "KAPPA square_img $K"
S3="100000 100100 100200"
# 3) the claims that matter: 1-NFE fixed start, 2-NFE fixed vs STEP-2 / DDIM-2
python experiments/fixed_start_sweep.py --modality image --action_predictor_ckpt $P --diffusion_policy_ckpt $DP \
  --grid 90:1 85:2 --kappa $K --eval_seeds $S3 --out_root $E/square_img_fixed
echo FIXED_DONE; date
python eval_combined_inference.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --modality image \
  --n_test 50 --n_test_vis 0 --steps_list 2 --use_ddim --eval_seeds $S3 --output_dir $E/square_img_main
echo SQIMG_DONE; date
