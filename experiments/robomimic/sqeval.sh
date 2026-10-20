#!/bin/bash
set -x
export MUJOCO_GL=egl
export N_ENVS=12
cd /workspace/STEP
E=/workspace/eval
P=/workspace/runs/square_img_pred/checkpoints/latest.ckpt; DP=/workspace/ckpt/square_img_dp.ckpt
K=$(python -c "import json;print(json.load(open('$E/calib_square_img.json'))['kappa'])"); echo "KAPPA square_img $K"
S3="100000 100100 100200"
rm -rf $E/square_img_fixed $E/square_img_main
python experiments/fixed_start_sweep.py --modality image --action_predictor_ckpt $P --diffusion_policy_ckpt $DP \
  --grid 90:1 85:2 --kappa $K --eval_seeds $S3 --out_root $E/square_img_fixed
echo FIXED_DONE; date
python eval_combined_inference.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --modality image \
  --n_test 50 --n_test_vis 0 --steps_list 1 2 --use_ddim --eval_seeds $S3 --output_dir $E/square_img_main
echo SQIMG_DONE; date
# Can image leftovers
P=/workspace/runs/can_img_pred/checkpoints/latest.ckpt; DP=/workspace/ckpt/can_img_dp.ckpt
K=$(python -c "import json;print(json.load(open('$E/calib_can_img.json'))['kappa'])")
python eval_combined_inference.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --modality image \
  --n_test 50 --n_test_vis 0 --steps_list 1 --use_ddim --eval_seeds $S3 --output_dir $E/can_img_main_s1
python experiments/fixed_start_sweep.py --modality image --action_predictor_ckpt $P --diffusion_policy_ckpt $DP \
  --grid 85:2 --kappa $K --eval_seeds $S3 --out_root $E/can_img_fixed2
echo CANFIX_DONE; date
