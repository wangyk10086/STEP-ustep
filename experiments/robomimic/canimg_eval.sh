#!/bin/bash
set -x
export MUJOCO_GL=egl
export N_ENVS=28
cd /workspace/STEP
E=/workspace/eval
H=$(readlink -f data/robomimic/datasets/can/ph/image_abs.hdf5)
python - $H <<'EOF'
import h5py, json, sys
with h5py.File(sys.argv[1], 'a') as f:
    m = json.loads(f['data'].attrs['env_args'])
    k = m['env_kwargs']
    k.update(dict(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                  camera_names=['agentview', 'robot0_eye_in_hand'], camera_heights=84, camera_widths=84,
                  camera_depths=False))
    f['data'].attrs['env_args'] = json.dumps(m)
    print(json.dumps(k)[:400])
EOF
P=/workspace/runs/can_img_pred/checkpoints/latest.ckpt; DP=/workspace/ckpt/can_img_dp.ckpt
K=$(python -c "import json;print(json.load(open('$E/calib_can_img.json'))['kappa'])"); echo "KAPPA can_img $K"
S3="100000 100100 100200"
rm -rf $E/can_img_main $E/can_img_fixed
python eval_combined_inference.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --modality image \
  --n_test 50 --n_test_vis 0 --steps_list 2 4 --use_ddim --eval_seeds $S3 \
  --ustep_deltas 1 --ustep_kappa $K --ustep_n_max 4 --output_dir $E/can_img_main
echo MAIN_DONE; date
python experiments/fixed_start_sweep.py --modality image --action_predictor_ckpt $P --diffusion_policy_ckpt $DP \
  --grid 90:1 85:2 --kappa $K --eval_seeds $S3 --out_root $E/can_img_fixed
echo CANIMG_DONE; date
