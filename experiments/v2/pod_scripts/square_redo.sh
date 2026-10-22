#!/bin/bash
set -x
W=/workspace; cd $W/STEP; source $W/venv/bin/activate
export MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 N_ENVS=18
P=$W/runs/square_pred/checkpoints/latest.ckpt; DP=$W/ckpt/square_dp.ckpt
rm -rf $W/eval/square_fixed $W/eval/sim_square
python experiments/v2/sim_rollouts.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --modality lowdim --t0 90 --n 1 --n_test 12 --out $W/eval/sim_square > $W/logs/sim_square.log 2>&1
echo STAGE sim_square_redo; date
python experiments/fixed_start_sweep.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --grid 70:1 80:1 85:1 90:1 95:1 80:2 85:2 90:2 95:2 --n_test 50 --eval_seeds 100000 100100 --out_root $W/eval/square_fixed > $W/logs/sweep_square.log 2>&1
echo STAGE sweep_square_redo; date
echo SQUARE_REDO_DONE
