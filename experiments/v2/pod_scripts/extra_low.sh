source /workspace/venv/bin/activate; export MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 N_ENVS=14
cd /workspace/STEP
python experiments/fixed_start_sweep.py --action_predictor_ckpt /workspace/runs/square_pred/checkpoints/latest.ckpt --diffusion_policy_ckpt /workspace/ckpt/square_dp.ckpt --grid 50:1 60:1 70:2 --n_test 50 --eval_seeds 100000 100100 --out_root /workspace/eval/square_fixed_low > /workspace/logs/sweep_square_low.log 2>&1
echo EXTRA_DONE
