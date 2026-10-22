source /workspace/venv/bin/activate; export MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 N_ENVS=14
cd /workspace/STEP
for T in transport tool_hang; do
  until grep -q "^STAGE solver_$T" /workspace/logs/eval_all.log; do sleep 30; done
  python experiments/fixed_start_sweep.py --action_predictor_ckpt /workspace/runs/${T}_pred/checkpoints/latest.ckpt --diffusion_policy_ckpt /workspace/ckpt/${T}_dp.ckpt --grid 50:1 60:1 70:2 --n_test 50 --eval_seeds 100000 100100 --out_root /workspace/eval/${T}_fixed_low > /workspace/logs/sweep_${T}_low.log 2>&1
done
echo EXTRA3_DONE
