#!/bin/bash
# The official host (diffusion-policy.cs.columbia.edu) returns 404 since Oct 2026; these are
# Internet Archive captures of the same files.
set -x
mkdir -p /workspace/ckpt /workspace/data_dl
W=https://web.archive.org/web; D=https://diffusion-policy.cs.columbia.edu/data
cd /workspace
[ -s ckpt/pusht_dp.ckpt ] || wget -q -O ckpt/pusht_dp.ckpt "$W/20251124020734id_/$D/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt"
[ -s data_dl/pusht.zip ] || wget -q -O data_dl/pusht.zip "$W/20251124020734id_/$D/training/pusht.zip"
[ -s data_dl/rm_lowdim.zip ] || wget -q -O data_dl/rm_lowdim.zip "$W/20230614054522id_/$D/training/robomimic_lowdim.zip"
ls -la ckpt data_dl; echo DL_DONE
