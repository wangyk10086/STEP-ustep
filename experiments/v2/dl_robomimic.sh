#!/bin/bash
# Parallel download of the official RoboMimic low-dim data and DP-CNN checkpoints (train_0, best epoch).
cd /workspace; mkdir -p logs ckpt eval
which aria2c || (apt-get update -qq && apt-get install -y -qq aria2 > /dev/null 2>&1)
aria2c -q -x 8 -s 8 -k 20M --file-allocation=none -d /workspace -o rm_lowdim.zip \
  https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip && echo ZIP_DONE >> /workspace/logs/zip.log
B=https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim
for p in "square epoch%3D1750-test_mean_score%3D1.000.ckpt" "can epoch%3D0350-test_mean_score%3D1.000.ckpt" \
         "transport epoch%3D2100-test_mean_score%3D0.955.ckpt" "tool_hang epoch%3D0850-test_mean_score%3D0.818.ckpt"; do
  set -- $p
  echo "$1 $B/$1_ph/diffusion_policy_cnn/train_0/checkpoints/$2" >> /workspace/eval/ckpt_sources.txt
  aria2c -q -x 8 -s 8 -k 20M --file-allocation=none -d /workspace/ckpt -o $1_dp.ckpt \
    "$B/$1_ph/diffusion_policy_cnn/train_0/checkpoints/$2" && echo "CKPT_OK $1"
done
echo DL_ALL_DONE
