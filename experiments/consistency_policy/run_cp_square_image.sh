#!/bin/bash
# Official Consistency Policy on Robomimic Square image (PH): EDM teacher (400 ep) -> CTM student (500 ep)
set -x
export MUJOCO_GL=egl
cd /workspace
[ -d CP ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 https://github.com/Aaditya-Prasad/consistency-policy CP
cd /workspace/CP
git rev-parse HEAD
[ -e data ] || ln -s /workspace/STEP/data data
while ! grep -q '^RENDER_DONE' /workspace/sqimg.log; do sleep 30; done
ls -la data/robomimic/datasets/square/ph/
# only change vs. official configs: no wandb, more parallel rollout envs, no videos (same n_test / schedule / epochs)
COMMON="logging.mode=disabled task.env_runner.n_envs=14 task.env_runner.n_test_vis=0 task.env_runner.n_train_vis=0 dataloader.num_workers=8 val_dataloader.num_workers=4"
if [ ! -f /workspace/cp_out/edm/checkpoints/latest.ckpt ] || ! grep -q '^TEACHER_DONE' /workspace/cp.log; then
python train.py --config-dir=configs/ --config-name=edm_square.yaml $COMMON \
  training.output_dir=/workspace/cp_out/edm hydra.run.dir=/workspace/cp_out/edm > /workspace/cp_teacher.log 2>&1
fi
ls -la /workspace/cp_out/edm/checkpoints/
echo TEACHER_DONE; date
T=/workspace/cp_out/edm/checkpoints/latest.ckpt
python train.py --config-dir=configs/ --config-name=ctmp_square.yaml $COMMON \
  policy.teacher_path=$T policy.edm=$T \
  training.output_dir=/workspace/cp_out/ctm hydra.run.dir=/workspace/cp_out/ctm > /workspace/cp_student.log 2>&1
ls -la /workspace/cp_out/ctm/checkpoints/
echo STUDENT_DONE; date
