#!/bin/bash
# Remaining paper experiments with the OFFICIAL DP checkpoints (RoboMimic, PH, absolute actions):
#   (1) Table I RoboMimic rows with freshly trained predictors: STEP-2/4 and HiStart at t0=90/85/95 (n=1/2/4), 3 seeds
#   (2) a third seed for the best one-step level of each task
#   (3) start levels missing from the one-step sweep table
#   (4) HiStart with 4 NFE on Can and Square with image observations (images re-rendered from the stored states)
# Launch after setup_v2.sh and dl_robomimic.sh have been started:
#   nohup bash experiments/v2/run_missing.sh > /workspace/logs/missing.log 2>&1 &
set -x
W=/workspace; cd $W/STEP
until grep -q SETUP_DONE $W/logs/setup.log 2>/dev/null; do sleep 20; done
source $W/venv/bin/activate
export MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 N_ENVS=${N_ENVS:-14}
mkdir -p $W/eval $W/runs $W/logs $W/ckpt
S3="100000 100100 100200"; S2="100000 100100"
IMG=https://diffusion-policy.cs.columbia.edu/data/experiments/image

# ---------------- data ----------------
until grep -q ZIP_DONE $W/logs/zip.log 2>/dev/null; do sleep 20; done
python - <<'EOF'
import zipfile
z = zipfile.ZipFile('/workspace/rm_lowdim.zip')
for t in ['square', 'can', 'transport', 'tool_hang']:
    for n in z.namelist():
        if f'/{t}/ph/' in n and n.endswith('.hdf5'):
            z.extract(n, '/workspace/rm_extract'); print(n)
EOF
for T in square can transport tool_hang; do
  mkdir -p data/robomimic/datasets/$T/ph
  D=$(find $W/rm_extract -path "*/$T/ph" -type d | head -1)
  ln -sf $D/low_dim.hdf5 data/robomimic/datasets/$T/ph/low_dim.hdf5
  ln -sf $D/low_dim_abs.hdf5 data/robomimic/datasets/$T/ph/low_dim_abs.hdf5
done
until grep -q DL_ALL_DONE $W/logs/dl.log 2>/dev/null; do sleep 20; done
for T in can square; do
  F=$(curl -s $IMG/${T}_ph/diffusion_policy_cnn/train_0/checkpoints/ | grep -o 'href="epoch[^"]*ckpt' | head -1 | sed 's/href="//')
  echo "${T}_img $IMG/${T}_ph/diffusion_policy_cnn/train_0/checkpoints/$F" >> $W/eval/ckpt_sources.txt
  [ -s $W/ckpt/${T}_img_dp.ckpt ] || aria2c -q -x 8 -s 8 -k 20M --file-allocation=none -d $W/ckpt -o ${T}_img_dp.ckpt \
    "$IMG/${T}_ph/diffusion_policy_cnn/train_0/checkpoints/$F"
done
ls -la $W/ckpt
echo STAGE data; date

# ---------------- image observations, re-rendered from the stored states (background) ----------------
(
for T in can square; do
  CD=$(dirname $(readlink -f data/robomimic/datasets/$T/ph/low_dim_abs.hdf5))
  python -c "import h5py,sys; h5py.File(sys.argv[1],'r')['data']" $CD/image_abs.hdf5 2>/dev/null || { rm -rf $CD/image_abs.hdf5 $CD/image_abs.hdf5.parts; python experiments/robomimic/render_image_obs.py $CD/low_dim_abs.hdf5 $CD/image_abs.hdf5 6; }
  python - $CD/image_abs.hdf5 <<'EOF'
import h5py, json, sys
with h5py.File(sys.argv[1], 'a') as f:
    m = json.loads(f['data'].attrs['env_args']); k = m['env_kwargs']
    k.update(dict(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                  camera_names=['agentview', 'robot0_eye_in_hand'], camera_heights=84, camera_widths=84, camera_depths=False))
    f['data'].attrs['env_args'] = json.dumps(m); print('env_args patched')
EOF
  ln -sf $CD/image_abs.hdf5 data/robomimic/datasets/$T/ph/image_abs.hdf5
  echo "RENDERED $T"; date
done
) > $W/logs/render.log 2>&1 &

# ---------------- branch A: predictors (GPU), sequential ----------------
epochs() { case $1 in square|can) echo 300;; transport) echo 100;; tool_hang) echo 70;; esac; }
train_low() {
  [ -f $W/runs/$1_pred/checkpoints/latest.ckpt ] && { echo "TRAINED $1"; return; }
  python train.py --config-name=train_action_predictor_lowdim_workspace task=$1_lowdim_abs logging.mode=disabled \
    '~task.env_runner' policy.model.predict_var=True training.num_epochs=$(epochs $1) training.rollout_every=100000 \
    training.checkpoint_every=25 training.val_every=5 dataloader.num_workers=6 val_dataloader.num_workers=2 \
    exp_name=$1_pred hydra.run.dir=$W/runs/$1_pred > $W/logs/train_$1.log 2>&1
  [ -f $W/runs/$1_pred/checkpoints/latest.ckpt ] && echo "TRAINED $1" || echo "TRAIN_FAILED $1"; date
}
train_img() {
  until grep -q "^RENDERED $1" $W/logs/render.log 2>/dev/null; do sleep 30; done
  [ -f $W/runs/$1_img_pred/checkpoints/latest.ckpt ] && { echo "TRAINED $1_img"; return; }
  python train.py --config-name=train_action_predictor_image_workspace task=$1_image_abs logging.mode=disabled \
    '~task.env_runner' policy.model.predict_var=True training.num_epochs=50 training.rollout_every=100000 \
    training.checkpoint_every=25 training.val_every=5 dataloader.num_workers=6 val_dataloader.num_workers=2 \
    hydra.run.dir=$W/runs/$1_img_pred > $W/logs/train_$1_img.log 2>&1
  [ -f $W/runs/$1_img_pred/checkpoints/latest.ckpt ] && echo "TRAINED $1_img" || echo "TRAIN_FAILED $1_img"; date
}
( for T in square can transport tool_hang; do train_low $T; done; for T in can square; do train_img $T; done ) \
  > $W/logs/train_all.log 2>&1 &

# ---------------- branch B: evaluation (CPU-bound), sequential ----------------
sweep() {  # task out grid seeds [modality] [pred_suffix] [dp_suffix]
  python experiments/fixed_start_sweep.py --modality ${5:-lowdim} \
    --action_predictor_ckpt $W/runs/$1${6:-}_pred/checkpoints/latest.ckpt --diffusion_policy_ckpt $W/ckpt/$1${7:-}_dp.ckpt \
    --grid $3 --n_test 50 --eval_seeds $4 --out_root $W/eval/$1_$2 > $W/logs/sweep_$1_$2.log 2>&1
}
best() { case $1 in square) echo 50;; can) echo 60;; transport) echo 50;; tool_hang) echo 80;; esac; }
missing() { case $1 in can) echo "30:1";; transport) echo "40:1 95:1";; tool_hang) echo "40:1 50:1 85:1 95:1";; esac; }
for T in square can transport tool_hang; do
  until grep -qE "^(TRAINED|TRAIN_FAILED) $T\$" $W/logs/train_all.log 2>/dev/null; do sleep 30; done
  P=$W/runs/${T}_pred/checkpoints/latest.ckpt; DP=$W/ckpt/${T}_dp.ckpt
  [ -f $P ] || { echo "SKIP $T (no predictor)"; continue; }
  # (1) STEP-2/4 with this predictor (DDIM and predictor-only do not change, so they are skipped)
  SKIP_DDIM=1 SKIP_AP=1 python eval_combined_inference.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP \
    --modality lowdim --n_test 50 --n_test_vis 0 --steps_list 2 4 --use_ddim --eval_seeds $S3 \
    --output_dir $W/eval/${T}_step > $W/logs/step_$T.log 2>&1
  echo STAGE step_$T; date
  # (1) HiStart at the global Table-I levels
  sweep $T tab1 "90:1 85:2 95:4" "$S3"; echo STAGE tab1_$T; date
  # (2) third seed at the best one-step level
  sweep $T best_s3 "$(best $T):1" "100200"; echo STAGE best_$T; date
  # (3) missing one-step levels
  M=$(missing $T); [ -n "$M" ] && { sweep $T missing "$M" "$S2"; echo STAGE missing_$T; date; }
done
# (4) image observations, 4 NFE
for T in can square; do
  until grep -qE "^(TRAINED|TRAIN_FAILED) ${T}_img" $W/logs/train_all.log 2>/dev/null; do sleep 30; done
  [ -f $W/runs/${T}_img_pred/checkpoints/latest.ckpt ] || { echo "SKIP ${T}_img"; continue; }
  sweep $T img4 "95:4" "$S3" image _img _img; echo STAGE img4_$T; date
done
wait

# ---------------- bundle, then stop the pod once fetched (or after 3 h) ----------------
cd $W && find eval -name '*.mp4' > /tmp/ex.txt; tar czf rm_missing.tar.gz -X /tmp/ex.txt eval logs STEP/experiments/v2/run_missing.sh
ls -la $W/rm_missing.tar.gz
echo ALL_DONE; date
for i in $(seq 1 360); do [ -f $W/FETCHED ] && break; sleep 30; done
runpodctl stop pod $RUNPOD_POD_ID
