#!/bin/bash
# Overnight queue: Push-T image, extra seeds (Can/Square), Square fidelity, Can image (state-rendered)
set -x
export MUJOCO_GL=egl
export N_ENVS=28
cd /workspace/STEP
D=https://diffusion-policy.cs.columbia.edu/data
E=/workspace/eval
log() { echo "NIGHT $*"; date; }

# ---------- Push-T image: data, DP ckpt, predictor (runs alongside the hard pipeline) ----------
mkdir -p data
[ -d data/pusht ] || ( wget -q -O /workspace/pusht.zip $D/training/pusht.zip && python -c "import zipfile;zipfile.ZipFile('/workspace/pusht.zip').extractall('data')" )
ls data/pusht
F=$(curl -s $D/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/ | grep -o 'href="epoch[^"]*ckpt' | head -1 | sed 's/href="//')
[ -f /workspace/ckpt/pusht_img_dp.ckpt ] || wget -q -O /workspace/ckpt/pusht_img_dp.ckpt "$D/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/$F"
trainimg() {  # name task epochs
  [ -f /workspace/runs/$1/checkpoints/latest.ckpt ] && return
  python train.py --config-name=train_action_predictor_image_workspace task=$2 logging.mode=disabled \
    '~task.env_runner' policy.model.predict_var=True training.num_epochs=$3 training.rollout_every=100000 \
    training.checkpoint_every=50 training.val_every=10 dataloader.num_workers=8 val_dataloader.num_workers=4 \
    exp_name=$1 hydra.run.dir=/workspace/runs/$1 > /workspace/train_$1.log 2>&1
  log "TRAINED $1"
}
trainimg pusht_img_pred pusht_image 150

# ---------- wait for the hard-task pipeline ----------
while ! grep -q '^HARD_PIPELINE_DONE' /workspace/hard_pipeline.log; do sleep 60; done
log "HARD_DONE seen"

# ---------- Push-T image eval (5 seeds) ----------
P=/workspace/runs/pusht_img_pred/checkpoints/latest.ckpt; DP=/workspace/ckpt/pusht_img_dp.ckpt
S5="100000 100100 100200 100300 100400"
python calibrate_uwarm.py --action_predictor_ckpt $P --modality image --max_batches 30 --batch_size 64 \
  --deltas 0.01 0.03 0.1 0.3 1 --out $E/calib_pusht_img.json
K=$(python -c "import json;print(json.load(open('$E/calib_pusht_img.json'))['kappa'])"); echo "KAPPA pusht_img $K"
python eval_combined_inference.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --modality image \
  --n_test 50 --n_test_vis 0 --steps_list 1 2 4 --use_ddim --eval_seeds $S5 \
  --ustep_deltas 0.03 0.3 1 --ustep_kappa $K --ustep_n_max 4 --output_dir $E/pusht_img_main
log "MAIN_DONE pusht_img"
python experiments/fixed_start_sweep.py --modality image --action_predictor_ckpt $P --diffusion_policy_ckpt $DP \
  --grid 95:1 90:1 95:2 90:2 85:2 95:4 --kappa $K --eval_seeds $S5 --out_root $E/pusht_img_fixed
log "FIXED_DONE pusht_img"

# ---------- extra seeds (100300, 100400) for Can + Square key configs -> 5 seeds total ----------
S2="100300 100400"
for T in can square; do
  P=/workspace/runs/${T}_pred/checkpoints/latest.ckpt; DP=/workspace/ckpt/${T}_dp.ckpt
  K=$(python -c "import json;print(json.load(open('$E/calib_$T.json'))['kappa'])")
  python eval_combined_inference.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --modality lowdim \
    --n_test 50 --n_test_vis 0 --steps_list 2 4 --use_ddim --eval_seeds $S2 \
    --ustep_deltas 0.3 1 --ustep_kappa $K --ustep_n_max 4 --output_dir $E/${T}_main_s2
  python experiments/fixed_start_sweep.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP \
    --grid 90:1 95:2 85:2 95:4 --kappa $K --eval_seeds $S2 --out_root $E/${T}_fixed_s2
  log "SEEDS_DONE $T"
done

# ---------- offline fidelity on Square ----------
P=/workspace/runs/square_pred/checkpoints/latest.ckpt; DP=/workspace/ckpt/square_dp.ckpt
K=$(python -c "import json;print(json.load(open('$E/calib_square.json'))['kappa'])")
python eval_distribution_fidelity.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --modality lowdim \
  --history none --n_contexts 96 --k_samples 32 --deltas 0.03 0.3 --kappa $K --n_max 4 \
  --step_steps 1 2 4 --ddim_steps 2 4 --fixed 90:1 95:2 85:2 95:4 --out $E/square_fidelity_none.json
log "FID_DONE square"

# ---------- Can image (obs rendered from simulator states at 84x84) ----------
CD=$(dirname $(readlink -f data/robomimic/datasets/can/ph/low_dim_abs.hdf5))
RS=$(python -c "import robomimic,os;print(os.path.join(os.path.dirname(robomimic.__file__),'scripts','dataset_states_to_obs.py'))")
if [ ! -f $CD/image_abs.hdf5 ]; then
  python $RS --dataset $CD/low_dim_abs.hdf5 --output_name image_abs.hdf5 --done_mode 2 \
    --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 > /workspace/render_can.log 2>&1
fi
ln -sf $CD/image_abs.hdf5 data/robomimic/datasets/can/ph/image_abs.hdf5
ls -la $CD
F=$(curl -s $D/experiments/image/can_ph/diffusion_policy_cnn/train_0/checkpoints/ | grep -o 'href="epoch[^"]*ckpt' | head -1 | sed 's/href="//')
[ -f /workspace/ckpt/can_img_dp.ckpt ] || wget -q -O /workspace/ckpt/can_img_dp.ckpt "$D/experiments/image/can_ph/diffusion_policy_cnn/train_0/checkpoints/$F"
trainimg can_img_pred can_image_abs 100
P=/workspace/runs/can_img_pred/checkpoints/latest.ckpt; DP=/workspace/ckpt/can_img_dp.ckpt
S3="100000 100100 100200"
python calibrate_uwarm.py --action_predictor_ckpt $P --modality image --max_batches 30 --batch_size 64 \
  --deltas 0.01 0.03 0.1 0.3 1 --out $E/calib_can_img.json
K=$(python -c "import json;print(json.load(open('$E/calib_can_img.json'))['kappa'])"); echo "KAPPA can_img $K"
python eval_combined_inference.py --action_predictor_ckpt $P --diffusion_policy_ckpt $DP --modality image \
  --n_test 50 --n_test_vis 0 --steps_list 2 4 --use_ddim --eval_seeds $S3 \
  --ustep_deltas 0.3 1 --ustep_kappa $K --ustep_n_max 4 --output_dir $E/can_img_main
log "MAIN_DONE can_img"
python experiments/fixed_start_sweep.py --modality image --action_predictor_ckpt $P --diffusion_policy_ckpt $DP \
  --grid 90:1 85:2 95:2 --kappa $K --eval_seeds $S3 --out_root $E/can_img_fixed
log "FIXED_DONE can_img"
echo NIGHT_ALL_DONE; date
