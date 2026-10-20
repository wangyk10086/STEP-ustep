#!/bin/bash
set -x
cd /workspace
# ---- data + ckpt in background (parallel with pip) ----
mkdir -p /workspace/STEP/data/robomimic /workspace/ckpt
( wget -q -O /workspace/rm_lowdim.zip https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip && \
  python -c "
import zipfile
z=zipfile.ZipFile('/workspace/rm_lowdim.zip')
names=[n for n in z.namelist() if '/can/ph/' in n]
print(names)
for n in names: z.extract(n,'/workspace/rm_extract')
" && echo DATA_OK ) > /workspace/data.log 2>&1 &
( wget -q -O /workspace/ckpt/can_dp.ckpt "https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/can_ph/diffusion_policy_cnn/train_0/checkpoints/epoch%3D0350-test_mean_score%3D1.000.ckpt" && echo CKPT_OK ) > /workspace/ckpt.log 2>&1 &

apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libegl1 unzip > /dev/null
pip install -q "numpy==1.23.5" "numba==0.56.4" "scipy==1.10.1" "zarr==2.12.0" "numcodecs==0.10.2" "hydra-core==1.2.0" "omegaconf==2.2.3" "einops==0.4.1" "dill==0.3.5.1" "pymunk==6.2.1" "shapely<2" "pygame==2.1.2" "scikit-image==0.19.3" "opencv-python-headless==4.6.0.66" "wandb==0.13.3" "huggingface_hub==0.23.5" "diffusers==0.11.1" torch_ema "imageio==2.22.0" imageio-ffmpeg termcolor threadpoolctl "matplotlib==3.6.3" tqdm psutil "scikit-video==1.1.11" pytest "protobuf<4" "pandas<2" "h5py" "setuptools<66" "wheel<0.40"
pip install -q --no-build-isolation "gym==0.21.0"
# mujoco stack (DP conda env versions)
pip install -q "Cython<3" "free-mujoco-py==2.1.6" "mujoco==2.3.2" "glfw"
pip install -q --no-deps https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz
pip install -q "robomimic==0.2.0" --no-deps
pip install -q egl_probe --no-deps || true
pip install -q tensorboardX "termcolor" "numpy==1.23.5"
cd /workspace/STEP && pip install -q -e . --no-deps
export MUJOCO_GL=egl
python - <<'EOF'
import robosuite, robomimic, mujoco_py
print('robosuite', robosuite.__version__, 'robomimic', robomimic.__version__)
EOF
echo SETUP_DONE
