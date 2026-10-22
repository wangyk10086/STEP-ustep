#!/bin/bash
# Fresh pod setup (Blackwell GPUs need torch>=2.7 / cu128). Everything lives on /workspace (persistent volume).
set -x
cd /workspace
mkdir -p ckpt data_dl logs runs eval
[ -d STEP ] || git clone -b ustep https://github.com/wangyk10086/STEP-ustep.git STEP
D=https://diffusion-policy.cs.columbia.edu/data
# checkpoints and data come from the Internet Archive copy of the official site: see dl_wayback.sh
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libegl1 unzip ffmpeg > /dev/null
# persistent venv on the volume, so a restarted pod only needs the apt packages
[ -x /workspace/venv/bin/python ] || python -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install -q "pip<24.1"   # gym 0.21 does not build with newer pip
python -c "import torch" 2>/dev/null && [ -f /workspace/venv/.deps_ok ] && { echo VENV_REUSED; export MUJOCO_GL=egl; touch /workspace/venv/.deps_ok
echo SETUP_DONE; exit 0; }
pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -q "numpy==1.23.5" "numba==0.56.4" "scipy==1.10.1" "zarr==2.12.0" "numcodecs==0.10.2" "hydra-core==1.2.0" "omegaconf==2.2.3" "einops==0.4.1" "dill==0.3.5.1" "pymunk==6.2.1" "shapely<2" "pygame==2.1.2" "scikit-image==0.19.3" "opencv-python-headless==4.6.0.66" "wandb==0.13.3" "huggingface_hub==0.23.5" "diffusers==0.11.1" torch_ema "imageio==2.22.0" imageio-ffmpeg termcolor threadpoolctl "matplotlib==3.6.3" tqdm psutil "scikit-video==1.1.11" pytest "protobuf<4" "pandas<2" h5py "setuptools<66" "wheel<0.40" "imagecodecs<2024" scikit-learn
pip install -q --no-build-isolation "gym==0.21.0"
pip install -q "Cython<3" "free-mujoco-py==2.1.6" "mujoco==2.3.2" glfw
pip install -q --no-deps https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz
pip install -q "robomimic==0.2.0" --no-deps
pip install -q egl_probe --no-deps || true
pip install -q tensorboardX "numpy==1.23.5"
pip install -q --only-binary :all: "av==12.3.0"   # video writer used by the env runners
# robosuite fork on py3.10
RS=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")/robosuite  # importing robosuite fails before the patch
sed -i 's/from collections import Iterable/from collections.abc import Iterable/' $RS/models/arenas/multi_table_arena.py
sed -i 's/collections\.Iterable/collections.abc.Iterable/g' $RS/utils/placement_samplers.py
RM=$(python -c "import robomimic,os;print(os.path.dirname(robomimic.__file__))" 2>/dev/null)
python - "$RM/envs/env_robosuite.py" <<'PY'
import sys; p=sys.argv[1]; s=open(p).read()
old = "                import egl_probe\n                valid_gpu_devices = egl_probe.get_available_devices()\n"
new = ("                try:\n                    import egl_probe\n                    valid_gpu_devices = egl_probe.get_available_devices()\n"
       "                except Exception:\n                    valid_gpu_devices = [0]\n")
if old in s:
    open(p, 'w').write(s.replace(old, new, 1))
PYcd /workspace/STEP && pip install -q -e . --no-deps
export MUJOCO_GL=egl
python - <<'PY'
import torch, robosuite, robomimic, mujoco_py
print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())
x=torch.randn(64,64,device='cuda'); print('matmul ok', (x@x).sum().item())
print('robosuite', robosuite.__version__, 'robomimic', robomimic.__version__)
PY
touch /workspace/venv/.deps_ok
echo SETUP_DONE
