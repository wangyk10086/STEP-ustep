# RoboMimic / Push-T image experiments (U-STEP follow-up)

These are the scripts behind the results in `results/RESULTS.md` and `results/all_closed_loop_results.csv`. They ran on a single RTX 4090 pod (Python 3.10, PyTorch 2.2).

## Environment notes
- `setup_rm.sh` installs the pinned stack: free-mujoco-py 2.1.6, the `cheng-chi/robosuite` fork, and robomimic 0.2.0.
- On Python 3.10 the robosuite fork needs one patch:
  - in `robosuite/models/arenas/multi_table_arena.py`, change `from collections import Iterable` to `from collections.abc import Iterable`;
  - in `robosuite/utils/placement_samplers.py`, change `collections.Iterable` to `collections.abc.Iterable`.
- `pytorch3d` is optional. `diffusion_policy/model/common/pt_shim.py` implements the four rotation transforms that are needed, and its round-trip error is about 1e-7.
- Image tasks also need `imagecodecs<2024` (the zarr cache codec) and `scikit-learn` (used by Consistency Policy).
- If robomimic's `env_robosuite.py` fails on `import egl_probe`, wrap that import in try/except and fall back to GPU 0.
- Official DP checkpoints for RoboMimic use **absolute actions**, so predictors are trained with `task=<task>_lowdim_abs` / `<task>_image_abs`.
- Set `N_ENVS=28` or lower when evaluating. Each simulator worker uses about 1.5 GB, and 56 workers exceeded a 41 GB container limit.

## Image observations
The official `robomimic_image.zip` is 85 GB. Instead, `render_image_obs.py SRC_low_dim_abs.hdf5 DST_image_abs.hdf5 8` re-renders the 84×84 `agentview` and `robot0_eye_in_hand` images from the stored simulator states, in parallel. Afterwards, patch `env_args` in the output hdf5 to enable the cameras; `sqimg.sh` shows the patch.

## Pipelines
| Script | What it runs |
| --- | --- |
| `can_pipeline.sh` | Can lowdim: predictor + variance head, calibration, STEP/DDIM/U-STEP, fixed-start sweep, fidelity |
| `hard_pipeline.sh` | Square, Transport and ToolHang lowdim (predictor training runs alongside evaluation) |
| `night.sh`, `night2.sh` | Push-T image, extra seeds for Can/Square, Square fidelity, low-t0 sweep for Transport/ToolHang |
| `sqimg.sh`, `sqeval.sh`, `canimg_eval.sh` | Square image / Can image predictor training and evaluation |
| `../consistency_policy/run_cp_square_image.sh` | Official Consistency Policy (EDM teacher for 400 epochs, then CTM student) plus `cp_eval.py` (1-step and 3-step, 3 seeds) |

Paths in the scripts assume `/workspace/STEP` and `/workspace/ckpt` on the pod.
