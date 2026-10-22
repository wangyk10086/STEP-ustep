# Start High, Step Once: One-Step Warm-Started Diffusion Policies (HiStart)

<p align="center">
  <img src="media/histart_pusht.gif" width="600" alt="Push-T, one network evaluation per action chunk: DDIM | STEP | HiStart (ours)">
  <br><em>Push-T, one network evaluation per action chunk, same initial state.
  Left to right: DDIM, STEP, HiStart (ours). Full video: <a href="video/pusht_one_step.mp4">video/pusht_one_step.mp4</a></em>
</p>
<p align="center">
  <img src="media/histart_square.gif" width="600" alt="Square (RoboMimic), one network evaluation: DDIM | STEP | HiStart (ours)">
  <br><em>RoboMimic Square with the official DP checkpoint, one network evaluation, same initial state.
  Over 12 episodes: DDIM 0, STEP 0, HiStart 9 successes. Full video: <a href="video/square_one_step.mp4">video/square_one_step.mp4</a></em>
</p>

HiStart is a training-free inference rule for pretrained Diffusion Policies (DP).
It builds on [STEP](STEP_README.md) (warm-starting DP from a predicted action chunk) and asks one question:
**at which noise level should the warm-started reverse process begin?**

STEP refines the predicted chunk from a low noise level (`t = 0` for one step, `t = 50` for two), so the
predictor's error survives the short reverse process and one-step success collapses to DDIM's level.
HiStart instead

1. forward-noises the predicted chunk `â` with the **exact forward marginal**
   `x_{t0} = sqrt(ᾱ_{t0}) · â + sqrt(1 − ᾱ_{t0}) · ε`, at a **high** start level `t0 ≈ 85–95`, and
2. runs `n = 1–4` DDIM steps of the **unchanged** pretrained DP.

No distillation, no teacher/student, no fine-tuning of the policy.

## Why a high start level

We bound the error of a warm start by two terms:

```
B(t0, n) = ½ · sqrt(SNR(t0) · m)  +  C(t0, n)
           └── start mismatch ──┘    └ solver ┘
```

- `m` is the predictor's mean-squared error.
- The **mismatch** term shrinks as `t0` rises.
- The **solver** term grows with the integration range per step; for DDIM, `C = ½ κ L` with `κ = nΣh²/Λ² ∈ [1, n]`.

Their sum is minimized at an intermediate `t0*`. It rises with the step budget `n` and with the predictor
error `m`, and it lies far above the level STEP uses.

`t0*` can be chosen **offline, without rollouts**:
1. Measure the solver term for each `(t0, n)` from the exact forward marginal of demonstrated actions.
2. Measure the predictor error `m` without its history input. This is a conservative proxy for closed-loop error.
3. Minimize `B`.

On Push-T this takes about one minute on one GPU.

## Results

Closed-loop success, mean over 5 seeds, frozen public DP checkpoints. Best per budget in **bold**.
† is reported by the STEP paper with its own checkpoints.

| Task | DDIM-1 | STEP-1 | **Ours-1** | DDIM-2 | STEP-2 | **Ours-2** | DDIM-4 | STEP-4 | **Ours-4** |
|---|---|---|---|---|---|---|---|---|---|
| Push-T | .12 | .09 | **.69** | .19 | .49† | **.82** | .71 | **.91†** | .89 |
| Push-T (img) | .10 | .10 | **.59** | .63 | **.75** | **.75** | .77 | .80 | **.81** |
| Can | .00 | .00 | **.28** | .41 | .96 | **.98** | .98 | .98 | **.99** |
| Can (img) | .00 | .00 | **.81** | .39 | **.99** | **.99** | – | – | – |
| Square | .00 | .00 | **.74** | .72 | .92 | **.94** | .90 | **.94** | .92 |
| Square (img) | .00 | .00 | **.83** | .62 | .93 | **.94** | – | – | – |
| Transport | – | – | .26 | .13 | **.86** | .85 | .80 | .82 | **.85** |
| ToolHang | – | – | .14 | .05 | **.62** | .51 | **.57** | .55 | .47 |

**Offline prediction of `t0*` on Push-T** (no rollouts used):

| n | predicted t0* | closed-loop best t0 | success at predicted t0* |
|---|---|---|---|
| 1 | 85 | 80–85 | .70 |
| 2 | 95 | 95 | .86 |
| 3 | 95 | 95 | .89 |
| 4 | 95 | 95 | .89 |

**RoboMimic with the official DP checkpoints** (low-dim, PH, absolute actions; predictors retrained with the STEP
recipe; closed loop, 2 seeds x 50 episodes). One-step success vs start level `t0`:

| Task | 40 | 50 | 60 | 70 | 80 | 85 | 90 | 95 |
|---|---|---|---|---|---|---|---|---|
| Square | .88 | **.90** | .85 | .88 | .83 | .87 | .75 | .12 |
| Can | .87 | .95 | **.98** | .94 | .91 | .72 | .22 | .00 |
| Transport | – | **.84** | .83 | .80 | .60 | .47 | .22 | – |
| ToolHang | – | – | .59 | .57 | **.66** | – | .17 | – |

DDIM-1 and STEP-1 reach 0 on all four tasks. One-step success is flat over a wide range and collapses near the top
(t0 >= 90). The offline bound picks t0 = 95 for two steps on every task, within 0.04 of the best level tested. For one
step it only locates the collapse: with the conservative `m` it picks 90 and loses 0.15–0.76 of success. Minimizing the
measured warm-start error directly picks 60–70 on Square, Transport and ToolHang, which loses 0.01–0.09. Raw numbers
are in `results/v2/robomimic/`.

**Latency** (Push-T lowdim, batch 1, RTX PRO 5000, ms per action chunk):

| DDPM-100 | DDIM-1 | DDIM-2 | DDIM-4 | STEP-1 | STEP-2 | STEP-4 | Ours-1 |
|---|---|---|---|---|---|---|---|
| 710 | 7.6 | 14.5 | 28.5 | 12.3 | 19.4 | 33.2 | 12.3–13.1 |

With one evaluation, HiStart costs the same as STEP-1 and is about 57× faster than 100-step DDPM.

## Quick start

The environment is the same as STEP and Diffusion Policy (`conda_environment.yaml`).
For a fresh Linux GPU machine with a recent CUDA, `experiments/v2/setup_v2.sh` builds a venv with torch 2.7 / cu128.
With torch ≥ 2.6, set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` to load the dill-pickled DP checkpoints.

**Checkpoints**
- Push-T lowdim action predictors are included in `checkpoints/pusht_lowdim/`.
  `step_varhead.ckpt` is the one used for all Push-T numbers.
- The official DP checkpoints come from the Diffusion Policy release.
  Its download site is currently offline; `experiments/v2/dl_wayback.sh` fetches the Push-T checkpoint from the Internet Archive.

**1. Pick the start level offline**

```bash
python experiments/v2/solver_error.py \
  --action_predictor_ckpt checkpoints/pusht_lowdim/step_varhead.ckpt \
  --diffusion_policy_ckpt /path/to/pusht_dp.ckpt \
  --history none --t0 50 60 70 80 85 90 95 --n 1 2 3 4 \
  --out results/solver_pusht.json
```

**2. Run HiStart in closed loop.** This example uses `t0 = 90` and one step, and records videos.

```bash
python experiments/v2/sim_rollouts.py \
  --action_predictor_ckpt checkpoints/pusht_lowdim/step_varhead.ckpt \
  --diffusion_policy_ckpt /path/to/pusht_dp.ckpt \
  --modality lowdim --t0 90 --n 1 --n_test 16 --out results/sim_pusht
```

It runs DDIM-1, STEP-1 and HiStart-1 from the same initial states and writes `manifest.json` with per-episode coverage and video paths.

**3. Latency**

```bash
python bench_latency.py --action_predictor_ckpt ... --diffusion_policy_ckpt ... --steps 1 2 4 --n_calls 300
```

To run everything at once, use `experiments/v2/run_pusht.sh` for Push-T. For RoboMimic, run
`experiments/v2/dl_robomimic.sh` (official data and checkpoints, parallel download), then `experiments/v2/run_robomimic.sh`.
The scripts that actually ran on the pod for the reported RoboMimic numbers are in `experiments/v2/pod_scripts/`.

## Repository layout (additions on top of STEP)

```text
diffusion_policy/policy/adaptive_warmstart.py   # exact forward-marginal warm start + DDIM schedule from t0
experiments/v2/
├── solver_error.py      # offline solver term and predictor error -> predicted t0*
├── sim_rollouts.py      # DDIM / STEP / HiStart rollouts with video from identical initial states
├── run_pusht.sh, run_all.sh, setup_v2.sh, dl_wayback.sh
paper_plotting/          # figures and tables of the paper (STEP-paper style), video maker
results/v2/              # raw JSON results behind the Push-T follow-up numbers
results/v2/robomimic/    # RoboMimic: offline solver error, fixed-start sweeps, Square rollout videos, logs
checkpoints/pusht_lowdim # Push-T action predictors
video/pusht_one_step.mp4, video/square_one_step.mp4  # supplementary videos
STEP_README.md           # original STEP documentation (training the predictor, real robot, etc.)
```

## Reproducing the figures

```bash
cd paper_plotting
python clean_data.py && python figures.py && python tables.py
python sim_figure.py --manifest <sim_out>/manifest.json --video_root <sim_out> --seed 100014 --out fig_sim
python make_video.py --manifest <sim_out>/manifest.json --video_root <sim_out> \
  --seeds 100014 100003 100002 100006 --out pusht_one_step.mp4
```

## Acknowledgements

This code builds on [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) and STEP.
The license is unchanged (see `LICENSE`).
