# Checkpoints

## pusht_lowdim/ (Push-T, low-dimensional observations)

| File | What it is |
| --- | --- |
| `step_pred.ckpt` | STEP action predictor, trained with the STEP recipe |
| `step_varhead.ckpt` | `step_pred` with a frozen mean and an added variance head. This is the predictor used for every Push-T (state) number in the paper. |
| `aligned_pred.ckpt` | Predictor trained with the aligned (forward-marginal) objective, an ablation |

The frozen Diffusion Policy is the official DP-CNN Push-T low-dim checkpoint, `epoch=0550-test_mean_score=0.969.ckpt` from the Diffusion Policy release. It is not stored here.

The RoboMimic and Consistency Policy checkpoints are still on the RunPod volume and are not yet in this repository. Files over 100 MB will go to GitHub Releases.
