"""
Shared helpers for U-STEP offline analysis (calibration, distribution fidelity).
"""

import copy
from typing import Dict, Optional, Tuple

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.aligned_prev_action_wrapper import AlignedPrevActionWrapper
from diffusion_policy.policy.action_predictor_lowdim_policy import ActionPredictorDatasetWrapper


def build_eval_dataset(cfg, split: str = 'val', aligned: Optional[bool] = None):
    """Instantiate the task dataset from a predictor checkpoint cfg and wrap it
    the same way the predictor was trained (or force `aligned`)."""
    base = hydra.utils.instantiate(cfg.task.dataset)
    if aligned is None:
        aligned = bool(cfg.get('aligned_prev_action', False))
    if aligned:
        ds = AlignedPrevActionWrapper(base, prev_action_horizon=cfg.prev_action_horizon,
                                      n_action_steps=cfg.n_action_steps, n_obs_steps=cfg.n_obs_steps)
    else:
        ds = ActionPredictorDatasetWrapper(base_dataset=base, prev_action_horizon=cfg.prev_action_horizon,
                                           n_action_steps=cfg.n_action_steps,
                                           pad_before=cfg.task.dataset.get('pad_before', 0))
    if split == 'val':
        val = ds.get_validation_dataset()
        if len(val) == 0:
            print('[U-STEP] validation split is empty (val_ratio=0); falling back to train split. '
                  'Set task.dataset.val_ratio>0 for a clean calibration.')
        else:
            ds = val
    return ds


def make_loader(ds, batch_size=256, num_workers=4, shuffle=True, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                      generator=g, pin_memory=True)


def is_image_predictor(policy) -> bool:
    return hasattr(policy, 'encoder') and hasattr(policy, '_encode_obs_seq')


@torch.no_grad()
def predictor_forward(policy, batch: Dict, device) -> Dict[str, torch.Tensor]:
    """Mirror of compute_loss without augmentation. Returns normalized
    mean/var/target and raw (unnormalized) mean for the diffusion warm start."""
    batch = dict_apply(batch, lambda x: x.to(device) if torch.is_tensor(x) else x)
    norm = policy.normalizer
    model = policy.model
    nprev = norm['action'].normalize(batch['prev_action'])
    if 'prev_valid' in batch:
        nprev = nprev * batch['prev_valid'].view(-1, 1, 1).to(nprev.dtype)
    if is_image_predictor(policy):
        feat = policy._encode_obs_seq(batch['obs'])
        nobs = norm['obs'].normalize(feat) if 'obs' in norm.params_dict else feat
    else:
        nobs = norm['obs'].normalize(batch['obs'])
    obs_in = nobs[:, :policy.n_obs_steps]
    target = norm['action'].normalize(batch['action'])[:, :model.pred_horizon]
    if getattr(model, 'predict_var', False):
        mean, logvar = model(nprev, obs_in, return_logvar=True)
        var = logvar.exp()
    else:
        mean, var = model(nprev, obs_in), None
    return {
        'mean': mean, 'var': var, 'target': target,
        'mean_raw': norm['action'].unnormalize(mean),
    }


# ----------------------------- sample-set metrics -----------------------------

def _pdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x.flatten(1).double(), y.flatten(1).double())


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, bandwidth: Optional[float] = None) -> float:
    """Unbiased MMD^2 with RBF kernel; bandwidth = median heuristic on pooled set."""
    xy = torch.cat([x, y], 0)
    d = _pdist(xy, xy)
    if bandwidth is None:
        bandwidth = float(d[d > 0].median()) if bool((d > 0).any()) else 1.0
    k = torch.exp(-(d ** 2) / (2 * bandwidth ** 2))
    n, m = x.shape[0], y.shape[0]
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    mmd = ((kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
           + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
           - 2 * kxy.mean())
    return float(mmd)


def precision_recall(gen: torch.Tensor, ref: torch.Tensor, k: int = 3) -> Tuple[float, float]:
    """Improved precision/recall (Kynkaanniemi et al. 2019) with k-NN balls.
    recall < 1 with precision ~ 1 is the signature of mode collapse."""
    def radii(s):
        d = _pdist(s, s)
        kk = min(k, s.shape[0] - 1)
        return d.kthvalue(kk + 1, dim=1).values  # +1 skips self
    r_ref, r_gen = radii(ref), radii(gen)
    d_gr = _pdist(gen, ref)
    precision = float((d_gr <= r_ref[None, :]).any(dim=1).double().mean())
    recall = float((d_gr.T <= r_gen[None, :]).any(dim=1).double().mean())
    return precision, recall


def diversity(x: torch.Tensor) -> float:
    d = _pdist(x, x)
    n = x.shape[0]
    return float(d.sum() / (n * (n - 1)))
