"""
U-STEP predictor training utilities.

* Heteroscedastic variance head trained with a Gaussian NLL on the *detached*
  residual.  With the mean detached, the mean predictor is trained exactly as
  in STEP (pure MSE), and the NLL is a proper scoring rule for the variance:
  its minimizer is sigma^2(c) = E[(A - mu(c))^2 | c] elementwise, so
  sum(sigma^2) is an unbiased plug-in for m(c) = E||A - mu(c)||^2 used by the
  adaptive start-time rule (docs/UWARM_THEORY.md, Sec. 3).

* History augmentation against copycat behaviour: Gaussian noise and random
  dropout (-> zeros, exactly what the predictor sees at episode start) on the
  normalized previous chunk.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def augment_prev_action(
    nprev: torch.Tensor,
    noise_std: float = 0.0,
    dropout: float = 0.0,
    prev_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """nprev: [B, H, Da] normalized previous chunk."""
    if noise_std > 0:
        nprev = nprev + noise_std * torch.randn_like(nprev)
    keep = torch.ones(nprev.shape[0], device=nprev.device, dtype=nprev.dtype)
    if dropout > 0:
        keep = keep * (torch.rand_like(keep) >= dropout).to(nprev.dtype)
    if prev_valid is not None:
        keep = keep * prev_valid.to(device=nprev.device, dtype=nprev.dtype).view(-1)
    return nprev * keep.view(-1, 1, 1)


def predictor_loss(
    model,
    nprev: torch.Tensor,
    obs_in: torch.Tensor,
    target: torch.Tensor,
    predict_var: bool = False,
    var_loss_weight: float = 1.0,
    freeze_mean: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logs = {}
    if predict_var:
        pred, logvar = model(nprev, obs_in, return_logvar=True)
    else:
        pred = model(nprev, obs_in)
        logvar = None

    mse = F.mse_loss(pred, target)
    logs['mse'] = float(mse.detach())
    loss = torch.zeros((), device=pred.device) if freeze_mean else mse

    if logvar is not None:
        resid2 = (target - pred.detach()) ** 2
        nll = 0.5 * (resid2 * torch.exp(-logvar) + logvar).mean()
        loss = loss + var_loss_weight * nll
        with torch.no_grad():
            var = logvar.exp()
            logs['nll'] = float(nll)
            # calibration ratio: E[resid^2] / E[var]; 1.0 = calibrated
            logs['calib_ratio'] = float(resid2.mean() / var.mean().clamp_min(1e-12))
            # per-sample correlation between ||e||^2 and tr(Sigma)
            e2 = resid2.flatten(1).sum(1)
            tr = var.flatten(1).sum(1)
            if e2.numel() > 2:
                c = torch.corrcoef(torch.stack([e2, tr]))[0, 1]
                logs['err_var_corr'] = float(c) if torch.isfinite(c) else 0.0
    return loss, logs


@torch.no_grad()
def predict_with_var(model, nprev, obs_in):
    """Returns (mean, var) in normalized action space; var=None if no head."""
    if getattr(model, 'predict_var', False):
        mean, logvar = model(nprev, obs_in, return_logvar=True)
        return mean, logvar.exp()
    return model(nprev, obs_in), None


def load_step_predictor_weights(policy, ckpt_path: str, device='cpu'):
    """Load a trained STEP predictor checkpoint into a (possibly var-head
    augmented) policy. Only the new logvar_head params may be missing."""
    import dill
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location=device)
    sd = payload['state_dicts']['model']
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    bad = [k for k in missing if 'logvar_head' not in k]
    if bad or unexpected:
        raise RuntimeError(f'incompatible checkpoint: missing={bad} unexpected={list(unexpected)}')
    print(f'[U-STEP] loaded STEP predictor from {ckpt_path}; new params: {missing}')
    return policy


def build_predictor_optimizer(policy, optimizer_cfg):
    """AdamW over all params, or only the variance head when freeze_mean=True."""
    model = policy.model
    if getattr(policy, 'freeze_mean', False):
        for p in model.mean_parameters():
            p.requires_grad_(False)
        enc = getattr(policy, 'encoder', None)
        if enc is not None:
            for p in enc.parameters():
                p.requires_grad_(False)
        proj = getattr(policy, 'obs_proj', None)
        if proj is not None:
            for p in proj.parameters():
                p.requires_grad_(False)
        params = model.var_parameters()
    else:
        params = [p for p in policy.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params, lr=optimizer_cfg.lr, betas=tuple(optimizer_cfg.betas),
        eps=optimizer_cfg.eps, weight_decay=optimizer_cfg.weight_decay)
