"""
Uncertainty-calibrated adaptive warm start (U-STEP).

Given a warm-start chunk mu(c) from the predictor and an estimate of its
conditional mean-squared error m(c) = E||A - mu(c)||^2 (in normalized action
space), choose a *per-sample* start time t*(c) for the reverse process:

    SNR(t*) = 2 * delta / m_hat(c),      m_hat(c) = kappa * tr Sigma_hat(c)

where SNR(t) = abar_t / (1 - abar_t). This is the closed-form rule from
docs/UWARM_THEORY.md (Corollary 1): it is the least-noisy start whose
warm-start mismatch  KL(w_t || q_t) <= 1/2 * SNR(t) * m(c)  stays below the
total budget delta (nats).  The warm start itself uses the exact forward marginal

    x_{t*} = sqrt(abar_{t*}) * mu + sqrt(1 - abar_{t*}) * eps

and is then integrated to 0 with a deterministic DDIM update whose number of
steps grows with t* (n = ceil(t* / stride), clipped to [n_min, n_max]).

Easy states (small m) -> t* near 0 -> 1 NFE.
Ambiguous / multimodal states (large m) -> t* near T -> the original diffusion
policy, so its multimodality is recovered instead of collapsed onto mu.
"""

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple
import math
import torch


@dataclass
class AdaptiveWarmStartConfig:
    # total KL budget (nats) for the warm-start mismatch: SNR(t*) = 2 delta / m_hat.
    # Smaller delta -> noisier start, more NFE, closer to the full diffusion policy.
    delta: float = 0.1
    # multiplicative calibration of the predicted variance (fit on val set by
    # calibrate_uwarm.py; 1.0 = trust the NLL head as is)
    kappa: float = 1.0
    # DDIM step size in training-timestep units; n_steps = ceil((t*+1)/stride).
    # None -> num_train_timesteps // n_max, i.e. a start at t=T uses n_max steps
    stride: Optional[int] = None
    n_min: int = 1
    n_max: int = 10
    # clamp the start timestep (in training timesteps, inclusive)
    t_min: int = 0
    t_max: Optional[int] = None
    # 'adaptive': n grows with t*;  'fixed': always n_fixed steps from t*
    step_mode: str = 'adaptive'
    n_fixed: int = 2
    # use the variance only over part of the chunk (e.g. the executed steps)
    exec_slice: Optional[Tuple[int, int]] = None

    def to_dict(self):
        return asdict(self)


def _alphas_cumprod(scheduler) -> torch.Tensor:
    ac = scheduler.alphas_cumprod
    if not torch.is_tensor(ac):
        ac = torch.as_tensor(ac)
    return ac.to(torch.float64)


def snr_from_alphas_cumprod(alphas_cumprod: torch.Tensor) -> torch.Tensor:
    ac = alphas_cumprod.to(torch.float64)
    return ac / (1.0 - ac).clamp_min(1e-12)


def reduce_variance(var: torch.Tensor, cfg: AdaptiveWarmStartConfig) -> torch.Tensor:
    """var: [B, T, Da] per-element predicted variance ->
    m_hat: [B] predicted squared error  kappa * tr(Sigma_hat)  (estimate of E||A-mu||^2)."""
    if cfg.exec_slice is not None:
        a, b = cfg.exec_slice
        var = var[:, a:b]
    return var.flatten(1).sum(dim=1) * cfg.kappa


def select_start_timesteps(
    m_hat: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    cfg: AdaptiveWarmStartConfig,
) -> torch.Tensor:
    """
    Closed-form start time:  t* = min{ t : SNR(t) <= 2 delta / m_hat }.
    SNR is monotonically decreasing in t, so we take the first timestep whose
    SNR is below the target.  Returns LongTensor [B] of training timesteps.
    """
    snr = snr_from_alphas_cumprod(alphas_cumprod).to(m_hat.device)   # [K]
    K = snr.shape[0]
    target = 2.0 * cfg.delta / m_hat.to(torch.float64).clamp_min(1e-12)  # [B]
    # number of timesteps whose SNR is still above target == first index below
    t_star = (snr[None, :] > target[:, None]).sum(dim=1)            # [B]
    t_max = K - 1 if cfg.t_max is None else min(cfg.t_max, K - 1)
    t_star = t_star.clamp(min=cfg.t_min, max=t_max)
    return t_star.long()


def build_schedules(
    t_star: torch.Tensor, cfg: AdaptiveWarmStartConfig, num_train_timesteps: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-sample DDIM timestep schedules from t*_b down to 0.
    Returns
        sched: LongTensor [B, n_max_used + 1]; sched[b, :n_b+1] = t*_b ... -1
               (-1 means "clean", abar=1), padded with -1.
        n_steps: LongTensor [B] number of network evaluations per sample.
    """
    B = t_star.shape[0]
    if cfg.step_mode == 'adaptive':
        stride = cfg.stride
        if stride is None:
            if num_train_timesteps is None:
                raise ValueError('stride=None requires num_train_timesteps')
            stride = max(1, num_train_timesteps // cfg.n_max)
        n = torch.ceil((t_star.double() + 1) / stride).long()
        n = n.clamp(min=cfg.n_min, max=cfg.n_max)
    elif cfg.step_mode == 'fixed':
        n = torch.full_like(t_star, cfg.n_fixed)
    else:
        raise ValueError(cfg.step_mode)
    n_max_used = int(n.max().item())
    sched = torch.full((B, n_max_used + 1), -1, dtype=torch.long, device=t_star.device)
    for b in range(B):
        nb = int(n[b].item())
        ts = torch.linspace(float(t_star[b].item()), 0.0, nb + 1)[:-1]
        ts = torch.round(ts).long()
        # strictly decreasing, unique timesteps
        ts = torch.unique_consecutive(ts)
        nb = ts.shape[0]
        n[b] = nb
        sched[b, :nb] = ts.to(t_star.device)
    return sched, n


def _gather_abar(alphas_cumprod: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    ac = alphas_cumprod.to(t.device)
    out = torch.ones_like(t, dtype=ac.dtype)
    valid = t >= 0
    out[valid] = ac[t[valid]]
    return out


@torch.no_grad()
def adaptive_warmstart_sample(
    denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    init_trajectory: torch.Tensor,
    init_var: torch.Tensor,
    scheduler,
    condition_data: torch.Tensor,
    condition_mask: torch.Tensor,
    cfg: AdaptiveWarmStartConfig,
    action_dim: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    denoise_fn(x [B,T,D], t LongTensor [B]) -> model output (epsilon or sample).
    init_trajectory: [B, T, D] normalized warm start (actions in [..., :Da]).
    init_var: [B, T, Da] per-element predicted variance (normalized space).
    """
    device = init_trajectory.device
    dtype = init_trajectory.dtype
    B = init_trajectory.shape[0]
    ac = _alphas_cumprod(scheduler).to(device)
    pred_type = scheduler.config.prediction_type
    clip_sample = bool(getattr(scheduler.config, 'clip_sample', False))
    clip_range = float(getattr(scheduler.config, 'clip_sample_range', 1.0))

    m_hat = reduce_variance(init_var.to(torch.float64), cfg)
    t_star = select_start_timesteps(m_hat, ac, cfg)
    sched, n_steps = build_schedules(t_star, cfg, num_train_timesteps=ac.shape[0])

    # exact forward marginal at t*
    noise = torch.randn(init_trajectory.shape, device=device, dtype=dtype, generator=generator)
    abar0 = _gather_abar(ac, sched[:, 0]).to(dtype).view(B, 1, 1)
    x = abar0.sqrt() * init_trajectory + (1 - abar0).sqrt() * noise

    n_iter = sched.shape[1] - 1
    for i in range(n_iter):
        t = sched[:, i]
        t_prev = sched[:, i + 1]
        active = t >= 0
        if not bool(active.any()):
            break
        x[condition_mask] = condition_data[condition_mask]
        t_in = t.clamp_min(0)
        out = denoise_fn(x, t_in)
        abar_t = _gather_abar(ac, t_in).to(dtype).view(B, 1, 1)
        abar_p = _gather_abar(ac, t_prev).to(dtype).view(B, 1, 1)
        if pred_type == 'epsilon':
            eps = out
            x0 = (x - (1 - abar_t).sqrt() * eps) / abar_t.sqrt()
        elif pred_type == 'sample':
            x0 = out
            eps = (x - abar_t.sqrt() * x0) / (1 - abar_t).sqrt().clamp_min(1e-8)
        else:
            raise ValueError(f'unsupported prediction_type {pred_type}')
        if clip_sample:
            x0 = x0.clamp(-clip_range, clip_range)
        x_new = abar_p.sqrt() * x0 + (1 - abar_p).sqrt() * eps
        a = active.view(B, 1, 1)
        x = torch.where(a, x_new, x)

    x[condition_mask] = condition_data[condition_mask]
    stats = {
        't_star': t_star,
        'nfe': n_steps,
        'm_hat': m_hat.to(dtype),
        'snr_start': snr_from_alphas_cumprod(ac)[t_star].to(dtype),
    }
    return x, stats
