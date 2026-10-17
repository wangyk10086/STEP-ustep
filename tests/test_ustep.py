"""
U-STEP unit / integration tests (CPU, tiny models).  Run: python -m pytest tests/test_ustep.py -q
"""
import math
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.policy.adaptive_warmstart import (
    AdaptiveWarmStartConfig, select_start_timesteps, build_schedules,
    adaptive_warmstart_sample, snr_from_alphas_cumprod)

T_TRAIN = 100


def _ddim():
    return DDIMScheduler(num_train_timesteps=T_TRAIN, beta_schedule='squaredcos_cap_v2', clip_sample=True,
                         set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')


def test_start_time_rule_matches_closed_form():
    ac = torch.as_tensor(_ddim().alphas_cumprod)
    snr = snr_from_alphas_cumprod(ac)
    assert torch.all(snr[1:] < snr[:-1]), 'SNR must decrease with t'
    cfg = AdaptiveWarmStartConfig(delta=0.1)
    m = torch.logspace(-8, 8, 400, dtype=torch.float64)
    t = select_start_timesteps(m, ac, cfg)
    assert torch.all(t[1:] >= t[:-1]), 't* must be monotone in predicted variance'
    assert t[0] == 0 and t[-1] == T_TRAIN - 1
    target = 2 * cfg.delta / m
    for i in range(len(m)):
        ti = int(t[i])
        if 0 < ti < T_TRAIN - 1:
            # least-noisy timestep whose SNR is at or below the budget
            assert snr[ti] <= target[i] < snr[ti - 1]


def test_schedules():
    cfg = AdaptiveWarmStartConfig(n_min=1, n_max=10)
    t_star = torch.tensor([0, 5, 9, 10, 49, 99])
    sched, n = build_schedules(t_star, cfg, num_train_timesteps=T_TRAIN)
    assert n.tolist() == [1, 1, 1, 2, 5, 10]
    for b in range(len(t_star)):
        ts = sched[b, :n[b]]
        assert ts[0] == t_star[b]
        assert torch.all(ts[1:] < ts[:-1])
        assert torch.all(sched[b, n[b]:] == -1)
    cfg_f = AdaptiveWarmStartConfig(step_mode='fixed', n_fixed=2)
    _, n = build_schedules(torch.tensor([3, 50, 99]), cfg_f, num_train_timesteps=T_TRAIN)
    assert n.tolist() == [2, 2, 2]


def test_sampler_is_exact_ddim_and_oracle_denoiser_recovers_data():
    """With an oracle epsilon for a point-mass target A0, deterministic DDIM must
    return A0 from any start and any number of steps (tests the update math)."""
    torch.manual_seed(0)
    sched = _ddim()
    sched.config.clip_sample = False
    ac = torch.as_tensor(sched.alphas_cumprod, dtype=torch.float32)
    B, T, D = 6, 16, 2
    A0 = torch.randn(B, T, D) * 0.5

    def oracle(x, t):
        a = ac[t].view(-1, 1, 1)
        return (x - a.sqrt() * A0) / (1 - a).sqrt()

    mu = A0 + 0.3 * torch.randn_like(A0)        # biased warm start
    var = torch.logspace(-4, 1, B).view(B, 1, 1).expand(B, T, D).clone()
    cond = torch.zeros(B, T, D); mask = torch.zeros_like(cond, dtype=torch.bool)
    x, st = adaptive_warmstart_sample(oracle, mu, var, sched, cond, mask, AdaptiveWarmStartConfig(delta=0.1))
    assert torch.allclose(x, A0, atol=1e-4), (x - A0).abs().max()
    assert st['nfe'].tolist() == sorted(st['nfe'].tolist())   # more variance -> more NFE


def test_sampler_matches_diffusers_ddim_single_step():
    torch.manual_seed(1)
    sched = _ddim()
    ac = torch.as_tensor(sched.alphas_cumprod, dtype=torch.float32)
    B, T, D = 3, 8, 2
    W = torch.randn(D, D) * 0.1
    def net(x, t):
        return torch.tanh(x @ W) + 0.01 * t.view(-1, 1, 1).float()
    mu = torch.randn(B, T, D) * 0.3
    var = torch.full((B, T, D), 0.05)
    cfg = AdaptiveWarmStartConfig(delta=0.1, step_mode='fixed', n_fixed=1)
    g = torch.Generator().manual_seed(7)
    cond = torch.zeros(B, T, D); mask = torch.zeros_like(cond, dtype=torch.bool)
    x, st = adaptive_warmstart_sample(net, mu, var, sched, cond, mask, cfg, generator=g)
    # manual: same noise, one DDIM step from t* to clean
    g = torch.Generator().manual_seed(7)
    noise = torch.randn(mu.shape, generator=g)
    t = st['t_star']
    a = ac[t].view(-1, 1, 1)
    xt = a.sqrt() * mu + (1 - a).sqrt() * noise
    eps = net(xt, t)
    x0 = ((xt - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
    assert torch.allclose(x, x0, atol=1e-5)


def _tiny_lowdim_stack(predict_var=True):
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusion_policy.policy.enhanced_diffusion_unet_lowdim_policy import EnhancedDiffusionUnetLowdimPolicy
    from diffusion_policy.model.action_predictor.action_predictor_transformer import ActionPredictorTransformer
    from diffusion_policy.policy.action_predictor_lowdim_policy import ActionPredictorLowdimPolicy
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    H, Da, Do, To = 16, 2, 5, 2
    unet = ConditionalUnet1D(input_dim=Da, global_cond_dim=Do * To, diffusion_step_embed_dim=16,
                             down_dims=[16, 32], kernel_size=3, n_groups=4)
    dp = EnhancedDiffusionUnetLowdimPolicy(
        model=unet, noise_scheduler=DDPMScheduler(num_train_timesteps=T_TRAIN, beta_schedule='squaredcos_cap_v2',
                                                  clip_sample=True, prediction_type='epsilon'),
        horizon=H, obs_dim=Do, action_dim=Da, n_action_steps=8, n_obs_steps=To,
        num_inference_steps=10, init_trajectory_steps=2, obs_as_global_cond=True, oa_step_convention=True)
    model = ActionPredictorTransformer(action_dim=Da, obs_dim=Do, pred_horizon=H, n_obs_steps=To,
                                       prev_action_horizon=H, d_model=32, n_head=4, n_layers=1,
                                       dim_feedforward=64, predict_var=predict_var)
    ap = ActionPredictorLowdimPolicy(model=model, horizon=H, obs_dim=Do, action_dim=Da,
                                     n_action_steps=8, n_obs_steps=To, prev_action_horizon=H)
    norm = LinearNormalizer()
    norm.fit({'obs': torch.randn(500, Do), 'action': torch.randn(500, Da)}, last_n_dims=1, mode='limits')
    norm['prev_action'] = norm['action']
    return dp, ap, norm, (H, Da, Do, To)


def test_combined_policy_adaptive_and_original_paths():
    from diffusion_policy.policy.combined_inference_policy import CombinedInferencePolicy
    torch.manual_seed(0)
    dp, ap, norm, (H, Da, Do, To) = _tiny_lowdim_stack()
    pol = CombinedInferencePolicy(action_predictor=ap, diffusion_policy=dp)
    pol.set_normalizer(norm)
    obs = {'obs': torch.randn(4, To, Do)}
    # original STEP path
    out = pol.predict_action(obs)
    assert out['action'].shape == (4, 8, Da) and 'warmstart_stats' not in out
    assert pol.get_warmstart_summary() == {}
    # U-STEP path
    dp.set_adaptive_warmstart(AdaptiveWarmStartConfig(delta=0.1))
    for _ in range(3):
        out = pol.predict_action(obs)
    assert out['action'].shape == (4, 8, Da)
    summ = pol.get_inference_stats()
    assert summ['ustep_calls'] == 3 and 1 <= summ['ustep_avg_nfe'] <= 10


def test_predictor_var_head_training_and_freeze_mean():
    from diffusion_policy.model.action_predictor.uncertainty import build_predictor_optimizer
    from omegaconf import OmegaConf
    torch.manual_seed(0)
    _, ap, norm, (H, Da, Do, To) = _tiny_lowdim_stack()
    ap.set_normalizer(norm)
    ap.freeze_mean = True
    opt = build_predictor_optimizer(ap, OmegaConf.create(
        dict(lr=1e-3, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.0)))
    mean_before = [p.detach().clone() for p in ap.model.mean_parameters()]
    batch = {'obs': torch.randn(64, H, Do), 'action': torch.randn(64, H, Da),
             'prev_action': torch.randn(64, H, Da), 'prev_valid': torch.ones(64)}
    for _ in range(400):
        loss = ap.compute_loss(batch); opt.zero_grad(); loss.backward(); opt.step()
    for a, b in zip(mean_before, ap.model.mean_parameters()):
        assert torch.equal(a, b), 'mean predictor must stay frozen'
    assert 'calib_ratio' in ap.last_train_logs
    # NLL with detached mean drives var toward E[resid^2]: calib ratio -> ~1
    assert 0.5 < ap.last_train_logs['calib_ratio'] < 2.0, ap.last_train_logs


def test_aligned_prev_action_wrapper():
    from diffusion_policy.common.replay_buffer import ReplayBuffer
    from diffusion_policy.common.sampler import SequenceSampler
    from diffusion_policy.dataset.aligned_prev_action_wrapper import AlignedPrevActionWrapper

    class Toy:
        def __init__(self):
            rb = ReplayBuffer.create_empty_numpy()
            for L in [40, 30]:
                t = np.arange(L, dtype=np.float32)
                rb.add_episode({'action': np.stack([t, t], -1), 'obs': np.stack([t] * 3, -1)})
            self.replay_buffer = rb
            self.sampler = SequenceSampler(rb, sequence_length=16, pad_before=1, pad_after=7)
        def __len__(self): return len(self.sampler)
        def __getitem__(self, i):
            d = self.sampler.sample_sequence(i)
            return {k: torch.from_numpy(v) for k, v in d.items()}
        def get_validation_dataset(self): return self

    w = AlignedPrevActionWrapper(Toy(), prev_action_horizon=16, n_action_steps=8, n_obs_steps=2)
    for i in range(len(w)):
        s = w[i]
        a, prev, valid = s['action'][:, 0], s['prev_action'][:, 0], bool(s['prev_valid'])
        if valid:
            # previous call's chunk is the same trajectory shifted by 8 steps:
            # its last 8 steps coincide with the first 8 of the current window
            assert torch.equal(prev[8:], a[:8]) or a[0] == a[1], i
            assert prev[0] <= a[0]
    # the STEP wrapper leaks the target: prev == action[:8]
    from diffusion_policy.policy.action_predictor_lowdim_policy import ActionPredictorDatasetWrapper
    old = ActionPredictorDatasetWrapper(Toy(), prev_action_horizon=8, n_action_steps=8)
    s = old[10]
    assert torch.equal(s['prev_action'], s['action'][:8])
