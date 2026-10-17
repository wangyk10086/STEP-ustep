"""
Toy sanity check for U-STEP (CPU, ~minutes).

State c ~ U[0,1]; action chunk A in R^{H x 2}.
  c <  0.5 : unimodal  A = f(c) * ramp + 0.02 noise           (easy state)
  c >= 0.5 : bimodal   A = +/- g(c) * ramp + 0.02 noise, 50/50  (ambiguous state,
             e.g. passing an obstacle left or right)

A small diffusion policy (repo's EnhancedDiffusionUnetLowdimPolicy, exact
sampling code) and a mean+variance predictor are trained; then we compare
  DDPM-100 (reference), STEP with K' in {1,2,4} (repo's fixed warm start), U-STEP.

Also checks Lemma 1 numerically: KL(w_t || q_t) <= 1/2 SNR(t) E||A - mu||^2.

  python experiments/toy_bimodal.py --out data/ustep/toy
"""
import argparse, json, math, os, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.policy.enhanced_diffusion_unet_lowdim_policy import EnhancedDiffusionUnetLowdimPolicy
from diffusion_policy.policy.adaptive_warmstart import AdaptiveWarmStartConfig, snr_from_alphas_cumprod
from diffusion_policy.model.common.normalizer import LinearNormalizer

H, DA, T_TRAIN = 8, 2, 100
RAMP = torch.linspace(0.1, 1.0, H)


def sample_data(n, c=None, gen=None):
    if c is None:
        c = torch.rand(n, generator=gen)
    uni = c < 0.5
    sign = torch.where(torch.rand(n, generator=gen) < 0.5, -1.0, 1.0)
    amp_u = 0.6 * torch.sin(2 * math.pi * c)                  # f(c)
    amp_b = 0.5 + 0.4 * (c - 0.5)                              # g(c)
    amp = torch.where(uni, amp_u, sign * amp_b)
    base = torch.stack([RAMP, 0.5 * RAMP], -1)                 # [H,2]
    A = amp.view(-1, 1, 1) * base.view(1, H, DA)
    A = A + 0.02 * torch.randn(A.shape, generator=gen)
    return c, A, uni


class EpsNet(nn.Module):
    """Denoiser with the ConditionalUnet1D call signature (x, t, local_cond, global_cond)."""
    def __init__(self, d=H * DA, hid=256):
        super().__init__()
        self.t_emb = nn.Embedding(T_TRAIN, 64)
        self.net = nn.Sequential(nn.Linear(d + 64 + 1, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU(),
                                 nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, d))
    def forward(self, x, t, local_cond=None, global_cond=None):
        B = x.shape[0]
        if not torch.is_tensor(t):
            t = torch.tensor([t])
        t = t.to(x.device).long().reshape(-1).expand(B)
        h = torch.cat([x.reshape(B, -1), self.t_emb(t), global_cond.reshape(B, -1)], -1)
        return self.net(h).view_as(x)


class Predictor(nn.Module):
    def __init__(self, d=H * DA, hid=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(1, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU())
        self.mu = nn.Linear(hid, d); self.lv = nn.Linear(hid, d)
    def forward(self, c):
        h = self.body(c.view(-1, 1))
        return self.mu(h).view(-1, H, DA), self.lv(h).clamp(-12, 4).view(-1, H, DA)


def train(args, gen):
    norm = LinearNormalizer()
    c, A, _ = sample_data(20000, gen=gen)
    norm.fit({'obs': c.view(-1, 1, 1), 'action': A}, last_n_dims=1, mode='limits')
    sched = DDPMScheduler(num_train_timesteps=T_TRAIN, beta_schedule='squaredcos_cap_v2',
                          clip_sample=True, prediction_type='epsilon')
    dp = EnhancedDiffusionUnetLowdimPolicy(model=EpsNet(), noise_scheduler=sched, horizon=H, obs_dim=1,
                                           action_dim=DA, n_action_steps=H, n_obs_steps=1,
                                           num_inference_steps=T_TRAIN, init_trajectory_steps=2,
                                           obs_as_global_cond=True, pred_action_steps_only=False,
                                           oa_step_convention=True)
    dp.set_normalizer(norm)
    pred = Predictor()
    opt = torch.optim.Adam(list(dp.model.parameters()) + list(pred.parameters()), lr=1e-3)
    for it in range(args.iters):
        c, A, _ = sample_data(512, gen=gen)
        batch = {'obs': c.view(-1, 1, 1).expand(-1, H, 1), 'action': A}
        loss_dp = dp.compute_loss(batch)
        nA = norm['action'].normalize(A)
        mu, lv = pred(c)
        mse = F.mse_loss(mu, nA)
        nll = 0.5 * (((nA - mu.detach()) ** 2) * torch.exp(-lv) + lv).mean()
        loss = loss_dp + mse + nll
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0:
            print(f'it {it}: dp {loss_dp.item():.4f}  pred mse {mse.item():.4f}  nll {nll.item():.3f}')
    return dp, pred, norm


@torch.no_grad()
def run_sampler(dp, pred, norm, c, mode, k=None, delta=None):
    dp.set_adaptive_warmstart(None)
    obs = {'obs': c.view(-1, 1, 1)}
    if mode == 'ddpm':
        dp.noise_scheduler = DDPMScheduler(num_train_timesteps=T_TRAIN, beta_schedule='squaredcos_cap_v2',
                                           clip_sample=True, prediction_type='epsilon')
        dp.num_inference_steps = T_TRAIN
        return dp.predict_action(obs)['action_pred'], float(T_TRAIN)
    dp.noise_scheduler = DDIMScheduler(num_train_timesteps=T_TRAIN, beta_schedule='squaredcos_cap_v2',
                                       clip_sample=True, set_alpha_to_one=True, steps_offset=0,
                                       prediction_type='epsilon')
    mu, lv = pred(c)
    init = norm['action'].unnormalize(mu)
    if mode == 'step':
        dp.init_trajectory_steps = k
        return dp.predict_action(obs, init_action=init)['action_pred'], float(k)
    if mode == 'ustep':
        dp.set_adaptive_warmstart(AdaptiveWarmStartConfig(delta=delta))
        out = dp.predict_action(obs, init_action=init, init_var=lv.exp())
        return out['action_pred'], float(out['warmstart_stats']['nfe'].float().mean())
    if mode == 'pred':
        return init, 0.0
    raise ValueError(mode)


def mode_stats(A, c):
    """For bimodal states: fraction of samples assigned to + / - / neither mode."""
    base = torch.stack([RAMP, 0.5 * RAMP], -1)
    amp = (A * base).sum((1, 2)) / (base ** 2).sum()             # projection on mode direction
    resid = (A - amp.view(-1, 1, 1) * base).flatten(1).norm(dim=1)
    g = 0.5 + 0.4 * (c - 0.5)
    on_plus = ((amp - g).abs() < 0.15) & (resid < 0.2)
    on_minus = ((amp + g).abs() < 0.15) & (resid < 0.2)
    return float(on_plus.float().mean()), float(on_minus.float().mean())


def check_lemma(dp, norm, gen, n_mc=200000):
    """Monte-Carlo KL(w_t || q_t) for a bimodal context vs the bound 1/2 SNR m."""
    ac = torch.as_tensor(DDIMScheduler(num_train_timesteps=T_TRAIN, beta_schedule='squaredcos_cap_v2').alphas_cumprod,
                         dtype=torch.float64)
    snr = snr_from_alphas_cumprod(ac)
    # 1-D projection of the bimodal target: A in {-a, +a} + N(0, s^2); mu = 0 (MSE optimum)
    a, s = 1.0, 0.05
    out = []
    for t in [5, 10, 20, 30, 40, 50, 60, 70]:
        ab = float(ac[t])
        w_std = math.sqrt(1 - ab)
        x = torch.randn(n_mc, generator=gen, dtype=torch.float64) * w_std          # x ~ w_t
        var_q = ab * s * s + 1 - ab
        log_w = -0.5 * x ** 2 / w_std ** 2 - math.log(w_std)
        comp = torch.stack([-0.5 * (x - math.sqrt(ab) * a) ** 2 / var_q,
                            -0.5 * (x + math.sqrt(ab) * a) ** 2 / var_q], 0)
        log_q = torch.logsumexp(comp, 0) - math.log(2) - 0.5 * math.log(var_q)
        kl = float((log_w - log_q).mean())
        bound = 0.5 * float(snr[t]) * (a * a + s * s)
        out.append({'t': t, 'snr': float(snr[t]), 'kl_mc': kl, 'bound': bound, 'holds': kl <= bound + 1e-3})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=6000)
    p.add_argument('--n_eval', type=int, default=400)
    p.add_argument('--out', default='data/ustep/toy')
    args = p.parse_args()
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    dp, pred, norm = train(args, gen)
    dp.eval(); pred.eval()

    lemma = check_lemma(dp, norm, gen)
    print('\nLemma 1 check (bimodal target, mu = mean):')
    for r in lemma:
        print(f"  t={r['t']:3d} SNR={r['snr']:8.3f}  KL_mc={r['kl_mc']:.4f}  bound={r['bound']:.4f}  holds={r['holds']}")

    # evaluation contexts: 8 unimodal + 8 bimodal states, n_eval samples each
    cs = torch.cat([torch.linspace(0.05, 0.45, 8), torch.linspace(0.55, 0.95, 8)])
    methods = [('DDPM-100 (ref)', 'ddpm', {}), ('predictor only', 'pred', {}),
               ('STEP K=1', 'step', {'k': 1}), ('STEP K=2', 'step', {'k': 2}), ('STEP K=4', 'step', {'k': 4}),
               ('U-STEP d=0.01', 'ustep', {'delta': 0.01}), ('U-STEP d=0.03', 'ustep', {'delta': 0.03}),
               ('U-STEP d=0.1', 'ustep', {'delta': 0.1}), ('U-STEP d=0.3', 'ustep', {'delta': 0.3}),
               ('U-STEP d=1', 'ustep', {'delta': 1.0})]
    table = {}
    samples_for_plot = {}
    for name, mode, kw in methods:
        uni_err, uni_nfe, bi_plus, bi_minus, bi_on, bi_nfe = [], [], [], [], [], []
        for c0 in cs:
            c = c0.repeat(args.n_eval)
            A, nfe = run_sampler(dp, pred, norm, c, mode, **kw)
            if c0 < 0.5:
                _, Atrue, _ = sample_data(1, c=c0.view(1), gen=gen)
                clean = 0.6 * math.sin(2 * math.pi * float(c0)) * torch.stack([RAMP, 0.5 * RAMP], -1)
                uni_err.append(float((A - clean).flatten(1).norm(dim=1).mean())); uni_nfe.append(nfe)
            else:
                pp, mm = mode_stats(A, c)
                bi_plus.append(pp); bi_minus.append(mm); bi_on.append(pp + mm); bi_nfe.append(nfe)
            samples_for_plot.setdefault(name, []).append((float(c0), A[:200].flatten(1)[:, -2:].numpy()))
        # balance = how evenly the two modes are covered (1 = 50/50, 0 = one mode only)
        bal = float(np.mean([1 - abs(a - b) / max(a + b, 1e-8) for a, b in zip(bi_plus, bi_minus)]))
        table[name] = {'unimodal_err': float(np.mean(uni_err)), 'unimodal_nfe': float(np.mean(uni_nfe)),
                       'bimodal_on_mode': float(np.mean(bi_on)), 'bimodal_balance': bal,
                       'bimodal_nfe': float(np.mean(bi_nfe))}
    print(f"\n{'method':16s} {'uni err':>8s} {'uni NFE':>8s} {'bi on-mode':>11s} {'bi balance':>11s} {'bi NFE':>7s}")
    for k, r in table.items():
        print(f"{k:16s} {r['unimodal_err']:8.4f} {r['unimodal_nfe']:8.2f} {r['bimodal_on_mode']:11.3f} "
              f"{r['bimodal_balance']:11.3f} {r['bimodal_nfe']:7.2f}")
    with open(os.path.join(args.out, 'toy_results.json'), 'w') as f:
        json.dump({'table': table, 'lemma_check': lemma}, f, indent=2)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        show = ['DDPM-100 (ref)', 'STEP K=2', 'STEP K=4', 'U-STEP d=0.03']
        fig, axes = plt.subplots(1, len(show), figsize=(11, 3.2), sharey=True)
        for ax, name in zip(axes, show):
            for c0, pts in samples_for_plot[name]:
                ax.scatter(np.full(len(pts), c0) + np.random.uniform(-0.01, 0.01, len(pts)), pts[:, 0],
                           s=2, alpha=0.25, color='#2a6fdb')
            ax.axvline(0.5, color='#999', lw=0.8, ls='--')
            r = table[name]
            ax.set_title(f"{name}\nNFE uni {r['unimodal_nfe']:.1f} / bi {r['bimodal_nfe']:.1f}", fontsize=9)
            ax.set_xlabel('state c')
        axes[0].set_ylabel('final action (dim 0)')
        fig.tight_layout(); fig.savefig(os.path.join(args.out, 'toy_bimodal.png'), dpi=200)
        print(f"figure -> {os.path.join(args.out, 'toy_bimodal.png')}")
    except Exception as e:
        print('(skip plot)', e)


if __name__ == '__main__':
    main()
