"""
Offline estimate of the two error terms of the warm-start bound, and of the start level t0*
they imply. This needs no rollouts.

For held-out contexts (o, a) from the validation split:
  ref(t0)       = DDIM with one step per training timestep (t0 steps) from the EXACT start
                  x_t0 = sqrt(abar) a + sigma eps, i.e. ~the exact probability-flow solution
  solver(t0,n)  = || Phi_n(x_t0) - ref(t0) ||          (same eps; Prop. 2's left-hand side)
  warm(t0,n)    = || Phi_n(xhat_t0) - ref(t0) ||       (xhat_t0 uses the predictor mean a_hat,
                                                         same eps; the end-to-end warm-start error)
  m             = E || a_hat - a ||^2 / dim            (predictor error, normalized action space)
All distances are RMS per action element in the normalized space. Fits C in
solver ~ C * Lambda^2 / n and reports t0*(n) = argmin_t0 warm(t0, n).

  python experiments/v2/solver_error.py --action_predictor_ckpt P --diffusion_policy_ckpt DP \
      --out /workspace/eval/solver_<task>.json
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
import argparse, json, math, os
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.adaptive_warmstart import AdaptiveWarmStartConfig
from diffusion_policy.policy.ustep_offline import build_eval_dataset, make_loader, predictor_forward


def ddim(T):
    return DDIMScheduler(num_train_timesteps=T, beta_schedule='squaredcos_cap_v2', clip_sample=True,
                         set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action_predictor_ckpt', required=True)
    p.add_argument('--diffusion_policy_ckpt', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--n_contexts', type=int, default=512)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--t0', type=int, nargs='+', default=[30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99])
    p.add_argument('--n', type=int, nargs='+', default=[1, 2, 3, 4])
    p.add_argument('--history', default='data', choices=['data', 'none'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    from eval_combined_inference import load_action_predictor, load_diffusion_policy
    pred, pred_cfg = load_action_predictor(a.action_predictor_ckpt, a.device)
    dp, _ = load_diffusion_policy(a.diffusion_policy_ckpt, a.device, use_ema=True)
    T = dp.noise_scheduler.config.num_train_timesteps
    dp.noise_scheduler = ddim(T)
    if hasattr(dp, 'policy'):
        dp.policy.noise_scheduler = dp.noise_scheduler
    norm_a = dp.normalizer['action']
    ac = dp.noise_scheduler.alphas_cumprod.double()
    lam = 0.5 * torch.log(ac / (1 - ac))

    ds = build_eval_dataset(pred_cfg, split='val')
    dl = make_loader(ds, batch_size=a.batch, num_workers=4, seed=a.seed)

    def run(obs, init, var, t0, n, seed):
        dp.set_adaptive_warmstart(AdaptiveWarmStartConfig(delta=1.0, kappa=1.0, t_min=t0, t_max=t0,
                                                          step_mode='fixed', n_fixed=n, n_max=max(n, 1)))
        torch.manual_seed(seed)
        out = dp.predict_action({'obs': obs}, init_action=init, init_var=var)
        dp.set_adaptive_warmstart(None)
        return norm_a.normalize(out['action_pred'])

    acc = {}
    m_sum, m_cnt, done = 0.0, 0, 0
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if done >= a.n_contexts:
                break
            if a.history == 'none':
                batch['prev_valid'] = torch.zeros(batch['action'].shape[0])
            pf = predictor_forward(pred, batch, a.device)
            obs = batch['obs'].to(a.device)
            Hp = pf['mean_raw'].shape[1]
            gt_raw = batch['action'][:, :Hp].to(a.device)
            var = pf['var'] if pf['var'] is not None else torch.ones_like(pf['mean'])
            m_sum += float(((pf['mean'] - pf['target'][:, :Hp]) ** 2).mean(-1).mean(-1).sum()); m_cnt += gt_raw.shape[0]
            for t0 in a.t0:
                s = a.seed * 100003 + bi * 1009 + t0
                ref = run(obs, gt_raw, var, t0, t0, s)
                for n in a.n:
                    if n > t0:
                        continue
                    ex = run(obs, gt_raw, var, t0, n, s)
                    wa = run(obs, pf['mean_raw'], var, t0, n, s)
                    H = min(ref.shape[1], ex.shape[1])
                    rms = lambda x: ((x[:, :H] - ref[:, :H]) ** 2).mean((-1, -2)).sqrt()
                    d = acc.setdefault(f'{t0}:{n}', {'solver': [], 'warm': []})
                    d['solver'] += rms(ex).tolist(); d['warm'] += rms(wa).tolist()
            done += gt_raw.shape[0]
            print(f'{done}/{a.n_contexts}', flush=True)

    m = m_sum / max(m_cnt, 1)
    rows = {}
    for k, d in acc.items():
        t0, n = (int(x) for x in k.split(':'))
        Lam = float(lam[0] - lam[t0])
        rows[k] = dict(t0=t0, n=n, Lambda=Lam, snr=float(ac[t0] / (1 - ac[t0])),
                       solver=float(np.mean(d['solver'])), warm=float(np.mean(d['warm'])),
                       warm_se=float(np.std(d['warm']) / math.sqrt(len(d['warm']))))
    C_hat = float(np.median([r['solver'] * r['n'] / r['Lambda'] ** 2 for r in rows.values() if r['Lambda'] > 0]))
    tstar = {}
    for n in a.n:
        cand = [r for r in rows.values() if r['n'] == n]
        if cand:
            b = min(cand, key=lambda r: r['warm'])
            bound = {r['t0']: 0.5 * math.sqrt(r['snr'] * m) + C_hat * r['Lambda'] ** 2 / n for r in cand}
            tstar[n] = dict(t0_offline=b['t0'], t0_bound=min(bound, key=bound.get))
    res = dict(args=vars(a), m=m, C_hat=C_hat, t0_star=tstar, rows=rows)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2)
    print(json.dumps(dict(m=m, C_hat=C_hat, t0_star=tstar), indent=2))
    for n in a.n:
        line = [f"{r['t0']}:{r['warm']:.3f}/{r['solver']:.3f}" for r in sorted(rows.values(), key=lambda r: r['t0']) if r['n'] == n]
        print(f'n={n} (t0: warm/solver)', ' '.join(line))


if __name__ == '__main__':
    main()
