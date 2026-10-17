"""
U-STEP calibration of the predictor's variance head.

Fits the scalar kappa such that  E[ ||A - mu||^2 ] = kappa * E[ tr Sigma_hat ]
on held-out data, and reports how well the predicted uncertainty ranks the
actual warm-start error (the property the adaptive start time relies on):

  * pearson / spearman correlation between ||e||^2 and tr(Sigma_hat)
  * reliability table: deciles of predicted tr(Sigma), mean actual vs predicted
  * implied distribution of start timesteps t* and NFE for a list of delta

Usage:
  python calibrate_uwarm.py --action_predictor_ckpt PRED.ckpt \
      --modality lowdim --deltas 0.02 0.05 0.1 0.3 1 --out data/ustep/calib_pusht.json
"""

if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).parent))

import argparse
import json
import os

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.policy.adaptive_warmstart import (
    AdaptiveWarmStartConfig, select_start_timesteps, build_schedules)
from diffusion_policy.policy.ustep_offline import build_eval_dataset, make_loader, predictor_forward


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action_predictor_ckpt', required=True)
    p.add_argument('--modality', default='lowdim', choices=['lowdim', 'image'])
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--split', default='val', choices=['val', 'train'])
    p.add_argument('--aligned', type=int, default=-1, help='-1: as trained, 0/1 force wrapper')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--max_batches', type=int, default=50)
    p.add_argument('--num_train_timesteps', type=int, default=100)
    p.add_argument('--deltas', type=float, nargs='+', default=[0.02, 0.05, 0.1, 0.3, 1.0])
    p.add_argument('--stride', type=int, default=None, help='DDIM stride (default T // n_max)')
    p.add_argument('--n_max', type=int, default=10)
    p.add_argument('--out', default='data/ustep/calibration.json')
    args = p.parse_args()

    from eval_combined_inference import load_action_predictor, load_action_predictor_image
    loader_fn = load_action_predictor_image if args.modality == 'image' else load_action_predictor
    policy, cfg = loader_fn(args.action_predictor_ckpt, args.device)
    if not getattr(policy.model, 'predict_var', False):
        raise SystemExit('predictor has no variance head; train with policy.model.predict_var=True')

    aligned = None if args.aligned < 0 else bool(args.aligned)
    ds = build_eval_dataset(cfg, split=args.split, aligned=aligned)
    dl = make_loader(ds, batch_size=args.batch_size)

    e2_all, tr_all, var_all = [], [], []
    for i, batch in enumerate(dl):
        if i >= args.max_batches:
            break
        out = predictor_forward(policy, batch, args.device)
        e2 = ((out['target'] - out['mean']) ** 2).flatten(1).sum(1)
        tr = out['var'].flatten(1).sum(1)
        e2_all.append(e2.cpu()); tr_all.append(tr.cpu())
        var_all.append(out['var'].cpu())
    e2 = torch.cat(e2_all).double().numpy()
    tr = torch.cat(tr_all).double().numpy()
    var = torch.cat(var_all)
    d = var[0].numel()

    kappa = float(e2.mean() / tr.mean())
    res = {
        'n_samples': int(e2.shape[0]),
        'dim': int(d),
        'kappa': kappa,
        'mean_err2_per_dim': float(e2.mean() / d),
        'mean_pred_var_per_dim': float(tr.mean() / d),
        'pearson_err2_tr': float(np.corrcoef(e2, tr)[0, 1]),
        'spearman_err2_tr': spearman(e2, tr),
    }
    # reliability by deciles of predicted variance
    q = np.quantile(tr, np.linspace(0, 1, 11))
    rel = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (tr >= lo) & (tr <= hi)
        if m.sum() == 0:
            continue
        rel.append({'pred_tr_kappa': float(kappa * tr[m].mean()), 'actual_err2': float(e2[m].mean()),
                    'n': int(m.sum())})
    res['reliability_deciles'] = rel

    # implied start times / NFE under each delta
    sched = DDIMScheduler(num_train_timesteps=args.num_train_timesteps, beta_schedule='squaredcos_cap_v2',
                          clip_sample=True, set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')
    ac = torch.as_tensor(sched.alphas_cumprod)
    m_hat = torch.as_tensor(kappa * tr)
    res['implied'] = {}
    for delta in args.deltas:
        cfg_ws = AdaptiveWarmStartConfig(delta=delta, stride=args.stride, n_max=args.n_max)
        t_star = select_start_timesteps(m_hat, ac, cfg_ws)
        _, n = build_schedules(t_star, cfg_ws, num_train_timesteps=ac.shape[0])
        res['implied'][str(delta)] = {
            't_star_mean': float(t_star.float().mean()), 't_star_p90': float(t_star.float().quantile(0.9)),
            'nfe_mean': float(n.float().mean()), 'frac_1nfe': float((n <= 1).float().mean()),
            'frac_full': float((t_star >= args.num_train_timesteps - 1).float().mean()),
        }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != 'reliability_deciles'}, indent=2))
    print(f'kappa = {kappa:.4f}  -> pass --ustep_kappa {kappa:.4f} to eval_combined_inference.py')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x = [r['pred_tr_kappa'] for r in rel]; y = [r['actual_err2'] for r in rel]
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        lim = [0, max(max(x), max(y)) * 1.05]
        ax.plot(lim, lim, color='#999', lw=1, ls='--')
        ax.plot(x, y, marker='o', lw=1.5)
        ax.set_xlabel(r'predicted $\kappa\,\mathrm{tr}\,\hat\Sigma$'); ax.set_ylabel(r'actual $\|A-\hat\mu\|^2$')
        ax.set_title(f"spearman={res['spearman_err2_tr']:.2f}")
        fig.tight_layout()
        fig.savefig(os.path.splitext(args.out)[0] + '_reliability.png', dpi=200)
    except Exception as e:  # plotting is optional
        print(f'(skip plot: {e})')


if __name__ == '__main__':
    main()
