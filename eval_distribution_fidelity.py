"""
Offline distribution-fidelity / multimodality evaluation (U-STEP vs STEP vs DDIM).

For a set of held-out contexts, draw K action chunks from each sampler and
compare them with K chunks from the full 100-step DDPM diffusion policy
(the distribution STEP claims to preserve):

  mmd       : unbiased RBF-MMD^2 to the reference set            (lower better)
  precision : fraction of samples inside the reference support    (quality)
  recall    : fraction of reference support covered by samples    (mode coverage;
              precision~1 & recall<<1 is the signature of mode collapse)
  div_ratio : mean pairwise distance / reference mean pairwise distance
  min_ade   : min over K samples of the distance to the demonstrated chunk
  nfe       : average network evaluations per chunk

--history none  uses prev_action = 0 (what the predictor sees at episode start);
                the right target is then p(A|o), exactly what the DP models.
--history data  uses the dataset's previous chunk (aligned or STEP-style,
                as the predictor was trained).

Usage:
  python eval_distribution_fidelity.py --action_predictor_ckpt PRED.ckpt \
     --diffusion_policy_ckpt DP.ckpt --modality lowdim --deltas 0.03 0.1 0.3 \
     --step_steps 2 4 --ddim_steps 2 4 --history none --out data/ustep/fid_pusht.json
"""

if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).parent))

import argparse
import copy
import json
import os
from collections import defaultdict

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.adaptive_warmstart import AdaptiveWarmStartConfig
from diffusion_policy.policy.ustep_offline import (
    build_eval_dataset, make_loader, predictor_forward,
    mmd_rbf, precision_recall, diversity)


def _ddim(num_train_timesteps):
    return DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule='squaredcos_cap_v2',
                         clip_sample=True, set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')


def _set_sched(dp, sched):
    dp.noise_scheduler = sched
    if hasattr(dp, 'policy'):
        dp.policy.noise_scheduler = sched


def _repeat_obs(obs, K):
    if isinstance(obs, dict):
        return {k: v.repeat_interleave(K, dim=0) for k, v in obs.items()}
    return obs.repeat_interleave(K, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action_predictor_ckpt', required=True)
    p.add_argument('--diffusion_policy_ckpt', required=True)
    p.add_argument('--modality', default='lowdim', choices=['lowdim', 'image'])
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--history', default='none', choices=['none', 'data'])
    p.add_argument('--n_contexts', type=int, default=128)
    p.add_argument('--ctx_batch', type=int, default=8)
    p.add_argument('--k_samples', type=int, default=32)
    p.add_argument('--deltas', type=float, nargs='*', default=[0.03, 0.1, 0.3])
    p.add_argument('--kappa', type=float, default=1.0)
    p.add_argument('--n_max', type=int, default=10)
    p.add_argument('--step_steps', type=int, nargs='*', default=[2, 4])
    p.add_argument('--ddim_steps', type=int, nargs='*', default=[2, 4])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='data/ustep/fidelity.json')
    args = p.parse_args()
    torch.manual_seed(args.seed)

    from eval_combined_inference import (load_action_predictor, load_action_predictor_image,
                                         load_diffusion_policy, load_diffusion_policy_image)
    if args.modality == 'image':
        pred, pred_cfg = load_action_predictor_image(args.action_predictor_ckpt, args.device)
        dp, _ = load_diffusion_policy_image(args.diffusion_policy_ckpt, args.device, use_ema=True)
    else:
        pred, pred_cfg = load_action_predictor(args.action_predictor_ckpt, args.device)
        dp, _ = load_diffusion_policy(args.diffusion_policy_ckpt, args.device, use_ema=True)
    has_var = getattr(pred.model, 'predict_var', False)
    if args.deltas and not has_var:
        print('[warn] predictor has no variance head -> skipping U-STEP rows')
        args.deltas = []
    ddpm = copy.deepcopy(dp.noise_scheduler)
    T_train = ddpm.config.num_train_timesteps
    norm_a = dp.normalizer['action']

    ds = build_eval_dataset(pred_cfg, split='val')
    dl = make_loader(ds, batch_size=args.ctx_batch, num_workers=2, seed=args.seed)

    K = args.k_samples
    rows = defaultdict(lambda: defaultdict(list))
    n_done = 0

    def sample(name, obs_rep, init=None, var=None):
        out = dp.predict_action({'obs': obs_rep}, init_action=init, **({'init_var': var} if var is not None else {}))
        a = norm_a.normalize(out['action_pred'])
        ws = out.get('warmstart_stats', None)
        nfe = float(ws['nfe'].float().mean()) if ws is not None else None
        return a, nfe

    with torch.no_grad():
        for batch in dl:
            if n_done >= args.n_contexts:
                break
            if args.history == 'none':
                batch['prev_valid'] = torch.zeros(batch['action'].shape[0])
            C = batch['action'].shape[0]
            pf = predictor_forward(pred, batch, args.device)
            obs = dict_apply(batch['obs'], lambda x: x.to(args.device)) if isinstance(batch['obs'], dict) \
                else batch['obs'].to(args.device)
            obs_rep = _repeat_obs(obs, K)
            init_rep = pf['mean_raw'].repeat_interleave(K, 0)
            var_rep = pf['var'].repeat_interleave(K, 0) if pf['var'] is not None else None
            gt = norm_a.normalize(batch['action'].to(args.device))

            results = {}
            # reference: full DDPM
            _set_sched(dp, copy.deepcopy(ddpm)); dp.set_adaptive_warmstart(None) if hasattr(dp, 'set_adaptive_warmstart') else None
            dp.num_inference_steps = T_train
            results['ref_ddpm'] = (sample('ref', obs_rep)[0], T_train)
            # DDIM from noise
            for k in args.ddim_steps:
                _set_sched(dp, _ddim(T_train)); dp.num_inference_steps = k
                results[f'ddim_{k}'] = (sample('ddim', obs_rep)[0], k)
            # STEP: fixed warm start
            for k in args.step_steps:
                _set_sched(dp, _ddim(T_train)); dp.init_trajectory_steps = k
                results[f'step_{k}'] = (sample('step', obs_rep, init=init_rep)[0], k)
            # predictor mean only (direct prediction)
            results['predictor_only'] = (pf['mean'].repeat_interleave(K, 0), 0)
            # U-STEP
            for delta in args.deltas:
                _set_sched(dp, _ddim(T_train))
                dp.set_adaptive_warmstart(AdaptiveWarmStartConfig(delta=delta, kappa=args.kappa, n_max=args.n_max))
                a, nfe = sample('ustep', obs_rep, init=init_rep, var=var_rep)
                results[f'ustep_delta{delta:g}'] = (a, nfe)
                dp.set_adaptive_warmstart(None)

            H = min(v[0].shape[1] for v in results.values())
            ref = results['ref_ddpm'][0][:, :H].view(C, K, -1).cpu()
            gt_f = gt[:, :H].reshape(C, -1).cpu()
            ref_div = [diversity(ref[c]) for c in range(C)]
            for name, (a, nfe) in results.items():
                a = a[:, :H].reshape(C, K, -1).cpu()
                for c in range(C):
                    r = rows[name]
                    if name != 'ref_ddpm':
                        r['mmd'].append(mmd_rbf(a[c], ref[c]))
                        pr, rc = precision_recall(a[c], ref[c])
                        r['precision'].append(pr); r['recall'].append(rc)
                    r['div_ratio'].append(diversity(a[c]) / max(ref_div[c], 1e-8))
                    r['min_ade'].append(float((a[c] - gt_f[c]).norm(dim=-1).min()))
                    r['nfe'].append(float(nfe) if nfe is not None else np.nan)
            n_done += C
            print(f'{n_done}/{args.n_contexts} contexts')

    summary = {name: {k: float(np.nanmean(v)) for k, v in r.items()} for name, r in rows.items()}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'args': vars(args), 'summary': summary}, f, indent=2)
    cols = ['nfe', 'mmd', 'precision', 'recall', 'div_ratio', 'min_ade']
    print(f"{'method':18s}" + ''.join(f'{c:>11s}' for c in cols))
    for name, r in summary.items():
        print(f'{name:18s}' + ''.join(f"{r.get(c, float('nan')):11.4f}" for c in cols))


if __name__ == '__main__':
    main()
