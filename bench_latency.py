"""
Batch-size-1 latency benchmark (the deployment setting) for
DDPM-100 / DDIM-k / STEP-k / U-STEP(delta), on held-out contexts.

  python bench_latency.py --action_predictor_ckpt PRED.ckpt --diffusion_policy_ckpt DP.ckpt \
      --steps 1 2 4 --deltas 0.1 0.3 1 --kappa K --n_calls 300 --out data/ustep/latency.json
"""
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).parent))

import argparse, copy, json, os, time
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.policy.adaptive_warmstart import AdaptiveWarmStartConfig
from diffusion_policy.policy.ustep_offline import build_eval_dataset, make_loader


def _ddim(T):
    return DDIMScheduler(num_train_timesteps=T, beta_schedule='squaredcos_cap_v2', clip_sample=True,
                         set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action_predictor_ckpt', required=True)
    p.add_argument('--diffusion_policy_ckpt', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', type=int, nargs='*', default=[1, 2, 4])
    p.add_argument('--deltas', type=float, nargs='*', default=[0.1, 0.3, 1.0])
    p.add_argument('--kappa', type=float, default=1.0)
    p.add_argument('--n_max', type=int, default=10)
    p.add_argument('--n_calls', type=int, default=300)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--out', default='data/ustep/latency.json')
    args = p.parse_args()

    from eval_combined_inference import load_action_predictor, load_diffusion_policy
    pred, pcfg = load_action_predictor(args.action_predictor_ckpt, args.device)
    dp, _ = load_diffusion_policy(args.diffusion_policy_ckpt, args.device, use_ema=True)
    pred.set_normalizer(dp.normalizer)
    ddpm = copy.deepcopy(dp.noise_scheduler)
    T = ddpm.config.num_train_timesteps

    ds = build_eval_dataset(pcfg, split='val')
    dl = make_loader(ds, batch_size=1, num_workers=0, shuffle=True)
    batches = []
    for b in dl:
        batches.append(b)
        if len(batches) >= args.n_calls + args.warmup:
            break

    def sync():
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    def timed(fn):
        ts, extra = [], []
        for i, b in enumerate(batches):
            obs = {'obs': b['obs'].to(args.device)}
            prev = b['prev_action'].to(args.device)
            sync(); t0 = time.perf_counter()
            e = fn(obs, prev)
            sync(); dt = (time.perf_counter() - t0) * 1000
            if i >= args.warmup:
                ts.append(dt); extra.append(e)
        return ts, extra

    results = {}
    with torch.no_grad():
        dp.set_adaptive_warmstart(None)
        dp.noise_scheduler = copy.deepcopy(ddpm); dp.num_inference_steps = T
        ts, _ = timed(lambda o, pv: dp.predict_action(o))
        results['ddpm_100'] = {'ms_mean': float(np.mean(ts)), 'ms_p90': float(np.percentile(ts, 90)), 'nfe': T}
        for k in args.steps:
            dp.noise_scheduler = _ddim(T); dp.num_inference_steps = k
            ts, _ = timed(lambda o, pv: dp.predict_action(o))
            results[f'ddim_{k}'] = {'ms_mean': float(np.mean(ts)), 'ms_p90': float(np.percentile(ts, 90)), 'nfe': k}
        # predictor alone
        ts, _ = timed(lambda o, pv: pred.predict_action(o, prev_action=pv))
        results['predictor_only'] = {'ms_mean': float(np.mean(ts)), 'ms_p90': float(np.percentile(ts, 90)), 'nfe': 0}
        for k in args.steps:
            dp.noise_scheduler = _ddim(T); dp.init_trajectory_steps = k
            def f(o, pv):
                r = pred.predict_action(o, prev_action=pv)
                dp.predict_action(o, init_action=r['action_pred'])
            ts, _ = timed(f)
            results[f'step_{k}'] = {'ms_mean': float(np.mean(ts)), 'ms_p90': float(np.percentile(ts, 90)), 'nfe': k}
        if getattr(pred.model, 'predict_var', False):
            for d in args.deltas:
                dp.noise_scheduler = _ddim(T)
                dp.set_adaptive_warmstart(AdaptiveWarmStartConfig(delta=d, kappa=args.kappa, n_max=args.n_max))
                def g(o, pv):
                    r = pred.predict_action(o, prev_action=pv)
                    out = dp.predict_action(o, init_action=r['action_pred'], init_var=r['action_var_normalized'])
                    return float(out['warmstart_stats']['nfe'].float().mean())
                ts, nfe = timed(g)
                results[f'ustep_delta{d:g}'] = {'ms_mean': float(np.mean(ts)), 'ms_p90': float(np.percentile(ts, 90)),
                                                'nfe': float(np.mean(nfe)), 'frac_1nfe': float(np.mean(np.array(nfe) <= 1))}
            dp.set_adaptive_warmstart(None)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    with open(args.out, 'w') as f:
        json.dump({'gpu': gpu, 'batch_size': 1, 'results': results}, f, indent=2)
    print(f'GPU: {gpu}  (batch size 1, {args.n_calls} calls)')
    print(f"{'method':16s}{'NFE':>7s}{'mean ms':>10s}{'p90 ms':>10s}")
    for k, r in results.items():
        print(f"{k:16s}{r['nfe']:7.2f}{r['ms_mean']:10.2f}{r['ms_p90']:10.2f}")


if __name__ == '__main__':
    main()
