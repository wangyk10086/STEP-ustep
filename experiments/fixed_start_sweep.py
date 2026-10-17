"""
Fixed-start sweep (closed loop): every sample starts from the exact forward
marginal at the same timestep t0 and is integrated to 0 with n DDIM steps.

This is the population version of Corollary 1 (one t0 for all states) and the
key ablation for per-sample adaptivity.  STEP-k corresponds to t0 = T(k-1)/k
with DDIM "leading" spacing (k=2 -> t0=50, k=4 -> t0=75).

  python experiments/fixed_start_sweep.py --action_predictor_ckpt PRED.ckpt \
      --diffusion_policy_ckpt DP.ckpt --grid 75:4 85:4 95:4 85:3 95:3 85:2 95:2 90:1 95:1 \
      --kappa K --out_root data/outputs/fixed_start
"""
import argparse, json, os, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from eval_combined_inference import evaluate_combined_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action_predictor_ckpt', required=True)
    p.add_argument('--diffusion_policy_ckpt', required=True)
    p.add_argument('--modality', default='lowdim', choices=['lowdim', 'image'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--grid', nargs='+', required=True, help='t0:n pairs, e.g. 95:2 90:1')
    p.add_argument('--kappa', type=float, default=1.0)
    p.add_argument('--n_test', type=int, default=50)
    p.add_argument('--eval_seeds', type=int, nargs='+', default=[100000, 100100, 100200, 100300, 100400])
    p.add_argument('--out_root', default='data/outputs/fixed_start')
    args = p.parse_args()

    summary = {}
    for spec in args.grid:
        t0, n = (int(x) for x in spec.split(':'))
        out = os.path.join(args.out_root, f't{t0}_n{n}')
        res = evaluate_combined_policy(
            args.action_predictor_ckpt, args.diffusion_policy_ckpt, out, device=args.device,
            modality=args.modality, n_test=args.n_test, n_test_vis=0, steps_list=[], use_ddim=True,
            eval_seeds=args.eval_seeds, skip_baselines=True, ustep_deltas=[1.0],
            ustep_kwargs=dict(kappa=args.kappa, t_min=t0, t_max=t0, step_mode='fixed', n_fixed=n))
        m = res['ustep_delta1']['metrics']
        summary[spec] = {'score': float(m.get('test/mean_score', 'nan')),
                         'std': float(m.get('test/mean_score_std', 0.0))}
        print(f'[fixed-start] t0={t0} n={n}: {summary[spec]}', flush=True)
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
