"""
Closed-loop rollouts with video for the qualitative figure: DDIM-1, STEP-1 and ours (fixed start
t0, one step) from the SAME initial states (same test seeds). Writes manifest.json mapping
method -> env seed -> {reward, video}.

  python experiments/v2/sim_rollouts.py --action_predictor_ckpt P --diffusion_policy_ckpt DP \
      --modality lowdim --t0 90 --n_test 16 --out /workspace/eval/sim_pusht
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
import argparse, json, os, re
from eval_combined_inference import evaluate_combined_policy


def collect(per_seed):
    out = {}
    for d in per_seed or []:
        for k, v in d.items():
            m = re.match(r'test/sim_max_reward_(\d+)$', k)
            if m:
                out.setdefault(m.group(1), {})['reward'] = float(v)
            m = re.match(r'test/sim_video_(\d+)$', k)
            if m:
                out.setdefault(m.group(1), {})['video'] = getattr(v, '_path', None) if not isinstance(v, dict) else v.get('path')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action_predictor_ckpt', required=True)
    p.add_argument('--diffusion_policy_ckpt', required=True)
    p.add_argument('--modality', default='lowdim')
    p.add_argument('--t0', type=int, default=90)
    p.add_argument('--n', type=int, default=1)
    p.add_argument('--n_test', type=int, default=16)
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    res = evaluate_combined_policy(
        a.action_predictor_ckpt, a.diffusion_policy_ckpt, a.out, device=a.device, modality=a.modality,
        n_test=a.n_test, n_test_vis=a.n_test, steps_list=[a.n], use_ddim=True, eval_seeds=[a.seed],
        ustep_deltas=[1.0], ustep_kwargs=dict(kappa=1.0, t_min=a.t0, t_max=a.t0, step_mode='fixed', n_fixed=a.n))
    man = {}
    for name, key in [('ddim', f'steps_{a.n}'), ('step', f'steps_{a.n}'), ('ours', 'ustep_delta1')]:
        r = res.get(key, {})
        if name == 'ddim':
            man[name] = collect(r.get('diffusion_only_per_seed_metrics'))
        elif name == 'step':
            man[name] = collect(r.get('combined_per_seed_metrics'))
        else:
            man[name] = collect(r.get('per_seed_metrics'))
    json.dump(man, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=2)
    for k, v in man.items():
        print(k, {s: round(e.get('reward', -1), 2) for s, e in sorted(v.items())})


if __name__ == '__main__':
    main()
