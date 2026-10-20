"""Closed-loop eval of the Consistency Policy student (1-step and 3-step chaining), 3 seeds x 50 episodes.
usage (from /workspace/CP): python cp_eval.py STUDENT.ckpt OUT.json"""
import sys, os, json, copy, tempfile
sys.path.insert(0, os.getcwd())
import torch, dill, hydra, numpy as np
from omegaconf import OmegaConf, open_dict
OmegaConf.register_new_resolver("eval", eval, replace=True)
from consistency_policy.utils import load_normalizer

ckpt, out = sys.argv[1], sys.argv[2]
payload = torch.load(open(ckpt, 'rb'), pickle_module=dill)
cfg = payload['cfg']
cfg.training.inference_mode = True
cfg.training.online_rollouts = False
ws = hydra.utils.get_class(cfg._target_)(cfg)
ws.load_checkpoint(path=ckpt, exclude_keys=['optimizer'])
policy = ws.ema_model if (cfg.training.use_ema and getattr(ws, 'ema_model', None) is not None) else ws.model
policy.set_normalizer(load_normalizer(torch.load(ckpt)))
policy = policy.to('cuda').eval()
print('using', 'ema' if policy is getattr(ws, 'ema_model', None) else 'model', flush=True)

res = {}
for mode in ['1step', '3step']:
    if mode == '1step':
        policy.disable_chaining()
    else:
        policy.enable_chaining()
    scores = []
    for seed in [100000, 100100, 100200]:
        rc = copy.deepcopy(cfg.task.env_runner)
        with open_dict(rc):
            rc.n_envs = 14; rc.n_test = 50; rc.n_test_vis = 0; rc.n_train = 0; rc.n_train_vis = 0
            rc.test_start_seed = seed
        runner = hydra.utils.instantiate(rc, output_dir=tempfile.mkdtemp())
        with torch.no_grad():
            log = runner.run(policy)
        s = float(log['test/mean_score']); scores.append(s)
        print(mode, seed, s, flush=True)
        try: runner.close()
        except Exception: pass
    res[mode] = {'mean': float(np.mean(scores)), 'std': float(np.std(scores)), 'per_seed': scores}
    print('RESULT', mode, res[mode], flush=True)
json.dump(res, open(out, 'w'), indent=2)
print('CP_EVAL_DONE', flush=True)
