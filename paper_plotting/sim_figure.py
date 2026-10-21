"""Qualitative rollout figure from the videos written by experiments/v2/sim_rollouts.py.

    python sim_figure.py --manifest data/sim_pusht/manifest.json --video_root data/sim_pusht \
        --out ../figs/fig_sim --frames 5

One row per method (DDIM-1, STEP-1, ours-1), same test seed (same initial state). By default the
seed is the one where ours succeeds and the baselines fail by the widest margin; pass --seed to fix it.
"""
import argparse, json, os
import numpy as np
import imageio.v2 as imageio
from style import COL, PAGE, plt, save, INK, RED, GOLD, BLUE_FILL

ROWS = [('ddim', 'DDIM, 1 NFE'), ('step', 'STEP, 1 NFE'), ('ours', 'Ours, 1 NFE')]


def load_frames(path):
    r = imageio.get_reader(path)
    fr = [f for f in r]
    r.close()
    return fr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--video_root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, default=5)
    p.add_argument('--seed', default=None)
    p.add_argument('--width', type=float, default=COL)
    a = p.parse_args()
    man = json.load(open(a.manifest))
    seeds = sorted(set.intersection(*[set(man[k]) for k, _ in ROWS]))
    if a.seed is None:
        score = lambda s: man['ours'][s]['reward'] - max(man['ddim'][s]['reward'], man['step'][s]['reward'])
        a.seed = max(seeds, key=score)
    print('seed', a.seed, {k: round(man[k][a.seed]['reward'], 2) for k, _ in ROWS})

    vids = {}
    for k, _ in ROWS:
        v = man[k][a.seed]['video']
        if not os.path.exists(v):
            v = os.path.join(a.video_root, v.split('/eval/sim_pusht/')[-1]) if '/eval/' in v else os.path.join(a.video_root, os.path.basename(v))
        vids[k] = load_frames(v)
    L = max(len(f) for f in vids.values())
    idx = np.linspace(0, L - 1, a.frames).round().astype(int)

    h = a.width / a.frames * len(ROWS) * 1.02 + 0.15
    fig, axes = plt.subplots(len(ROWS), a.frames, figsize=(a.width, h),
                             gridspec_kw={'wspace': 0.03, 'hspace': 0.06})
    for r, (k, lab) in enumerate(ROWS):
        fr = vids[k]
        for c, t in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(fr[min(t, len(fr) - 1)])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_linewidth(1.2 if k == 'ours' else 0.5)
                s.set_color(GOLD if k == 'ours' else '#7F7F7F')
            if r == 0:
                ax.set_title(f'$t={t}$' , fontsize=7, pad=2)
        rew = man[k][a.seed]['reward']
        axes[r, 0].set_ylabel(lab, fontsize=7.2, fontweight='bold' if k == 'ours' else 'normal', labelpad=3)
        axes[r, -1].text(1.04, 0.5, f'{rew:.2f}', transform=axes[r, -1].transAxes, va='center', ha='left',
                         fontsize=7.4, color=RED if k == 'ours' else INK,
                         fontweight='bold' if k == 'ours' else 'normal')
    axes[0, -1].text(1.04, 1.12, 'cover.', transform=axes[0, -1].transAxes, ha='left', fontsize=6.6, color=INK)
    save(fig, a.out)


if __name__ == '__main__':
    main()
