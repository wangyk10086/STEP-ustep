"""README / project-page teaser: (a) where denoising starts, (b) one-step rollouts, (c) one-step success.

    python teaser_figure.py --pusht_root data/v2/raw/eval/sim_pusht --square_root data/v2/raw_rm/sim_square \
        --out ../figs/teaser
"""
import argparse, json, os
import numpy as np
import imageio.v2 as imageio
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from style import PAGE, plt, save, INK, MUTED, RED, GOLD, GOLD_DARK, BLUE_FILL, BLUE_DARK, EDGE

GREEN = '#2E7D32'


def last_frame(root, man, method, seed):
    v = man[method][seed]['video']
    key = '/' + os.path.basename(os.path.normpath(root)) + '/'
    p = v if os.path.exists(v) else os.path.join(root, v.split(key)[-1])
    r = imageio.get_reader(p)
    fr = [f for f in r]
    r.close()
    return fr[-1]


def panel_a(ax):
    ax.set_xlim(-34, 106)
    ax.set_ylim(-0.95, 3.25)
    ax.axis('off')
    ax.text(-34, 3.2, '(a) Where denoising starts', fontsize=9, weight='bold', va='top')
    rows = [(2.35, 'DDIM', 100, BLUE_FILL, BLUE_DARK, 'starts from pure noise: one step is far too coarse', RED),
            (1.4, 'STEP', 0, BLUE_FILL, BLUE_DARK, 'starts at the guess ($t{=}0$): the predictor error survives', RED),
            (0.45, 'HiStart', 75, GOLD, GOLD_DARK, 're-noises the guess to $t_0$, then one step: works', GREEN)]
    for y, name, t0, fc, lc, note, ncol in rows:
        ax.text(-33, y, name, ha='left', va='center', fontsize=8.5, weight='bold' if name == 'HiStart' else 'normal')
        ax.plot([0, 100], [y, y], color='#BFBFBF', lw=2.2, solid_capstyle='round', zorder=1)
        if name == 'HiStart':
            ax.plot([50, 88], [y, y], color=GOLD, lw=5, alpha=0.35, solid_capstyle='round', zorder=1)
            ax.text(69, y + 0.2, 'good $t_0$', ha='center', va='bottom', fontsize=6.5, color=GOLD_DARK, style='italic')
            ax.add_patch(FancyArrowPatch((1.5, y - 0.07), (t0 - 1.5, y - 0.07), connectionstyle='arc3,rad=0.22',
                                         arrowstyle='-|>', mutation_scale=7, lw=0.8, ls=(0, (2.5, 1.5)), color=MUTED,
                                         zorder=3))
            ax.text(36, y - 0.18, 're-noise', ha='center', va='top', fontsize=6.3, color=MUTED)
        if t0 > 0:
            ax.add_patch(FancyArrowPatch((t0 - 2, y + 0.07), (2, y + 0.07), connectionstyle='arc3,rad=%s' % (0.16 if name == 'HiStart' else 0.25),
                                         arrowstyle='-|>', mutation_scale=9, lw=1.5 if name == 'HiStart' else 1.0,
                                         color=lc, zorder=4))
            ax.text(t0 / 2, y + 0.58, '1 step', ha='center', fontsize=6.3,
                    color=lc, va='bottom') if name == 'DDIM' else None
        ax.plot([t0], [y], 'o', ms=7.5, mfc=fc, mec=EDGE, mew=0.7, zorder=5)
        ax.plot([0], [y], 's', ms=5.5, mfc='white', mec=EDGE, mew=0.7, zorder=6)
        ax.text(50, y - (0.62 if name == 'HiStart' else 0.3), note, ha='center', va='top', fontsize=6.6, color=ncol)
    # noise axis
    yA = -0.55
    ax.annotate('', xy=(104, yA), xytext=(0, yA), arrowprops=dict(arrowstyle='->', color=INK, lw=0.7))
    for t, lab in ((0, '0\nclean'), (50, '50'), (100, '100\nnoise')):
        ax.plot([t, t], [yA - 0.04, yA + 0.04], color=INK, lw=0.7)
        ax.text(t, yA - 0.08, lab, ha='center', va='top', fontsize=6.3, color=MUTED, linespacing=0.95)
    ax.text(-33, yA, 'noise level $t$', ha='left', va='center', fontsize=7)


def panel_b(fig, gs, frames):
    sub = gs.subgridspec(2, 2, wspace=0.04, hspace=0.06)
    titles = ['STEP, 1 NFE', 'HiStart, 1 NFE']
    rows = ['Push-T', 'Square']
    marks = [['✗', '✓'], ['✗', '✓']]
    axes = []
    for r in range(2):
        for c in range(2):
            ax = fig.add_subplot(sub[r, c])
            ax.imshow(frames[r][c])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(GOLD if c == 1 else '#7F7F7F'); s.set_linewidth(1.6 if c == 1 else 0.6)
            ax.text(0.95, 0.06, marks[r][c], transform=ax.transAxes, ha='right', va='bottom', fontsize=12,
                    weight='bold', color=GREEN if c == 1 else RED, fontfamily='DejaVu Sans',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.8))
            if r == 0:
                ax.set_title(titles[c], fontsize=7.5, pad=2, weight='bold' if c == 1 else 'normal')
            if c == 0:
                ax.set_ylabel(rows[r], fontsize=7.5)
            axes.append(ax)
    axes[0].text(-0.12, 1.32, '(b) Same initial state, final frame', transform=axes[0].transAxes,
                 fontsize=9, weight='bold', va='bottom')


def panel_c(ax):
    tasks = ['Push-T', 'Can', 'Square']
    vals = {'DDIM': [0.12, 0.00, 0.00], 'STEP': [0.09, 0.00, 0.00], 'HiStart': [0.70, 0.99, 0.91]}
    cols = {'DDIM': '#D9D9D9', 'STEP': BLUE_FILL, 'HiStart': GOLD}
    x = np.arange(len(tasks)); w = 0.26
    for i, (k, v) in enumerate(vals.items()):
        b = ax.bar(x + (i - 1) * w, v, w, color=cols[k], edgecolor=EDGE, lw=0.6, label=k)
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.text(xi, vi + 0.02, f'{vi:.2f}'.lstrip('0') if vi > 0 else '0', ha='center', va='bottom',
                    fontsize=6.3, weight='bold' if k == 'HiStart' else 'normal',
                    color=RED if k == 'HiStart' else INK)
    ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=7.5)
    ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel('success, 1 NFE', fontsize=7.5)
    ax.legend(loc='upper left', fontsize=6.5, ncol=3, handlelength=1.0, columnspacing=0.8,
              bbox_to_anchor=(-0.02, 1.02))
    ax.set_title('(c) One network evaluation', loc='left', fontsize=9, weight='bold', pad=4)
    ax.text(0.99, -0.22, 'best $t_0$ per task; frozen DP', transform=ax.transAxes, ha='right', fontsize=6, color=MUTED)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pusht_root', required=True)
    p.add_argument('--square_root', required=True)
    p.add_argument('--pusht_seed', default='100014')
    p.add_argument('--square_seed', default='100002')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    mp = json.load(open(os.path.join(a.pusht_root, 'manifest.json')))
    ms = json.load(open(os.path.join(a.square_root, 'manifest.json')))
    frames = [[last_frame(a.pusht_root, mp, m, a.pusht_seed) for m in ('step', 'ours')],
              [last_frame(a.square_root, ms, m, a.square_seed) for m in ('step', 'ours')]]
    fig = plt.figure(figsize=(PAGE * 1.25, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.95, 0.95], wspace=0.28)
    panel_a(fig.add_subplot(gs[0]))
    panel_b(fig, gs[1], frames)
    panel_c(fig.add_subplot(gs[2]))
    save(fig, a.out)


if __name__ == '__main__':
    main()
