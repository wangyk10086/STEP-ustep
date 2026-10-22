"""One-step success vs start level on RoboMimic (official DP checkpoints) and Push-T.

    python rm_sweep_figure.py --rm data/v2/robomimic_v2_summary.json --pusht data/raw_results.csv \
        --out ../figs/fig_rm_sweep
"""
import argparse, json
from style import COL, plt, save, style_axes, INK, MUTED, RED, EDGE

TASKS = [('square', 'Square', '#FFC000', 'o'), ('can', 'Can', '#2E75B6', 's'),
         ('transport', 'Transport', '#70AD47', 'D'), ('tool_hang', 'ToolHang', '#7F7F7F', '^')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rm', required=True)
    p.add_argument('--pusht', default=None)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    d = json.load(open(a.rm))['closed_loop_fixed_start']
    fig, ax = plt.subplots(figsize=(COL, 2.05))
    style_axes(ax, grid='y')
    for key, lab, col, mk in TASKS:
        pts = sorted((int(k.split(':')[0]), v['score']) for k, v in d[key].items() if k.endswith(':1'))
        x, y = zip(*pts)
        ax.plot(x, y, color=col, marker=mk, lw=1.3, ms=4, mec=EDGE, mew=0.5, label=lab)
        if key in ('square', 'can'):  # STEP-1 / DDIM-1 were run on these two only
            ax.plot([0], [0], color=col, marker=mk, ms=4, mec=EDGE, mew=0.5, ls='none', alpha=0.9)
    if a.pusht:
        import pandas as pd
        r = pd.read_csv(a.pusht)
        r = r[(r.task == 'Push-T') & (r.obs == 'lowdim') & (r.method == 'fixed_start') & (r.nfe == 1)]
        pts = sorted({int(t): s for t, s in zip(r.t0, r.success_mean)}.items())
        x, y = zip(*pts)
        ax.plot(x, y, color=INK, ls=(0, (3, 1.5)), marker='x', lw=1.0, ms=4, label='Push-T')
    ax.annotate('STEP-1 ($t_0{=}0$): 0.00 on Can and Square, 0.09 on Push-T', xy=(1.0, 0.03), xytext=(4, 0.13), fontsize=6.8, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=0.7))
    ax.axvspan(87.5, 101, color='#F8CBAD', alpha=0.45, lw=0, zorder=0)
    ax.text(94.2, 1.0, 'RoboMimic\ncollapse', ha='center', va='top', fontsize=6.8, color=RED, style='italic')
    ax.set_xlim(-2, 101)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel('start level $t_0$ (one network evaluation)')
    ax.set_ylabel('success rate')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=1, fontsize=6.8, handlelength=1.6)
    save(fig, a.out)


if __name__ == '__main__':
    main()
