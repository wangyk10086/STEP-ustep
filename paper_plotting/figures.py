"""All paper figures, generated from data/results_clean.csv.

    python clean_data.py      # once, or whenever raw_results.csv changes
    python figures.py         # writes ../figs/fig_*.pdf and ../figs/fig_*.png

In the paper:
Fig. 1  fig_overview  full width   pipeline / where denoising starts / bound that sets t0
Fig. 2  fig_t0        one column   start-level sensitivity
Spare (python figures.py --all):
        fig_nfe       full width   success vs NFE on all eight settings (+-1 std over seeds)
        fig_one_step  one column   1-NFE success: ours vs STEP / DDIM / Consistency Policy
"""
import math
import os

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from style import (COL, COLORS, GRID, INK, LABELS, MUTED, OURS_RAMP, PAGE, method_style,
                   panel_title, plt, save, style_axes)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs')
DF = pd.read_csv(os.path.join(HERE, 'data', 'results_clean.csv'))

# The eight evaluation settings, in display order: (label, task, obs)
SETTINGS = [('Push-T', 'Push-T', 'lowdim'), ('Push-T (img)', 'Push-T', 'image'),
            ('Can', 'Can', 'lowdim'), ('Can (img)', 'Can', 'image'),
            ('Square', 'Square', 'lowdim'), ('Square (img)', 'Square', 'image'),
            ('Transport', 'Transport', 'lowdim'), ('ToolHang', 'ToolHang', 'lowdim')]
OURS_T0 = {1: 90, 2: 85, 4: 95}   # one global start level per step budget


def get(task, obs, method, nfe, t0=None, prefer_reported=True):
    """(mean, std) for one configuration, or None if it was not run."""
    q = DF[(DF.task == task) & (DF.obs == obs) & (DF.method == method) & (DF.nfe == nfe)]
    if method == 'Ours':
        q = q[q.t0 == (t0 if t0 is not None else OURS_T0.get(nfe))]
    if q.empty:
        return None
    if prefer_reported and (q.source == 'STEP paper').any():
        q = q[q.source == 'STEP paper']
    r = q.iloc[0]
    return float(r['mean']), (None if pd.isna(r['std']) else float(r['std']))


def series(task, obs, method, nfes=(1, 2, 3, 4)):
    pts = [(n, get(task, obs, method, n)) for n in nfes]
    return [(n, v[0], v[1]) for n, v in pts if v is not None]


def schedule(T=100):
    f = lambda u: math.cos((u / T + 0.008) / 1.008 * math.pi / 2) ** 2
    ab, p = [], 1.0
    for i in range(T):
        b = min(1 - f(i + 1) / f(i), 0.999); p *= 1 - b; ab.append(p)
    return ab


# ================================================================ Fig. 1: overview
def fig_overview():
    from style import BLUE_FILL, BLUE_LIGHT, GOLD, GOLD_LIGHT, GOLD_DARK, EDGE, RED
    fig = plt.figure(figsize=(PAGE, 2.05))
    ax = fig.add_axes([0, 0, 0.64, 1]); ax.set_xlim(0, 64); ax.set_ylim(0, 20.5); ax.axis('off')

    def box(xc, yc, w, text, fc=BLUE_FILL, bold=False, h=2.5, fs=7.6):
        ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                                    boxstyle='round,pad=0,rounding_size=0.45', fc=fc, ec=EDGE, lw=0.7, zorder=2))
        ax.text(xc, yc, text, ha='center', va='center', fontsize=fs, color=INK,
                fontweight='bold' if bold else 'normal', zorder=3, linespacing=1.0)
        return xc - w / 2, xc + w / 2

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=7,
                                     color=INK, lw=0.7, shrinkA=0, shrinkB=0, zorder=1))

    def bump(xc, y0, width=2.6, height=1.1, multi=False):
        x = np.linspace(-1, 1, 80)
        if multi:
            yy = 0.9 * np.exp(-((x + 0.35) / 0.18) ** 2) + np.exp(-((x - 0.3) / 0.2) ** 2)
        else:
            yy = np.exp(-(x / 0.35) ** 2)
        ax.plot(xc + x * width / 2, y0 + yy / yy.max() * height, color=INK, lw=0.6)

    # frame and header, as in STEP's Fig. 1
    top, bot = 19.6, 0.4
    ax.add_patch(plt.Rectangle((0.3, bot), 63.4, top - bot, fc='none', ec=INK, lw=0.7))
    cols = dict(method=4.6, start=50.5, nfe=59.0)
    hy = 18.3
    for k, s in (('method', 'Method'), ('start', 'Start level'), ('nfe', 'NFE')):
        ax.text(cols[k], hy, s, ha='center', va='center', fontsize=7.6, fontweight='bold')
    ax.text(27.5, hy, 'Inference pipeline', ha='center', va='center', fontsize=7.6, fontweight='bold')
    ax.plot([0.3, 63.7], [17.2, 17.2], color=INK, lw=0.6)
    for yl in (11.9, 6.6):
        ax.plot([0.3, 63.7], [yl, yl], color='#A6A6A6', lw=0.5)
    ax.plot([46.0, 46.0], [bot, top], color='#A6A6A6', lw=0.5, ls=(0, (3, 2)))

    # row 1: DDIM
    y = 14.6
    ax.text(cols['method'], y, 'DDIM', ha='center', va='center', fontsize=7.8)
    ax.text(12.5, y, 'Noise', ha='center', va='center', fontsize=7.6)
    bump(12.5, y - 2.4)
    l, r = box(27.5, y, 7.5, 'Policy')
    arrow(15.0, l, y); arrow(r, 37.6, y)
    ax.text(40.6, y, 'Action', ha='center', va='center', fontsize=7.6)
    ax.text(cols['start'], y, '$t=100$', ha='center', va='center', fontsize=7.6)
    ax.text(cols['nfe'], y, '4–8', ha='center', va='center', fontsize=7.6)

    # row 2: STEP
    y = 9.3
    ax.text(cols['method'], y, 'STEP', ha='center', va='center', fontsize=7.8)
    ax.text(12.5, y + 0.55, 'Previous', ha='center', va='center', fontsize=7.6)
    ax.text(12.5, y - 0.65, 'Action', ha='center', va='center', fontsize=7.6)
    l1, r1 = box(21.5, y, 6.6, 'Predictor', fc=BLUE_LIGHT, fs=7.2, h=2.2)
    l2, r2 = box(32.0, y, 7.5, 'Policy')
    arrow(15.6, l1, y); arrow(r1, l2, y); arrow(r2, 37.6, y)
    ax.text(40.6, y, 'Action', ha='center', va='center', fontsize=7.6)
    ax.text(cols['start'], y, '$t=50$', ha='center', va='center', fontsize=7.6)
    ax.text(cols['nfe'], y, '2', ha='center', va='center', fontsize=7.6)

    # row 3: ours
    y = 3.9
    ax.text(cols['method'], y, 'Ours', ha='center', va='center', fontsize=7.8, fontweight='bold')
    ax.text(12.5, y + 0.55, 'Previous', ha='center', va='center', fontsize=7.6, fontweight='bold')
    ax.text(12.5, y - 0.65, 'Action', ha='center', va='center', fontsize=7.6, fontweight='bold')
    l1, r1 = box(19.9, y, 5.6, 'Predictor', fc=BLUE_LIGHT, fs=7.2, h=2.2)
    l2, r2 = box(29.4, y, 9.6, 'Re-noise to $t_0$', fc=GOLD, bold=True, h=2.5, fs=7.4)
    l3, r3 = box(38.6, y, 5.6, 'Policy', bold=True)
    arrow(15.6, l1, y); arrow(r1, l2, y); arrow(r2, l3, y); arrow(r3, 41.9, y)
    ax.text(44.4, y, 'Action', ha='center', va='center', fontsize=7.6, fontweight='bold')
    ax.text(29.4, y - 2.25, '$\\sqrt{\\bar\\alpha_{t_0}}\\,\\hat a+\\sigma_{t_0}\\epsilon$', ha='center',
            va='center', fontsize=6.8)
    ax.text(38.6, y + 1.95, 'frozen', ha='center', va='center', fontsize=6.4, style='italic', color=MUTED)
    ax.text(cols['start'], y, '$t_0\\approx 90$', ha='center', va='center', fontsize=7.6, fontweight='bold')
    ax.text(cols['nfe'], y, '1', ha='center', va='center', fontsize=8.2, fontweight='bold', color=RED)
    ax.text(0.9, 20.0, '(a)', fontsize=8, fontweight='bold', va='bottom')

    # (b) error decomposition at one step (illustrative constants; replace m, C with offline estimates)
    from matplotlib.patches import Rectangle
    ab = schedule(); lam = [0.5 * math.log(a / (1 - a)) for a in ab]
    m, C = 0.6, 0.007
    ts = [50, 70, 80, 90, 95, 99]
    mis = [0.5 * math.sqrt(math.exp(2 * lam[t]) * m) for t in ts]
    sol = [C * (lam[0] - lam[t]) ** 2 for t in ts]
    ins = fig.add_axes([0.715, 0.17, 0.27, 0.66])
    for i, (a, d) in enumerate(zip(mis, sol)):
        ins.bar(i, a, 0.62, color=BLUE_FILL, ec=EDGE, lw=0.6)
        ins.bar(i, d, 0.62, bottom=a, color=GOLD, ec=EDGE, lw=0.6)
    tot = [a + d for a, d in zip(mis, sol)]; b = int(np.argmin(tot))
    ins.plot(b, tot[b] + 0.035, marker='*', ms=9, color=RED, mec=EDGE, mew=0.4, clip_on=False)
    ins.text(b, tot[b] + 0.075, 'best', color=RED, fontsize=7.5, fontweight='bold', style='italic', ha='center')
    ins.set_xticks(range(len(ts))); ins.set_xticklabels([str(t) for t in ts]); ins.tick_params(axis='x', length=0)
    ins.set_yticks([]); ins.set_ylabel('error (illustrative)', labelpad=2)
    ins.set_xlabel('start level $t_0$ (one step)', labelpad=1)
    ins.legend(handles=[Rectangle((0, 0), 1, 1, fc=GOLD, ec=EDGE, lw=0.6),
                        Rectangle((0, 0), 1, 1, fc=BLUE_FILL, ec=EDGE, lw=0.6)],
               labels=['solver error', 'start mismatch'], loc='lower center', bbox_to_anchor=(0.5, 0.98),
               ncol=2, fontsize=7, columnspacing=1.0, handletextpad=0.4)
    fig.text(0.665, 0.985, '(b)', fontsize=8, fontweight='bold', va='top')
    save(fig, os.path.join(OUT, 'fig_overview'))


# ================================================================ Fig. 2: success vs NFE
def fig_nfe():
    fig, axes = plt.subplots(2, 4, figsize=(PAGE, 2.8), sharex=True, sharey=True)
    for a, (label, task, obs) in zip(axes.flat, SETTINGS):
        for meth in ('DDIM', 'STEP', 'CP', 'Ours'):
            pts = series(task, obs, meth, nfes=(1, 2, 3, 4) if meth == 'CP' else (1, 2, 4))
            if not pts:
                continue
            xs, ys, ss = zip(*pts)
            st = method_style(meth)
            a.plot(xs, ys, **st)
            if any(s is not None for s in ss):
                lo = [y - (s or 0) for y, s in zip(ys, ss)]
                hi = [y + (s or 0) for y, s in zip(ys, ss)]
                a.fill_between(xs, lo, hi, color=COLORS[meth], alpha=0.13 if meth == 'Ours' else 0.08,
                               lw=0, zorder=1)
        o1 = get(task, obs, 'Ours', 1)[0]
        a.text(1.0, o1 + 0.08, f'{o1:.2f}', fontsize=6.4, color=COLORS['Ours'], va='bottom', ha='center',
               bbox=dict(fc='white', ec='none', pad=0.4, alpha=0.85), zorder=6)
        panel_title(a, label)
        a.set_xscale('log', base=2); a.set_xticks([1, 2, 4]); a.set_xticklabels(['1', '2', '4'])
        a.minorticks_off()
        a.set_xlim(0.85, 4.6); a.set_ylim(-0.04, 1.06)
        a.set_yticks([0, 0.5, 1]); a.set_yticklabels(['0', '.5', '1'])
        style_axes(a)
    for a in axes[1]:
        a.set_xlabel('network evaluations (NFE)', labelpad=1)
    for a in axes[:, 0]:
        a.set_ylabel('success rate', labelpad=2)
    handles = [Line2D([], [], **{k: v for k, v in method_style(m).items() if k != 'zorder'},
                      label=LABELS[m] + (' (distilled)' if m == 'CP' else ''))
               for m in ('Ours', 'STEP', 'DDIM', 'CP')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.985), ncol=4,
               handlelength=2.0, columnspacing=1.6, handletextpad=0.4)
    fig.tight_layout(h_pad=0.6, w_pad=0.5)
    save(fig, os.path.join(OUT, 'fig_nfe'))


# ================================================================ Fig. 3: one-step success
def fig_one_step():
    rows = []
    for label, task, obs in SETTINGS:
        d, s, o = (get(task, obs, m, 1) for m in ('DDIM', 'STEP', 'Ours'))
        if d is None or s is None:
            continue
        c = get(task, obs, 'CP', 1)
        rows.append((label, d[0], s[0], o, c))
    rows.sort(key=lambda r: -r[3][0])
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    for i, (name, d, s, (o, osd), c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.axhspan(y - 0.5, y + 0.5, color='#F7F8F9' if i % 2 == 0 else 'white', zorder=0, lw=0)
        ax.plot([max(d, s), o], [y, y], color='#DCEAF6', lw=2.6, solid_capstyle='round', zorder=1)
        ax.plot(d, y + 0.13, 'o', ms=4.0, mfc='white', mec=COLORS['DDIM'], mew=0.9, zorder=2)
        ax.plot(s, y - 0.13, 's', ms=3.6, color=COLORS['STEP'], mec='white', mew=0.5, zorder=3)
        if c is not None:
            ax.errorbar(c[0], y, xerr=c[1], fmt='D', ms=4.0, color=COLORS['CP'], mec='white', mew=0.5,
                        elinewidth=0.8, zorder=3)
            ax.text(c[0], y + 0.28, 'CP', fontsize=6.2, color=COLORS['CP'], ha='center', va='bottom')
        ax.errorbar(o, y, xerr=osd, fmt='o', ms=5.2, color=COLORS['Ours'], mec='white', mew=0.7,
                    elinewidth=0.9, zorder=4)
        ax.text(o + (osd or 0) + 0.025, y, f'{o:.2f}', va='center', fontsize=7, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlim(-0.03, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '.25', '.50', '.75', '1'])
    ax.set_xlabel('success rate with a single network evaluation')
    ax.spines['left'].set_visible(False)
    style_axes(ax, grid='x')
    ax.tick_params(axis='y', length=0)
    handles = [Line2D([], [], marker='o', ls='', color=COLORS['Ours'], ms=5, label='Ours'),
               Line2D([], [], marker='s', ls='', color=COLORS['STEP'], ms=3.8, label='STEP'),
               Line2D([], [], marker='o', ls='', mfc='white', mec=COLORS['DDIM'], ms=4.2, label='DDIM'),
               Line2D([], [], marker='D', ls='', color=COLORS['CP'], ms=3.8, label='Consistency Policy')]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=4,
              handletextpad=0.15, columnspacing=1.0, borderaxespad=0.2)
    save(fig, os.path.join(OUT, 'fig_one_step'))


# ================================================================ Fig. 4: start level
def fig_t0():
    from style import BLUE_FILL, GOLD, GOLD_LIGHT, GOLD_DARK, EDGE, RED
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 2.9), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.75})

    # (a) one step: t0 = 90 vs 95, bars in STEP's style, plus a hatched average
    one = []
    for label, task, obs in SETTINGS:
        v90, v95 = get(task, obs, 'Ours', 1, 90), get(task, obs, 'Ours', 1, 95)
        if v90 and v95:
            one.append(({'Push-T (img)': 'Push-T\n(img)', 'Transport': 'Trans-\nport', 'ToolHang': 'Tool-\nHang'}.get(label, label), v90[0], v95[0]))
    one.sort(key=lambda r: -r[1])
    w, gap = 0.36, 0.04
    xs = np.arange(len(one) + 1, dtype=float); xs[-1] += 0.35
    for i, (name, v90, v95) in enumerate(one + [('avg.', np.mean([r[1] for r in one]), np.mean([r[2] for r in one]))]):
        avg = name == 'avg.'
        b1 = a1.bar(xs[i] - w / 2 - gap / 2, v95, w, color=BLUE_FILL, ec=EDGE, lw=0.6,
                    hatch='////' if avg else None)
        b2 = a1.bar(xs[i] + w / 2 + gap / 2, v90, w, color=GOLD, ec=EDGE, lw=0.6, hatch='////' if avg else None)
        a1.text(xs[i] - w / 2 - gap / 2, v95 + 0.02, f'{v95:.2f}'.lstrip('0') if v95 > 0 else '0',
                ha='center', va='bottom', fontsize=6.4)
        a1.text(xs[i] + w / 2 + gap / 2, v90 + 0.02, f'{v90:.2f}'.lstrip('0'), ha='center', va='bottom', fontsize=6.4)
        if avg:
            drop = 1 - v95 / v90
            xa = xs[i] + w + gap / 2 + 0.12
            a1.annotate('', xy=(xa, v95 + 0.01), xytext=(xa, v90),
                        arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.0, mutation_scale=7))
            a1.text(xa + 0.06, (v90 + v95) / 2, f'$-${drop:.0%}', color=RED, fontsize=7.6,
                    fontweight='bold', style='italic', ha='left', va='center')
    a1.axvline(xs[-1] - 0.67, color=INK, lw=0.6, ls=(0, (3, 2)))
    a1.set_xticks(xs); a1.set_xticklabels([r[0] for r in one] + ['avg.'], fontsize=6.8)
    for t in a1.get_xticklabels()[-1:]:
        t.set_style('italic')
    a1.tick_params(axis='x', length=0)
    a1.set_xlim(-0.6, xs[-1] + 1.0); a1.set_ylim(0, 1.0); a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a1.set_yticklabels(['0', '.25', '.50', '.75', '1'])
    a1.set_ylabel('success rate')
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=GOLD, ec=EDGE, lw=0.6),
                       plt.Rectangle((0, 0), 1, 1, fc=BLUE_FILL, ec=EDGE, lw=0.6)],
              labels=['$t_0=90$', '$t_0=95$'], loc='lower center', bbox_to_anchor=(0.5, 0.96), ncol=2,
              columnspacing=1.2, handletextpad=0.4)
    a1.text(-0.2, 1.17, '(a) one step', transform=a1.transAxes, fontsize=8, fontweight='bold', va='bottom')

    # (b) Push-T (state): success vs t0 for n = 1, 2, 4, +-1 std as error bars
    q = DF[(DF.task == 'Push-T') & (DF.obs == 'lowdim') & (DF.method == 'Ours')]
    shade = {1: '#FFD966', 2: '#FFA500', 4: GOLD_DARK}
    mk = {1: 'o', 2: 's', 4: '^'}
    for n in (1, 2, 4):
        g = q[q.nfe == n].sort_values('t0')
        x, y, s = g.t0.to_numpy(float), g['mean'].to_numpy(), g['std'].to_numpy()
        a2.errorbar(x, y, yerr=s, color=shade[n], lw=1.4, marker=mk[n], ms=4.0, mec=EDGE, mew=0.5,
                    elinewidth=0.7, ecolor=shade[n], label=f'$n={n}$', zorder=3 + n)
        b = int(np.argmax(y))
        a2.plot(x[b], y[b], marker='*', ms=8, color=RED, mec=EDGE, mew=0.4, zorder=10, ls='none')
    # offline prediction of t0* (Sec. IV-C): measured solver term + start mismatch with m from data
    pred = {1: 85, 2: 95, 4: 95}
    for n, t in pred.items():
        a2.plot(t + {1: 0, 2: -0.9, 4: 0.9}[n], 0.97, marker='v', ms=5, color=shade[n], mec=EDGE, mew=0.5,
                ls='none', zorder=8, clip_on=False)
    a2.plot([], [], marker='*', ms=7, color=RED, mec=EDGE, mew=0.4, ls='none', label='best $t_0$')
    a2.plot([], [], marker='v', ms=5, color='white', mec=EDGE, mew=0.5, ls='none', label='predicted')
    a2.set_xlim(47, 101.5); a2.set_ylim(0.0, 1.0); a2.set_xticks([50, 60, 70, 80, 90, 99])
    a2.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); a2.set_yticklabels(['0', '.25', '.50', '.75', '1'])
    a2.set_xlabel('start level $t_0$', labelpad=1); a2.set_ylabel('success rate')
    a2.legend(loc='lower center', bbox_to_anchor=(0.58, 0.02), ncol=3, columnspacing=0.8, handletextpad=0.2, fontsize=6.8, handlelength=1.6)
    a2.text(-0.2, 1.17, '(b) Push-T, $n$ steps', transform=a2.transAxes, fontsize=8, fontweight='bold',
            va='bottom')
    save(fig, os.path.join(OUT, 'fig_t0'))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_overview(); fig_t0()          # in the paper
    if '--all' in __import__('sys').argv:
        fig_nfe(); fig_one_step()     # spare figures (not in the current draft)
    print('figures ->', os.path.relpath(OUT))
