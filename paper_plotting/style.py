"""Shared figure style, matched to the STEP paper (arXiv 2602.08245).

STEP's figures are drawn in an Office-like style. The conventions we copy from them:
- Times New Roman text in black; group labels in italics.
- Flow-chart boxes: rounded corners, light-blue fill, dark outline, black arrows.
  The method's own component is highlighted, and STEP's flagship rows are bold.
- Bars and markers: light blue (#BDD7EE) for baselines and gold (#FFC000) for the proposed
  method, both with a thin black edge. Values are printed on the bars.
- Emphasis (e.g. an average drop) uses a red arrow with bold-italic red text; averages use hatching.
- The legend sits on top with square swatches. Axes are black with no grid, and dashed
  vertical separators divide groups.

Sizes follow the IEEE template: one column 3.5 in, full width 7.16 in.
Output is vector PDF with embedded TrueType fonts (Type 42), plus a 600 dpi PNG.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL, PAGE = 3.5, 7.16

# STEP-paper palette (Office tones). Gold vs. blue stays distinguishable under all common CVD types;
# every series also has its own marker / hatch, so nothing relies on colour alone.
BLUE_FILL = '#BDD7EE'   # STEP's "policy" boxes and baseline bars
BLUE_LIGHT = '#DEEBF7'  # STEP's "predictor" box
BLUE_DARK = '#2E75B6'
GOLD = '#FFC000'        # proposed method
GOLD_LIGHT = '#FFE699'
GOLD_DARK = '#BF8F00'
RED = '#C00000'         # emphasis
GRAY = '#D9D9D9'
EDGE = '#1F1F1F'
INK, MUTED = '#000000', '#595959'

COLORS = {'Ours': GOLD, 'STEP': BLUE_DARK, 'DDIM': '#A6A6A6', 'CP': '#70AD47'}
MARKERS = {'Ours': 'o', 'STEP': 's', 'CP': 'D', 'DDIM': '^'}
LABELS = {'Ours': 'Ours', 'STEP': 'STEP', 'CP': 'Consistency Policy', 'DDIM': 'DDIM'}
OURS_RAMP = [GOLD_LIGHT, GOLD, GOLD_DARK]   # n = 1, 2, 4
GRID = '#EDEDED'


def method_style(m, lw=None, ms=None):
    s = dict(color=COLORS[m], marker=MARKERS[m], lw=lw or (1.5 if m == 'Ours' else 1.0),
             ms=ms or (4.4 if m == 'Ours' else 3.6), mec=EDGE, mew=0.5,
             zorder=4 if m == 'Ours' else 3)
    if m == 'CP':
        s.update(ls=(0, (2.4, 1.2)))
    return s


plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'TeX Gyre Termes'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.6, 'axes.edgecolor': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.handlelength': 1.2, 'legend.handleheight': 0.8,
    'hatch.linewidth': 0.6,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})


def style_axes(ax, grid=None):
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def panel_title(ax, text):
    ax.set_title(text, loc='left', fontsize=8, color=INK, pad=3)


def save(fig, path_no_ext):
    fig.savefig(f'{path_no_ext}.pdf')
    fig.savefig(f'{path_no_ext}.png')
    plt.close(fig)
