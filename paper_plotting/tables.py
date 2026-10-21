"""LaTeX tables (booktabs) generated from data/results_clean.csv.

    python tables.py   ->  ../tables/tab_main.tex      main comparison (mean; best per budget shaded)
                           ../tables/tab_main_std.tex  same, as mean +- std over seeds
                           ../tables/tab_t0.tex        start-level sweep

Conventions
- Success rates are given to two decimals with the leading zero dropped (.94), the
  standard compact form for two-column papers.
- Best per NFE budget among DDIM / STEP / Ours is shaded (\\cellcolor{best}). Ties after
  rounding are all shaded, so the shading never overstates a 0.001 difference.
- Numbers taken from another paper carry a dagger and are explained in the caption.
- "--" means the configuration was not run.
The preamble needs: booktabs, multirow, xcolor[table], and \\definecolor{best}{...}.
"""
import os

import pandas as pd

from figures import SETTINGS, get

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'tables')
DF = pd.read_csv(os.path.join(HERE, 'data', 'results_clean.csv'))
METHODS = ('DDIM', 'STEP', 'Ours')
NFES = (1, 2, 4)
SHORT = {'Push-T (img)': 'Push-T img', 'Can (img)': 'Can img', 'Square (img)': 'Square img'}


def f2(x):
    s = f'{x:.2f}'
    return s[1:] if s.startswith('0') else s


def is_reported(task, obs, m, n):
    q = DF[(DF.task == task) & (DF.obs == obs) & (DF.method == m) & (DF.nfe == n)]
    return (q.source == 'STEP paper').any()


def main_rows(with_std=False):
    lines = []
    for label, task, obs in SETTINGS:
        cells = []
        for n in NFES:
            vals = {m: get(task, obs, m, n) for m in METHODS}
            have = [round(v[0], 2) for v in vals.values() if v is not None]
            best = max(have) if have else None
            for m in METHODS:
                v = vals[m]
                if v is None:
                    cells.append('--'); continue
                txt = f2(v[0])
                if is_reported(task, obs, m, n):
                    txt += '$^\\dagger$'
                elif with_std and v[1] is not None:
                    txt += '{\\scriptsize$\\,\\pm$' + f2(v[1]) + '}'
                if round(v[0], 2) == best and best > 0 and len(have) > 1:
                    txt = '\\cellcolor{best}' + txt
                cells.append(txt)
        lines.append(f'{SHORT.get(label, label):<12} & ' + ' & '.join(cells) + '\\\\')
    return lines


def tab_main(with_std=False):
    cap = ('Closed-loop success at 1, 2 and 4 network evaluations' +
           (' (mean$\\pm$std over seeds)' if with_std else '') +
           '. \\ours{} uses one global start level per budget ($t_0{=}90/85/95$). ``img\'\': image '
           'observations. Shaded: best per budget. $^\\dagger$Reported by the STEP paper~\\cite{step2026} with '
           'its own checkpoints; all other entries are our runs with the public DP checkpoints.')
    body = '\n'.join(main_rows(with_std))
    env = 'table*' if with_std else 'table'
    return f"""\\begin{{{env}}}[t]
\\centering
\\caption{{{cap}}}
\\label{{tab:main}}
\\setlength{{\\tabcolsep}}{{{'4.5' if with_std else '2.6'}pt}}
\\renewcommand{{\\arraystretch}}{{1.08}}
\\footnotesize
\\begin{{tabular}}{{l ccc ccc ccc}}
\\toprule
& \\multicolumn{{3}}{{c}}{{1 NFE}} & \\multicolumn{{3}}{{c}}{{2 NFE}} & \\multicolumn{{3}}{{c}}{{4 NFE}}\\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}\\cmidrule(lr){{8-10}}
Task & DDIM & STEP & Ours & DDIM & STEP & Ours & DDIM & STEP & Ours\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{{env}}}
"""


def tab_t0():
    q = DF[DF.method == 'Ours'].dropna(subset=['t0'])
    t0s = sorted(q.t0.astype(int).unique())
    lines = []
    for label, task, obs in SETTINGS:
        for n in NFES:
            g = q[(q.task == task) & (q.obs == obs) & (q.nfe == n)]
            if len(g) < 2:
                continue
            vals = {int(r.t0): r['mean'] for _, r in g.iterrows()}
            best = round(max(vals.values()), 2)
            cells = []
            for t in t0s:
                if t not in vals:
                    cells.append('--')
                else:
                    c = f2(vals[t])
                    cells.append(('\\cellcolor{best}' + c) if round(vals[t], 2) == best else c)
            lines.append(f'{SHORT.get(label, label):<12} & {n} & ' + ' & '.join(cells) + '\\\\')
    head = ' & '.join(str(t) for t in t0s)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Success of \\ours{{}} versus start level $t_0$ and step budget $n$. Shaded: best $t_0$ in each row.}}
\\label{{tab:t0}}
\\setlength{{\\tabcolsep}}{{3pt}}
\\footnotesize
\\begin{{tabular}}{{l c {'c' * len(t0s)}}}
\\toprule
& & \\multicolumn{{{len(t0s)}}}{{c}}{{start level $t_0$}}\\\\
\\cmidrule(lr){{3-{2 + len(t0s)}}}
Task & $n$ & {head}\\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, tex in (('tab_main', tab_main()), ('tab_main_std', tab_main(True)), ('tab_t0', tab_t0())):
        open(os.path.join(OUT, name + '.tex'), 'w').write(tex)
    print('tables ->', os.path.relpath(OUT))
