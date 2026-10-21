"""Clean and consolidate all closed-loop results into one tidy table.

Input
  data/raw_results.csv        one row per evaluation run, exported from the pod.
                              The `seeds` column is free text, e.g. "3 (100000-100200)".
  data/pusht_raw/*/eval_results.json
                              raw Push-T (state) evaluations. They fill the per-seed
                              standard deviations that are missing from the CSV.

Output
  data/results_clean.csv with columns
      task, obs, method, nfe, t0, mean, std, n_seeds, source

Cleaning steps
  1. Normalise method names (fixed_start -> Ours, ConsistencyPolicy -> CP, ustep_* -> U-STEP).
  2. Parse the seed count out of the free-text `seeds` column.
  3. Fill missing Push-T (state) std values from the raw JSON files.
  4. Pool runs of the same configuration that used disjoint seed sets (for example
     3 seeds + 2 extra seeds). This gives a seed-weighted mean and the pooled
     population std,
         sqrt( sum_i n_i (s_i^2 + (m_i - M)^2) / N ),
     which equals the std over all N seeds.
  5. Add the numbers that the STEP paper reports for Push-T (state) as separate rows
     with source = 'STEP paper'. Our own STEP runs are kept with source = 'ours'.

Run:  python clean_data.py
"""
import glob
import json
import math
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'data', 'raw_results.csv')
PUSHT = os.path.join(HERE, 'data', 'pusht_raw')
OUT = os.path.join(HERE, 'data', 'results_clean.csv')

REPORTED = [  # STEP paper (arXiv 2602.08245), Push-T state, its own checkpoints
    dict(task='Push-T', obs='lowdim', method='STEP', nfe=2, t0=None, mean=0.49, std=None, n_seeds=5),
    dict(task='Push-T', obs='lowdim', method='STEP', nfe=4, t0=None, mean=0.91, std=None, n_seeds=5),
]


def norm_method(m):
    if m == 'fixed_start':
        return 'Ours'
    if m == 'ConsistencyPolicy':
        return 'CP'
    if m.startswith('ustep_delta'):
        return 'U-STEP(delta=' + m[len('ustep_delta'):] + ')'
    if m == 'predictor_only':
        return 'Predictor only'
    return m


def pusht_std_lookup():
    """(method, nfe, t0) -> (mean, std), read from the raw Push-T JSON files."""
    out = {}
    for f in glob.glob(os.path.join(PUSHT, '*', 'eval_results.json')):
        name = os.path.basename(os.path.dirname(f))
        d = json.load(open(f))
        if name.startswith('fixed_t'):
            mt = re.match(r'fixed_t(\d+)(?:_n(\d+))?$', name)
            t0, n = int(mt.group(1)), int(mt.group(2) or 4)
            met = d['ustep_delta1']['metrics']
            out[('Ours', n, t0)] = (float(met['test/mean_score']), float(met['test/mean_score_std']))
        else:
            for k, v in d.items():
                if not k.startswith('steps_'):
                    continue
                n = int(k.split('_')[1])
                for key, meth in (('combined', 'STEP'), ('diffusion_only', 'DDIM')):
                    if key in v:
                        met = v[key]['metrics']
                        out[(meth, n, None)] = (float(met['test/mean_score']), float(met['test/mean_score_std']))
    return out


def pool(g):
    n = g['n_seeds'].astype(float)
    N = n.sum()
    M = (n * g['mean']).sum() / N
    if g['std'].isna().any():
        S = float('nan') if len(g) > 1 or g['std'].isna().all() else g['std'].iloc[0]
    else:
        S = math.sqrt((n * (g['std'] ** 2 + (g['mean'] - M) ** 2)).sum() / N)
    return pd.Series(dict(mean=M, std=S, n_seeds=int(N)))


def main():
    df = pd.read_csv(RAW)
    df = df.rename(columns={'success_mean': 'mean', 'success_std_over_seeds': 'std'})
    df['method'] = df['method'].map(norm_method)
    df['n_seeds'] = df['seeds'].astype(str).str.extract(r'^\s*(\d+)')[0].astype(int)
    df['t0'] = pd.to_numeric(df['t0'], errors='coerce')
    df['nfe'] = pd.to_numeric(df['nfe'])

    # 3. fill Push-T (state) std from raw JSON (and check the means agree)
    lk = pusht_std_lookup()
    n_filled = 0
    for i, r in df[(df.task == 'Push-T') & (df.obs == 'lowdim') & df['std'].isna()].iterrows():
        key = (r.method, int(r.nfe) if float(r.nfe).is_integer() else r.nfe,
               None if pd.isna(r.t0) else int(r.t0))
        if key in lk:
            m, s = lk[key]
            assert abs(m - r['mean']) < 2e-3, (key, m, r['mean'])
            df.at[i, 'std'] = s
            n_filled += 1

    # 4. pool disjoint seed sets of the same configuration
    keys = ['task', 'obs', 'method', 'nfe', 't0']
    df['t0'] = df['t0'].fillna(-1)
    clean = df.groupby(keys, sort=False)[['mean', 'std', 'n_seeds']].apply(pool).reset_index()
    clean['t0'] = clean['t0'].replace(-1, pd.NA)
    clean['source'] = 'ours'

    # 5. reported numbers
    rep = pd.DataFrame(REPORTED)
    rep['source'] = 'STEP paper'
    clean = pd.concat([clean, rep], ignore_index=True)
    clean['n_seeds'] = clean['n_seeds'].astype(int)
    clean = clean.sort_values(['task', 'obs', 'method', 'nfe', 't0'], na_position='first')
    clean.to_csv(OUT, index=False, float_format='%.4f')
    print(f'{len(df)} raw rows -> {len(clean)} clean rows; filled {n_filled} Push-T std values; '
          f'{(clean["std"].isna()).sum()} rows still without std -> {os.path.relpath(OUT, HERE)}')


if __name__ == '__main__':
    main()
