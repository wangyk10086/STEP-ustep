# Figures and tables for the paper

All figures and tables in the paper are generated from one cleaned data file. To rebuild them:

```bash
pip install matplotlib pandas numpy        # seaborn is not required
python clean_data.py    # data/raw_results.csv + data/pusht_raw/  ->  data/results_clean.csv
python figures.py       # ->  ../figs/fig_{overview,nfe,one_step,t0}.{pdf,png}
python tables.py        # ->  ../tables/tab_{main,main_std,t0}.tex
```

When new results come in, append rows to `data/raw_results.csv` and rerun the three commands. No number in the paper is typed by hand. The one exception is the STEP paper's Push-T values, which are listed in `REPORTED` in `clean_data.py`.

## Files

| File | Contents |
| --- | --- |
| `style.py` | Global style: sizes, fonts, colours, markers, save helpers |
| `clean_data.py` | Data cleaning and seed pooling |
| `figures.py` | The four figures |
| `tables.py` | LaTeX booktabs tables |
| `data/results_clean.csv` | Tidy table: `task, obs, method, nfe, t0, mean, std, n_seeds, source` |

## What the cleaning step does

1. **Method names.** Normalised to `fixed_start` → `Ours`, `ConsistencyPolicy` → `CP`, and `ustep_delta*` → `U-STEP(delta=…)`.
2. **Seed counts.** Parsed out of the free-text `seeds` column.
3. **Push-T (state) std.** 25 values missing from the CSV are filled from the raw evaluation JSON files. Each mean is checked against the CSV (tolerance 2e-3).
4. **Pooling.** Runs of the same configuration on disjoint seed sets (3 seeds + 2 more) are merged into a seed-weighted mean and the pooled std over all N seeds, sqrt(Σ nᵢ(sᵢ² + (mᵢ − M)²)/N).
5. **Reported numbers.** Values taken from the STEP paper are stored with `source = 'STEP paper'`. They get a † in the tables, and figures use them in preference to our own runs.
6. **Still missing std.** 12 rows have no std. Two of them are table rows: DDIM and STEP at 2 NFE on Can (img), whose per-seed results were not exported. The others are U-STEP and predictor-only runs that are not used in the paper.

## Colour and layout conventions

**Colour**
- **Palette.** Colours come from the Okabe–Ito set, which is colour-blind safe:
  - Ours: blue `#0072B2`;
  - STEP: vermillion `#D55E00`;
  - Consistency Policy: bluish green `#009E73`;
  - DDIM: neutral grey `#8C8C8C`, always with an open marker.
- **Validation.** The three hues pass a CVD separation check (deutan, protan, tritan; ΔE ≥ 8) and have ≥ 3:1 contrast on white.
- **Never colour alone.** Each method also has a marker shape (● ■ ◆ ○), and Consistency Policy is dashed. The figures stay readable in grayscale print.
- **Sequential encodings use one hue.** The step budget n = 1, 2, 4 is drawn as light → dark blue (`OURS_RAMP`), not as different hues.

**Layout**
- **Size and fonts.** Width is 3.5 in (one column) or 7.16 in (full width). Fonts are 6.5–8 pt STIX, which matches Times in the IEEE body. Figures are inserted at `width=\columnwidth` or `\textwidth` with no scaling, so the font sizes stay correct.
- **Axes.** Only the left and bottom spines are kept; grid lines are light grey on the value axis only. All panels sharing a quantity use the same scale; there are no dual y-axes. Success is drawn on [0, 1] in every panel so tasks can be compared.
- **Uncertainty.** Shaded bands and error bars are ±1 std over evaluation seeds, as stated in the captions.
- **Direct labels instead of legends where possible.** n = 1, 2, 4 and task names are labelled on the curves; one shared legend is placed above the small multiples.
- **Labelling numbers.** Only the headline values (one-step success) are printed on the plot; everything else is in the table.
- **Tables.**
  - booktabs rules only, with no vertical lines;
  - two decimals with the leading zero dropped (`.94`);
  - best per budget shaded, with ties after rounding all shaded;
  - an entry is shaded only when at least two methods were run in that column;
  - `--` means not run.
- **Output formats.** Vector PDF with embedded TrueType fonts (`pdf.fonttype = 42`), which IEEE PDF eXpress requires to pass. A 600 dpi PNG is also written for slides.
