# Dissimilarity-adaptive conformal prediction for spatial models under uneven monitoring

Code, raw experiment outputs and verification scripts for the manuscript
*Dissimilarity-adaptive conformal prediction: improving per-region coverage for spatial
models under uneven monitoring* by Sana Mustafa, prepared for Geoscientific Model
Development.

Archived on Zenodo: https://doi.org/10.5281/zenodo.23181222 (resolves to the latest
version; each release also has its own version DOI).

## What the study does

Environmental maps are usually built from monitoring networks that cluster where
access is easy. Split conformal prediction is valid only when calibration sites and
prediction locations are exchangeable, so its intervals are too narrow in the
under-monitored parts of the map. The study asks whether the dissimilarity index
(DI) of the area-of-applicability framework (Meyer and Pebesma, 2021), which spatial
machine-learning practitioners already compute, can be used to condition conformal
calibration.

Two DI-based procedures are compared with standard and principled alternatives:

- **DI-normalized**: residual score divided by `kappa + DI`.
- **DI-CQR**: conformalized quantile regression with a Mondrian taxonomy over DI
  quantile bins (monotone corrections, small-bin fallback, top-bin clipping).
- **Comparators**: split, normalized, CQR, spatial Mondrian, localized conformal
  prediction (Guan, 2023) with its adjusted level and a fixed kernel (in feature
  space, on CQR scores, and in geographic space), weighted conformal prediction with
  oracle weights from design-based inclusion probabilities estimated by Monte Carlo
  simulation (20,000 replicate designs) and with estimated weights, and a
  width-matched control.

## Experiments

1. **Simulation**: 56 x 56 landscape with ground truth everywhere; 500 monitored
   sites; scored on the 2,636 unmonitored cells. Six regimes x 30 seeds: covariate
   accessibility at bias none / mild / moderate / severe, and a hidden road-access
   field (not a model covariate) at moderate / severe bias. Includes the DI-bin
   ablation, kappa and region-partition sensitivity, coverage by DI decile and the
   sensitivity to the cap used for unbounded intervals.
2. **Robustness**: moderate and severe regimes with a gradient-boosting mean model.
3. **Meuse**: 155 floodplain soil samples (log zinc), imposed clustered monitoring,
   30 repeats.
4. **LUCAS 2015**: 21,687 EU topsoil sites (log organic carbon from 23 climate and
   terrain covariates), imposed clustered monitoring, 20 repeats.

Results are in `results/`. The manuscript's tables and every number quoted in its
text are generated from these files by `code/make_tables.py`.

## Repository layout

```
code/      aoa.py          dissimilarity index (fitting-set only)
           landscape.py    simulator, monitoring designs, inclusion probabilities
           conformal.py    all interval methods, incl. localized conformal (Guan, 2023)
           metrics.py      coverage, width, interval score, per-region summaries
           run_sim.py      simulation benchmark (resumable)
           aggregate.py    simulation summaries, paired tests, Holm adjustment
           run_robust.py   gradient-boosting mean-model robustness check
           run_meuse.py    Meuse experiment
           prep_lucas.py   builds the LUCAS analysis table from the ESDAC files
           run_lucas.py    LUCAS experiment
           make_tables.py  LaTeX tables and paper/numbers.tex from results
           figures.py      all figures
           verify.py       re-derives the quoted numbers and checks the stated comparisons
results/   raw per-run outputs (JSONL) and summaries (CSV/JSON)
figures/   figures of the manuscript: fig01-fig04 (main text) and figB1-figB4
           (Appendix B), each as 300 dpi PNG and vector PDF
paper/     generated LaTeX tables and numbers.tex (written by make_tables.py;
           the manuscript itself is not part of this repository)
```

## Reproducing

Python 3.10 with the packages pinned in `requirements.txt`.

```bash
cd code
python run_sim.py && python aggregate.py
python run_robust.py && python run_robust.py agg
python run_meuse.py && python run_meuse.py agg
python prep_lucas.py /path/to/ESDAC/files     # needs the LUCAS 2015 download
python run_lucas.py && python run_lucas.py agg
python make_tables.py
python figures.py
python verify.py
```

All runners are resumable: they append one line per run and skip runs already in
the output file. Setting the environment variable `BUDGET` (seconds) makes a runner
stop early so it can be resumed later.

`verify.py` regenerates `paper/numbers.tex` and checks that it is unchanged,
re-derives the quoted LUCAS width ratios from the raw per-repeat output, re-checks the
design invariants (held-out scoring, width matching, Holm columns, localized conformal
against a brute-force implementation of Guan's definitions) and asserts the
qualitative comparisons stated in the manuscript (orderings, directions of differences
and significance calls). In this repository 143 checks run; one further check, that
every number used in the manuscript source exists, runs only where the manuscript
source is present. Without the experiments, running
`make_tables.py`, `figures.py` and `verify.py` on the included results reproduces every
number, table and figure of the manuscript.

The LUCAS 2015 topsoil data and ancillary covariates are distributed by the
European Soil Data Centre (ESDAC) under the JRC licence and are not included here.
Download the topsoil CSV, the point shapefile and `LUCAS2015_AncillaryData_20201007.csv`
from ESDAC and pass their folder to `prep_lucas.py`. The derived per-repeat outputs
(`results/lucas_raw.jsonl`, `lucas_summary.csv`, `lucas_sig.json`, `lucas_diag.json`,
`lucas_cap_sens.csv`, `lucas_di_curve.csv`) are included, so the reported LUCAS numbers
can be checked without the raw data.

## Scope

The real datasets use imposed (semi-synthetic) clustered monitoring rather than a
naturally biased network. The localized and weighted conformal baselines are our
implementations of the published methods, not the authors' code. DI-based
calibration improves per-region coverage but does not provide a per-region coverage
guarantee; see the manuscript's limitations section.

## License

MIT; see `LICENSE`.
