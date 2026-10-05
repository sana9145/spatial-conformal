# Dissimilarity-Adaptive Conformal Prediction for Spatial Models under Uneven Monitoring

Code, raw experiment outputs and verification scripts for the paper
*Dissimilarity-Adaptive Conformal Prediction: Improving Per-Region Coverage for
Spatial Models under Uneven Monitoring* (Sana Mustafa).

## What the study does

Environmental maps are usually built from monitoring networks that cluster where
access is easy. Split conformal prediction is valid only when calibration sites and
prediction locations are exchangeable, so its intervals are too narrow in the
under-monitored parts of the map. The study asks whether the dissimilarity index
(DI) of the area-of-applicability framework (Meyer & Pebesma, 2021), which spatial
ML practitioners already compute, can be used to condition conformal calibration.

Two DI-based procedures are compared with standard and principled alternatives:

- **DI-normalized**: residual score divided by `kappa + DI`.
- **DI-CQR**: conformalized quantile regression with a Mondrian taxonomy over DI
  quantile bins (monotone corrections, small-bin fallback, top-bin clipping).
- **Comparators**: split, normalized, CQR, spatial Mondrian, localized conformal
  prediction (Guan, 2023) implemented exactly with the adjusted level and a fixed
  kernel (in feature space, on CQR scores, and in geographic space), weighted
  conformal prediction with oracle weights from the exact inclusion probabilities of
  the monitoring design and with estimated weights, and a width-matched control.

## Experiments

1. **Simulation**: 56 x 56 landscape with ground truth everywhere; 500 monitored
   sites; scored on the 2,636 unmonitored cells. Six regimes x 30 seeds: covariate
   accessibility at bias none / mild / moderate / severe, and a hidden road-access
   field (not a model covariate) at moderate / severe bias.
2. **Robustness**: moderate and severe regimes with a gradient-boosting mean model.
3. **Meuse**: 155 floodplain soil samples (log zinc), imposed clustered monitoring,
   30 repeats.
4. **LUCAS 2015**: 21,687 EU topsoil sites (log organic carbon from 23 climate and
   terrain covariates), imposed clustered monitoring, 20 repeats.

Results are in `results/`; the manuscript's tables and every number quoted in its
text are generated from these files by `code/make_tables.py`.

## Repository layout

```
code/      aoa.py          dissimilarity index (fitting-set only)
           landscape.py    simulator, monitoring designs, inclusion probabilities
           conformal.py    all interval methods, incl. exact localized conformal
           metrics.py      coverage, width, interval score, per-region summaries
           run_sim.py      simulation benchmark (resumable)
           aggregate.py    simulation summaries, paired tests, Holm adjustment
           run_robust.py   gradient-boosting mean-model robustness check
           run_meuse.py    Meuse experiment
           prep_lucas.py   builds the LUCAS analysis table from the ESDAC files
           run_lucas.py    LUCAS experiment
           make_tables.py  LaTeX tables and paper/numbers.tex from results
           figures.py      all figures
           verify.py       re-derives every quoted number and checks every claim
results/   raw per-run outputs (JSONL) and summaries (CSV/JSON)
figures/   figures used in the paper
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

The LUCAS 2015 topsoil data and ancillary covariates are distributed by the
European Soil Data Centre (ESDAC) under the JRC licence and are not included here.
Download the topsoil CSV, the point shapefile and `LUCAS2015_AncillaryData_20201007.csv`
from ESDAC and pass their folder to `prep_lucas.py`. The derived per-repeat outputs
(`results/lucas_raw.jsonl`, `lucas_summary.csv`, `lucas_sig.json`, `lucas_diag.json`)
are included, so the reported LUCAS numbers can be checked without the raw data.

## Scope

The real datasets use imposed (semi-synthetic) clustered monitoring rather than a
naturally biased network. The localized and weighted conformal baselines are our
implementations of the published methods, not the authors' code. DI-based
calibration improves per-region coverage but does not provide a per-region coverage
guarantee; see the paper's limitations section.

## License

MIT; see `LICENSE`.
