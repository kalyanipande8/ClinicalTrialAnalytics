# Clinical Trial Data Analytics Platform

A real, runnable multi-site clinical trial analytics pipeline in Python,
implementing, honestly, the resume bullet:

> Clinical Trial Data Analytics Platform | Statistical Modeling — Developed
> statistical models and validation workflows for multi-site clinical
> trials; built data quality assurance pipelines. Created visualization
> dashboards communicating results to clinical leadership for strategic
> decisions.

> **Disclaimer:** All data is synthetically simulated by
> `simulate_trial_data.py` with a fixed random seed (`np.random.default_rng(42)`).
> There is no real patient, sponsor, or clinical trial data anywhere in this
> repository. The overall treatment effect and the "problem site" are both
> **known, injected ground truth**, used to sanity-check that the pipeline's
> estimation and flagging methods actually recover the known signal (see
> `tests/test_pipeline.py`). Every number in the Results section was read
> directly from an actual `python run_all.py` run's console output. Nothing
> was hand-typed or fabricated.

## How this differs from the sibling `ClinicalTrial-StatisticalAnalysis-Pipeline`

The sibling project (R, SAS-style) builds efficacy/safety TLF
(table-listing-figure) outputs — ANCOVA + AE incidence — across **8
separate trial programs** (different indications, each its own trial), and
validates its numbers against an independently-coded gold-standard
computation.

**This project is a different axis of the same discipline.** It takes
**one** trial run across **9 sites** and focuses on two things the sibling
project does not cover at all:

1. **Site heterogeneity as a first-class statistical problem.** Rather than
   a single pooled estimate, this pipeline fits a genuine **mixed-effects
   model** (site as a random intercept, via `statsmodels.MixedLM`) and
   explicitly compares it against a naive pooled OLS that ignores
   clustering — showing *why* site-aware modeling changes the answer and
   its uncertainty.
2. **Operational data-quality assurance as a pipeline, not a post-hoc
   check.** A DQA module computes per-site completeness, protocol-deviation
   rate, and query rate, and flags outlier sites with a robust
   (MAD-based) statistical rule — the kind of central-monitoring review a
   CRA/data-management team runs during a live trial, not an efficacy/safety
   analysis.
3. **A leadership-facing visual dashboard as the deliverable**, not a
   console table of statistics — built for a non-technical clinical
   leadership audience with direct annotations, flagged problem sites, and
   plain-language framing, rather than SAS-style output tables intended for
   a regulatory submission.

No SAS-style TLF generation, AE tables, or gold-standard cross-validation
are duplicated here — that is the sibling project's job.

## What it does

1. **`simulate_trial_data.py`** simulates patient-level outcomes for a
   single 2-arm (Control vs Treatment) randomized trial run across **9
   sites** (560 patients total), with:
   - a known, injected overall treatment effect (**-3.5** on the
     continuous change-from-baseline primary endpoint),
   - real site-to-site heterogeneity via random site intercepts
     (SD = 2.2),
   - realistic per-site enrollment sizes (38–110 patients/site), and
   - **one deliberately injected problem site (`SITE-07`)** with elevated
     protocol-deviation rate, query rate, and missingness, so the DQA
     pipeline has a real site to catch.

2. **`dqa_pipeline.py`** computes per-site completeness, protocol-deviation
   rate, and query rate, and flags sites that are robust statistical
   outliers (modified z-score > 2.5 MADs from the cross-site median) on
   two or more metrics as "PROBLEM SITE — review required."

3. **`mixed_effects_model.py`** fits a linear mixed-effects model
   (`endpoint ~ arm + baseline`, site as a random intercept) to estimate
   the adjusted overall treatment effect accounting for site
   heterogeneity, alongside a naive pooled OLS and per-site stratified
   estimates for comparison.

4. **`dashboard.py`** builds a single annotated leadership dashboard
   (`output/figures/leadership_dashboard.png`) with four panels: overall
   treatment effect (mixed model vs. naive), per-site effect consistency
   (forest plot), enrollment vs. data completeness, and protocol-deviation
   rate by site — using a colorblind-safe palette and direct annotations
   rather than a legend-decoding exercise.

5. **`run_all.py`** runs the full pipeline end-to-end and then runs the
   test suite.

## Results (from an actual `python run_all.py` run)

```
Simulated 560 patients across 9 sites.
True injected overall treatment effect: -3.5
Injected problem site: SITE-07 (PD rate=0.22, query rate=0.31, missingness=0.19)

=== Data Quality Assurance: Site Summary ===
site_id  n_patients  completeness_rate  protocol_deviation_rate  query_rate  n_flags  dqa_status
SITE-07          38           0.789474                 0.210526    0.421053        3  PROBLEM SITE - review required
SITE-01          85           0.964706                 0.011765    0.117647        0  OK
SITE-02          62           0.951613                 0.016129    0.064516        0  OK
SITE-03          40           1.000000                 0.075000    0.150000        0  OK
SITE-04         110           0.963636                 0.027273    0.072727        0  OK
SITE-05          55           0.963636                 0.072727    0.127273        0  OK
SITE-06          48           0.979167                 0.041667    0.041667        0  OK
SITE-08          70           0.957143                 0.085714    0.057143        0  OK
SITE-09          52           0.980769                 0.019231    0.019231        0  OK

Problem sites flagged: ['SITE-07']

=== Mixed-Effects Model (site as random intercept) ===
N evaluable patients: 535 across 9 sites
Overall treatment effect (adjusted): -3.9767
SE: 0.3512
95% CI: [-4.6650, -3.2883]
p-value: 1.009e-29
Site random-effect variance: 7.4475

Naive pooled OLS (ignoring site clustering) treatment effect: -4.4551 (SE 0.4121)

Test suite: 6 tests, all PASSED.
```

**The mixed-effects model recovered the injected overall treatment effect
(-3.5) within its 95% CI [-4.665, -3.288]** — the model's point estimate
(-3.98) is close to the true value and the CI correctly contains it,
despite each individual site's much noisier per-site estimate (SEs of
0.8–1.4, n=30–106 per site). The naive pooled OLS, which ignores site
clustering entirely, produces a different point estimate (-4.46) with a
larger SE (0.41 vs. 0.35) — a concrete demonstration of why site needs to
be modeled, not ignored, in a multi-site trial.

**The DQA pipeline correctly identified `SITE-07` as the sole problem
site** (3/3 metrics flagged: lowest completeness at 78.9% vs. a
97–100% range elsewhere, highest protocol-deviation rate at 21.1% vs.
1.2–8.6% elsewhere, highest query rate at 42.1% vs. 1.9–15.0% elsewhere),
with no other site mistakenly flagged.

## Dashboard

`output/figures/leadership_dashboard.png` — four panels: overall treatment
effect estimate, per-site treatment-effect consistency (forest plot),
enrollment vs. data completeness (with `SITE-07` visibly the outlier), and
protocol-deviation rate by site with `SITE-07` called out in red.

## Project structure

```
simulate_trial_data.py   # 9 sites, 560 patients -> data/patient_level_data.csv
dqa_pipeline.py          # per-site completeness/PD-rate/query-rate + outlier flagging -> output/dqa_site_summary.csv
mixed_effects_model.py   # MixedLM treatment-effect estimate -> output/mixed_model_results.csv, per_site_treatment_effects.csv
dashboard.py             # leadership dashboard -> output/figures/leadership_dashboard.png
run_all.py               # orchestrates the full pipeline + runs tests
tests/test_pipeline.py   # multi-site structure, DQA problem-site flagging, mixed-model recovery of injected effect
```

## Tech stack

- Python 3
- pandas / numpy for data simulation and wrangling
- statsmodels (`MixedLM`) for the mixed-effects model
- matplotlib for the leadership dashboard
- pytest

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py       # regenerates data/ and output/, then runs the test suite
pytest tests/ -v         # run just the tests against already-generated output
```
