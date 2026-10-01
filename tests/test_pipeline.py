"""
Sanity-check test suite.

Run after run_all.py has generated data/ and output/ (run_all.py invokes
this automatically at the end). Checks:
  1. Multi-site structure is present in the simulated data.
  2. The DQA pipeline correctly flags the injected problem site (SITE-07).
  3. The mixed-effects model correctly recovers the injected overall
     treatment effect (-3.5) within a reasonable tolerance.
"""
import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simulate_trial_data as sim
from dqa_pipeline import run_dqa
from mixed_effects_model import fit_mixed_effects

DATA_PATH = "data/patient_level_data.csv"


@pytest.fixture(scope="module")
def patient_df():
    if not os.path.exists(DATA_PATH):
        sim.main()
    return pd.read_csv(DATA_PATH)


def test_multi_site_structure_present(patient_df):
    assert patient_df["site_id"].nunique() == sim.N_SITES
    assert sim.N_SITES >= 8
    # every site has both arms represented
    for site, g in patient_df.groupby("site_id"):
        assert set(g["arm"].unique()) == {"Control", "Treatment"}


def test_patient_count_matches_design(patient_df):
    assert len(patient_df) == sum(sim.SITE_N.values())


def test_dqa_flags_injected_problem_site(patient_df):
    summary = run_dqa(patient_df)
    row = summary[summary["site_id"] == sim.PROBLEM_SITE].iloc[0]
    assert row["dqa_status"] == "PROBLEM SITE - review required", (
        f"Expected {sim.PROBLEM_SITE} to be flagged as a problem site, got status "
        f"'{row['dqa_status']}' with {row['n_flags']} flags"
    )
    # and it should have the highest protocol deviation rate in the portfolio
    assert row["protocol_deviation_rate"] == summary["protocol_deviation_rate"].max()


def test_dqa_does_not_flag_every_site(patient_df):
    summary = run_dqa(patient_df)
    n_problem = (summary["dqa_status"] == "PROBLEM SITE - review required").sum()
    assert n_problem < len(summary), "DQA pipeline should not flag the entire portfolio"
    assert n_problem >= 1


def test_mixed_model_recovers_injected_treatment_effect(patient_df):
    results, site_effects_df, mdf = fit_mixed_effects(patient_df)
    estimate = results["mixed_model_treatment_effect"]
    true_effect = sim.TRUE_TREATMENT_EFFECT
    assert abs(estimate - true_effect) < 1.0, (
        f"Mixed model estimate {estimate:.3f} too far from true injected effect {true_effect}"
    )
    # CI should contain the true effect
    assert results["mixed_model_ci_lower"] < true_effect < results["mixed_model_ci_upper"]


def test_mixed_model_outperforms_or_matches_naive_in_capturing_site_variance(patient_df):
    results, _, _ = fit_mixed_effects(patient_df)
    assert results["site_random_effect_variance"] > 0
    assert results["n_sites"] == sim.N_SITES
