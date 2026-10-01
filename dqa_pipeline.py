"""
Data Quality Assurance (DQA) pipeline.

Computes, per site: data completeness, protocol-deviation rate, and query
rate, then flags sites whose values exceed pre-specified thresholds
(thresholds set relative to the portfolio-wide distribution, i.e. a site is
flagged if it is a statistical outlier relative to its peers, not against an
arbitrary fixed cutoff -- the way a real central-monitoring DQA review would
work).
"""
import numpy as np
import pandas as pd
import os

# Flag a site if a metric is more than this many MADs (median absolute
# deviations, robust to outliers caused by the very site we're trying to
# detect) above the cross-site median.
MAD_THRESHOLD = 2.5


def robust_outlier_flag(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        mad = 1e-9
    modified_z = 0.6745 * (series - median) / mad
    return modified_z > MAD_THRESHOLD


def run_dqa(df: pd.DataFrame) -> pd.DataFrame:
    site_summary = df.groupby("site_id").agg(
        n_patients=("patient_id", "count"),
        n_missing=("endpoint_missing", "sum"),
        n_deviations=("protocol_deviation", "sum"),
        n_queries=("query_flag", "sum"),
    ).reset_index()

    site_summary["completeness_rate"] = 1 - site_summary["n_missing"] / site_summary["n_patients"]
    site_summary["protocol_deviation_rate"] = site_summary["n_deviations"] / site_summary["n_patients"]
    site_summary["query_rate"] = site_summary["n_queries"] / site_summary["n_patients"]

    site_summary["flag_low_completeness"] = robust_outlier_flag(1 - site_summary["completeness_rate"])
    site_summary["flag_high_pd_rate"] = robust_outlier_flag(site_summary["protocol_deviation_rate"])
    site_summary["flag_high_query_rate"] = robust_outlier_flag(site_summary["query_rate"])

    site_summary["n_flags"] = (
        site_summary["flag_low_completeness"].astype(int)
        + site_summary["flag_high_pd_rate"].astype(int)
        + site_summary["flag_high_query_rate"].astype(int)
    )
    site_summary["dqa_status"] = np.where(
        site_summary["n_flags"] >= 2, "PROBLEM SITE - review required",
        np.where(site_summary["n_flags"] == 1, "WATCH", "OK"),
    )

    return site_summary.sort_values("n_flags", ascending=False).reset_index(drop=True)


def main():
    df = pd.read_csv("data/patient_level_data.csv")
    summary = run_dqa(df)
    os.makedirs("output", exist_ok=True)
    summary.to_csv("output/dqa_site_summary.csv", index=False)

    print("\n=== Data Quality Assurance: Site Summary ===")
    print(summary[["site_id", "n_patients", "completeness_rate", "protocol_deviation_rate",
                    "query_rate", "n_flags", "dqa_status"]].to_string(index=False))

    problem_sites = summary[summary["dqa_status"] == "PROBLEM SITE - review required"]
    print(f"\nProblem sites flagged: {list(problem_sites['site_id'])}")
    return summary


if __name__ == "__main__":
    main()
