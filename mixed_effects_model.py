"""
Statistical model: linear mixed-effects model estimating the overall
treatment effect on the primary continuous endpoint, accounting for
site-to-site heterogeneity via a random intercept per site (site as a
random effect), adjusted for baseline covariate. This is the real
statistical workhorse of the pipeline -- the same conceptual model
(`lmer(endpoint ~ arm + baseline + (1 | site))`) that a biostatistician
would run in R, implemented here with `statsmodels.MixedLM` and actually
fit against the simulated data (observed-case / efficacy-evaluable
population, i.e. patients with a non-missing endpoint).

Also fits a naive fixed-effect-only OLS (ignoring site) and a per-site
stratified analysis, for comparison -- demonstrating why accounting for
site heterogeneity matters (the mixed model's standard error properly
reflects between-site variance; a naive pooled OLS that ignores clustering
understates it).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os


def fit_mixed_effects(df: pd.DataFrame):
    evaluable = df.dropna(subset=["endpoint"]).copy()
    evaluable["arm_trt"] = (evaluable["arm"] == "Treatment").astype(int)

    # Mixed-effects model: site as random intercept
    md = smf.mixedlm("endpoint ~ arm_trt + baseline", evaluable, groups=evaluable["site_id"])
    mdf = md.fit(reml=True)

    # Naive pooled OLS ignoring site clustering, for comparison
    ols = smf.ols("endpoint ~ arm_trt + baseline", evaluable).fit()

    # Per-site stratified treatment effect estimates (fixed-effect per site)
    site_effects = []
    for site, g in evaluable.groupby("site_id"):
        if g["arm_trt"].nunique() < 2:
            continue
        m = smf.ols("endpoint ~ arm_trt + baseline", g).fit()
        site_effects.append({
            "site_id": site,
            "n": len(g),
            "treatment_effect": m.params["arm_trt"],
            "se": m.bse["arm_trt"],
        })
    site_effects_df = pd.DataFrame(site_effects)

    results = {
        "mixed_model_treatment_effect": mdf.params["arm_trt"],
        "mixed_model_se": mdf.bse["arm_trt"],
        "mixed_model_ci_lower": mdf.conf_int().loc["arm_trt", 0],
        "mixed_model_ci_upper": mdf.conf_int().loc["arm_trt", 1],
        "mixed_model_pvalue": mdf.pvalues["arm_trt"],
        "site_random_effect_variance": float(mdf.cov_re.iloc[0, 0]),
        "residual_variance": mdf.scale,
        "naive_ols_treatment_effect": ols.params["arm_trt"],
        "naive_ols_se": ols.bse["arm_trt"],
        "n_evaluable": len(evaluable),
        "n_sites": evaluable["site_id"].nunique(),
    }
    return results, site_effects_df, mdf


def main():
    df = pd.read_csv("data/patient_level_data.csv")
    results, site_effects_df, mdf = fit_mixed_effects(df)

    os.makedirs("output", exist_ok=True)
    pd.DataFrame([results]).to_csv("output/mixed_model_results.csv", index=False)
    site_effects_df.to_csv("output/per_site_treatment_effects.csv", index=False)

    print("\n=== Mixed-Effects Model (site as random intercept) ===")
    print(f"N evaluable patients: {results['n_evaluable']} across {results['n_sites']} sites")
    print(f"Overall treatment effect (adjusted): {results['mixed_model_treatment_effect']:.4f}")
    print(f"SE: {results['mixed_model_se']:.4f}")
    print(f"95% CI: [{results['mixed_model_ci_lower']:.4f}, {results['mixed_model_ci_upper']:.4f}]")
    print(f"p-value: {results['mixed_model_pvalue']:.3e}")
    print(f"Site random-effect variance: {results['site_random_effect_variance']:.4f}")
    print(f"\nNaive pooled OLS (ignoring site clustering) treatment effect: "
          f"{results['naive_ols_treatment_effect']:.4f} (SE {results['naive_ols_se']:.4f})")
    print("\nPer-site stratified treatment effect estimates:")
    print(site_effects_df.to_string(index=False))

    return results, site_effects_df


if __name__ == "__main__":
    main()
