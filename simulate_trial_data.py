"""
Simulate multi-site clinical trial data with real site-to-site heterogeneity.

Generates patient-level outcomes for a single 2-arm (Control vs Treatment)
randomized trial run across 9 sites, plus site-level operational metadata
(enrollment, data-quality characteristics). Site heterogeneity is injected
in three independent ways so the downstream pipeline has real signal to
recover and real problems to flag:

1. Random-effect site intercepts (sites differ in baseline outcome level).
2. A known, injected overall treatment effect (the mixed-effects model
   below is validated against recovering this).
3. One "problem site" (SITE-07) is given a deliberately elevated
   protocol-deviation rate, query rate, and missingness so the DQA
   pipeline has a real site to flag (not a hypothetical).

Fixed random seed -> fully reproducible.
"""
import numpy as np
import pandas as pd
import os

SEED = 42
rng = np.random.default_rng(SEED)

N_SITES = 9
TRUE_TREATMENT_EFFECT = -3.5  # true adjusted mean difference, Treatment - Control, on continuous endpoint
RESIDUAL_SD = 4.0
SITE_RANDOM_EFFECT_SD = 2.2  # site-to-site heterogeneity in baseline outcome level

PROBLEM_SITE = "SITE-07"

SITE_IDS = [f"SITE-{i:02d}" for i in range(1, N_SITES + 1)]

# Site-level enrollment sizes: real-world heterogeneity (some big academic
# centers, some small community sites)
SITE_N = {
    "SITE-01": 85, "SITE-02": 62, "SITE-03": 40, "SITE-04": 110,
    "SITE-05": 55, "SITE-06": 48, "SITE-07": 38, "SITE-08": 70,
    "SITE-09": 52,
}

# Site-level random intercepts (true, latent, heterogeneity in outcome level
# unrelated to treatment) -- drawn once, used to generate patient outcomes
SITE_RANDOM_INTERCEPT = {
    site: rng.normal(0, SITE_RANDOM_EFFECT_SD) for site in SITE_IDS
}

# Baseline protocol-deviation and query rates (per-patient probability),
# with SITE-07 deliberately elevated to simulate a real problem site
BASE_PD_RATE = 0.04
BASE_QUERY_RATE = 0.08
BASE_MISSING_RATE = 0.03

SITE_PD_RATE = {site: BASE_PD_RATE + rng.normal(0, 0.01) for site in SITE_IDS}
SITE_QUERY_RATE = {site: BASE_QUERY_RATE + rng.normal(0, 0.02) for site in SITE_IDS}
SITE_MISSING_RATE = {site: BASE_MISSING_RATE + rng.normal(0, 0.01) for site in SITE_IDS}

# Inject the problem site's elevated rates
SITE_PD_RATE[PROBLEM_SITE] = 0.22
SITE_QUERY_RATE[PROBLEM_SITE] = 0.31
SITE_MISSING_RATE[PROBLEM_SITE] = 0.19

for d in (SITE_PD_RATE, SITE_QUERY_RATE, SITE_MISSING_RATE):
    for k in d:
        d[k] = max(0.0, d[k])


def simulate():
    rows = []
    patient_counter = 1
    for site in SITE_IDS:
        n = SITE_N[site]
        site_int = SITE_RANDOM_INTERCEPT[site]
        pd_rate = SITE_PD_RATE[site]
        query_rate = SITE_QUERY_RATE[site]
        missing_rate = SITE_MISSING_RATE[site]

        arm = rng.choice(["Control", "Treatment"], size=n, p=[0.5, 0.5])
        baseline = rng.normal(50, 8, size=n)

        trt_effect = np.where(arm == "Treatment", TRUE_TREATMENT_EFFECT, 0.0)
        noise = rng.normal(0, RESIDUAL_SD, size=n)
        endpoint = 50 + site_int + 0.3 * (baseline - 50) + trt_effect + noise

        protocol_deviation = rng.binomial(1, pd_rate, size=n)
        query_flag = rng.binomial(1, query_rate, size=n)
        missing_flag = rng.binomial(1, missing_rate, size=n)

        endpoint_observed = endpoint.copy()
        endpoint_observed[missing_flag == 1] = np.nan

        for i in range(n):
            rows.append({
                "patient_id": f"P-{patient_counter:04d}",
                "site_id": site,
                "arm": arm[i],
                "baseline": round(baseline[i], 2),
                "endpoint": round(endpoint_observed[i], 2) if not np.isnan(endpoint_observed[i]) else np.nan,
                "endpoint_missing": int(missing_flag[i]),
                "protocol_deviation": int(protocol_deviation[i]),
                "query_flag": int(query_flag[i]),
            })
            patient_counter += 1

    df = pd.DataFrame(rows)
    return df


def main():
    os.makedirs("data", exist_ok=True)
    df = simulate()
    df.to_csv("data/patient_level_data.csv", index=False)

    print(f"Simulated {len(df)} patients across {N_SITES} sites.")
    print(f"True injected overall treatment effect: {TRUE_TREATMENT_EFFECT}")
    print(f"Injected problem site: {PROBLEM_SITE} "
          f"(PD rate={SITE_PD_RATE[PROBLEM_SITE]:.2f}, "
          f"query rate={SITE_QUERY_RATE[PROBLEM_SITE]:.2f}, "
          f"missingness={SITE_MISSING_RATE[PROBLEM_SITE]:.2f})")
    return df


if __name__ == "__main__":
    main()
