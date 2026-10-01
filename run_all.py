"""Runs the full Clinical Trial Data Analytics pipeline end-to-end."""
import subprocess
import sys

import simulate_trial_data
import dqa_pipeline
import mixed_effects_model
import dashboard


def main():
    print("=" * 70)
    print("STEP 1: Simulating multi-site trial data")
    print("=" * 70)
    simulate_trial_data.main()

    print("\n" + "=" * 70)
    print("STEP 2: Data Quality Assurance pipeline")
    print("=" * 70)
    dqa_pipeline.main()

    print("\n" + "=" * 70)
    print("STEP 3: Mixed-effects statistical model")
    print("=" * 70)
    mixed_effects_model.main()

    print("\n" + "=" * 70)
    print("STEP 4: Leadership dashboard")
    print("=" * 70)
    dashboard.build_dashboard()

    print("\n" + "=" * 70)
    print("STEP 5: Running test suite")
    print("=" * 70)
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
