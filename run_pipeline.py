"""
Master pipeline runner.
Calls all analysis scripts in order, passing the loaded DataFrame
between steps to avoid redundant file I/O.

Usage:
    python run_pipeline.py           # run all steps
    python run_pipeline.py --step 3  # run only script 03 (0-indexed by list)
"""

import os
import sys
import argparse
import traceback


#Adds project root to path so config and src scripts can all find each other

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


# Creates all required directories before anything else runs

from config import PATHS

for dir_path in [
    os.path.join(PROJECT_ROOT, "data", "raw"),
    os.path.join(PROJECT_ROOT, "data", "processed"),
    PATHS["tables"],
    PATHS["figures"],
    PATHS["logs"],
    os.path.join(PROJECT_ROOT, "notebooks"),
    os.path.join(PROJECT_ROOT, "configs"),
]:
    os.makedirs(dir_path, exist_ok=True)

# Import all script modules
import importlib.util

def load_module(filename: str):
    """Dynamically load a src script as a module."""
    path = os.path.join(PROJECT_ROOT, "src", filename)
    if not os.path.exists(path):
        # Also try project root
        path = os.path.join(PROJECT_ROOT, filename)
    spec   = importlib.util.spec_from_file_location(filename.replace(".py",""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STEPS = [
    ("00_environment_log.py",    "log_environment",  False),  # no df in/out
    ("01_dataset_audit.py",      "run",              True),   # returns df
    ("02_target_construction.py","run",              True),   # returns df
    ("03_descriptive_stats.py",  "run",              True),   # returns df
    ("04_distributions.py",      "run",              False),  # no return
    ("05_correlations.py",       "run",              False),
    ("06_near_neutral.py",       "run",              False),
    ("07_subgroups.py",          "run",              False),
    ("08_feature_report.py",     "run",              False),
    ("09_pilot_modeling.py",     "run",              False),
    ("10_robustness.py",         "run",              False),
]


def run_all(start_step: int = 0) -> None:
    df = None   # DataFrame passed between steps that need it

    from config import PATHS as P
    import os

    # Special handling for step 00 
    for i, (filename, func_name, returns_df) in enumerate(STEPS):
        if i < start_step:
            continue

        print(f"\n{'='*60}")
        print(f"STEP {i:02d}: {filename}")
        print(f"{'='*60}")

        try:
            mod = load_module(filename)
            fn  = getattr(mod, func_name)

            if filename == "00_environment_log.py":
                # Step 00 is different
                log_path = os.path.join(P["logs"], "environment_log.txt")
                fn(log_path)

            elif df is not None and returns_df:
                df = fn(df)

            elif df is not None and not returns_df:
                fn(df)

            elif df is None and returns_df:
                result = fn()
                if result is not None:
                    df = result

            else:
                fn()

            print(f"\n  ✓ Step {i:02d} completed successfully.")

        except FileNotFoundError as e:
            print(f"\n  [ERROR] Step {i:02d} failed  file not found:")
            print(f"  {e}")
            print(f"  Pipeline halted at step {i:02d}. Resolve the issue and re-run.")
            sys.exit(1)

        except Exception as e:
            print(f"\n  [ERROR] Step {i:02d} raised an exception:")
            traceback.print_exc()
            print(f"\n  Continuing to next step...")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Outputs:")
    print(f"  Tables  → {P['tables']}")
    print(f"  Figures → {P['figures']}")
    print(f"  Logs    → {P['logs']}")
    print(f"  Data    → {os.path.dirname(P['processed_data'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run thesis analysis pipeline")
    parser.add_argument("--step", type=int, default=0,
                        help="Start from this step index (0-based). Default: 0 (run all).")
    args = parser.parse_args()
    run_all(start_step=args.step)
