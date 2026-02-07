"""
Master script to regenerate all figures for the LaTeX paper with consistent font.
This script runs all necessary analysis scripts to create publication-ready figures.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_script(script_path, font='serif', extra_args=None):
    """Run a Python script with optional font argument."""
    cmd = [sys.executable, script_path, '--font', font]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*80}")
    print(f"Running: {Path(script_path).name}")
    print(f"Font: {font}")
    if extra_args:
        print(f"Extra args: {' '.join(extra_args)}")
    print('='*80)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {script_path}:")
        print(e.stdout)
        print(e.stderr)
        return False
    except Exception as e:
        print(f"EXCEPTION running {script_path}: {e}")
        return False

def regenerate_all_figures(font='serif'):
    """Regenerate all figures needed for the LaTeX paper."""
    print(f"\n{'#'*80}")
    print(f"# REGENERATING ALL FIGURES WITH FONT: {font}")
    print(f"{'#'*80}\n")

    scripts_to_run = [
        # Two custom scripts for latex paper
        ('analysis/create_trajectory_figure.py', None),
        ('analysis/create_career_distribution_figure.py', None),

        # 2024 coach analysis figures
        ('analysis/coach_2024_scatter_png.py', None),
        ('analysis/coach_2024_trajectories_png.py', None),
        ('analysis/coach_2024_single_year_bar.py', None),

        # Background analysis
        ('analysis/coach_background_from_history.py', None),

        # Persistence and validation analysis
        ('analysis/coaching_war_persistence_analysis.py', None),
        ('analysis/coaching_war_persistence_by_background.py', None),
        ('analysis/coaching_war_multiyear_persistence.py', None),
        ('analysis/coaching_war_survivorship_bias_analysis.py', None),
        ('analysis/win_pct_persistence_analysis.py', None),
    ]

    successes = 0
    failures = 0

    for script, extra_args in scripts_to_run:
        if run_script(script, font, extra_args):
            successes += 1
        else:
            failures += 1

    # Summary
    print(f"\n{'#'*80}")
    print(f"# SUMMARY")
    print(f"{'#'*80}")
    print(f"Successful: {successes}/{len(scripts_to_run)}")
    print(f"Failed: {failures}/{len(scripts_to_run)}")

    if failures == 0:
        print("\nAll figures generated successfully!")
        print(f"\nFigures with font '{font}' are ready in:")
        print("  - latex/figures/")
        print("  - analysis/outputs/png/")
    else:
        print(f"\nWARNING: {failures} script(s) failed. Check output above.")

    return failures == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Regenerate all figures for LaTeX paper with consistent font'
    )
    parser.add_argument('--font', type=str, default='serif',
                       help='Font family to use for all figures (default: serif)')
    args = parser.parse_args()

    success = regenerate_all_figures(font=args.font)
    sys.exit(0 if success else 1)
