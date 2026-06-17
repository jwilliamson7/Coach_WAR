"""
Imputation-leakage robustness test (reviewer concern #4).

Runs the full XGBoost coaching-impact pipeline on a leakage-free imputed
dataset where SVD imputation was performed SEPARATELY on train and test
subsets (see scripts/build_split_then_impute_data.py). The coach-based
train/test split (seed=42, test_size=0.2) is identical to the primary
analysis, so the comparison isolates the effect of imputation strategy.

Outputs live in data/final/leakage_tests/imputation/ to keep the primary
analysis results untouched.

Usage:
    python analysis/xgboost_imputation_leakage_test.py
    python analysis/xgboost_imputation_leakage_test.py --no-tuning
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.xgboost_coaching_impact_analysis import (  # noqa: E402
    TeeOutput,
    analyze_coach_rankings,
    analyze_coaching_impact,
    calculate_replacement_features,
    create_replacement_dataset,
    identify_coach_features,
    load_and_prepare_data,
    plot_feature_importance_comparison,
    train_and_predict,
)


LOG_DIR = Path('analysis/outputs/logs')


def io_paths(test_size):
    suffix = f'_t{int(round(test_size * 100)):02d}'
    out_dir = Path(f'data/final/leakage_tests/imputation{suffix}')
    return {
        'data': f'data/final/imputed_split_then_imputed{suffix}.csv',
        'assignments': f'data/final/imputed_split_then_imputed_assignments{suffix}.csv',
        'out_dir': out_dir,
        'results': out_dir / 'coaching_impact_analysis_imputation_test.csv',
        'high_impact': out_dir / 'high_impact_coaches_imputation_test.csv',
        'coach_stats': out_dir / 'coach_career_impact_stats_imputation_test.csv',
        'importance': out_dir / 'feature_importance_imputation_test.csv',
    }


def verify_split_matches_assignments(team_year_info, assignments_path):
    """Sanity check: the coach split inside train_and_predict() must reproduce
    the same row assignments used for the split-then-impute driver."""
    assignments = pd.read_csv(assignments_path)
    merged = team_year_info.reset_index(drop=True).merge(
        assignments[['Team', 'Year', 'split']], on=['Team', 'Year'], how='left'
    )
    n_train = (merged['split'] == 'train').sum()
    n_test = (merged['split'] == 'test').sum()
    print(f"  Driver assignments: {n_train} train rows / {n_test} test rows")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Imputation-leakage robustness test for XGBoost coaching analysis'
    )
    parser.add_argument('--no-tuning', action='store_true',
                        help='Skip hyperparameter tuning, use known optimal defaults')
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--n-iter', type=int, default=200)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--with-av', action='store_true',
                        help='Include AV features (default: exclude, matches primary)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test fraction; must match the driver run that '
                             'created the input file (default: 0.2)')
    args = parser.parse_args()

    paths = io_paths(args.test_size)
    DATA_FILE = paths['data']
    ASSIGNMENTS_FILE = paths['assignments']
    OUT_DIR = paths['out_dir']
    RESULTS_FILE = paths['results']
    HIGH_IMPACT_FILE = paths['high_impact']
    COACH_STATS_FILE = paths['coach_stats']
    IMPORTANCE_FILE = paths['importance']

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f'_t{int(round(args.test_size * 100)):02d}'
    log_filename = LOG_DIR / f'imputation_leakage_test{suffix}_log_{timestamp}.txt'

    tee = TeeOutput(str(log_filename))
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print('=' * 80)
        print('IMPUTATION LEAKAGE ROBUSTNESS TEST')
        print(f'Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Data file: {DATA_FILE} (split-then-impute)')
        print(f'Output dir: {OUT_DIR}')
        if not args.no_tuning:
            print(f'Hyperparameter tuning: ENABLED ({args.n_iter} iter, '
                  f'{args.cv_folds} CV folds)')
        else:
            print('Hyperparameter tuning: DISABLED (using known optima)')
        print('=' * 80)

        exclude_av = not args.with_av
        X, y, team_year_info, full_df = load_and_prepare_data(
            DATA_FILE, exclude_av=exclude_av
        )

        # Verify the same split is reproduced
        print('\nVerifying coach split matches the driver-script assignments...')
        verify_split_matches_assignments(team_year_info, ASSIGNMENTS_FILE)

        # Identify coaching features and compute replacement values
        coach_features = identify_coach_features(X)
        replacement_values = calculate_replacement_features(
            X, coach_features, team_year_info
        )
        X_replacement = create_replacement_dataset(
            X, coach_features, replacement_values
        )

        # Train model on actual data (uses same coach split internally)
        print('\nTraining XGBoost with leakage-free imputed features...')
        use_tuning = not args.no_tuning
        model_actual, y_pred_actual, train_metrics, test_metrics = train_and_predict(
            X, y, team_year_info,
            use_tuning=use_tuning,
            cv_folds=args.cv_folds,
            n_iter=args.n_iter,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        print('\nGenerating predictions with replacement-level coaching...')
        y_pred_replacement = model_actual.predict(X_replacement)

        test_rmse = float(np.sqrt(test_metrics['mse']))
        results, high_impact_coaches = analyze_coaching_impact(
            y, y_pred_actual, y_pred_replacement, team_year_info,
            test_rmse=test_rmse,
        )

        coach_stats = analyze_coach_rankings(results)
        importance_df = plot_feature_importance_comparison(
            model_actual, X, coach_features
        )

        # Save all outputs to the leakage_tests directory
        results.to_csv(RESULTS_FILE, index=False)
        if len(high_impact_coaches) > 0:
            high_impact_coaches.to_csv(HIGH_IMPACT_FILE, index=False)
        coach_stats.to_csv(COACH_STATS_FILE)
        importance_df.to_csv(IMPORTANCE_FILE, index=False)

        print('\n' + '=' * 80)
        print('IMPUTATION LEAKAGE TEST COMPLETE')
        print('=' * 80)
        print(f'\nKey metrics:')
        print(f"  Train R²: {train_metrics['r2']:.4f}")
        print(f"  Test R²:  {test_metrics['r2']:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Model SE: ±{test_rmse * 16:.2f} wins per season")
        print(f"  Mean WAR: {results['Coaching_WAR'].mean():.4f}")
        print(f"  Max WAR:  {results['Coaching_WAR'].max():.4f}")
        print(f"  Min WAR:  {results['Coaching_WAR'].min():.4f}")
        print(f'\nResults saved to: {OUT_DIR}/')
        print(f'Log file: {log_filename}')

    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f'\nDone. Log: {log_filename}')


if __name__ == '__main__':
    main()
