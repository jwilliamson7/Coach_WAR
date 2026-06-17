"""
Pipeline 2: Full-data WAR estimation.

Companion to xgboost_imputation_leakage_test.py (Pipeline 1, validation).
Pipeline 2 trains an XGBoost model on ALL 1,637 rows (no train/test split)
to maximize data efficiency for in-sample WAR estimation. This is standard
ML practice for deployment: validate methodology with a holdout (Pipeline 1),
then refit on all data for final estimates.

Hyperparameters are fixed at the known optima from prior tuning runs (matches
the --no-tuning defaults in xgboost_coaching_impact_analysis.py). The reported
WAR_SE is taken from Pipeline 1's test RMSE so that uncertainty reflects honest
out-of-sample generalization rather than in-sample fit.

Outputs:
    data/final/pipeline2/coaching_impact_analysis_pipeline2.csv
    data/final/pipeline2/coach_career_impact_stats_pipeline2.csv
    data/final/pipeline2/feature_importance_pipeline2.csv
    data/final/pipeline2/high_impact_coaches_pipeline2.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
)


DATA_FILE = 'data/final/imputed_final_data.csv'

# Hyperparameters: known optima from prior tuning (200-iter RandomizedSearchCV,
# 5-fold CV). Match xgboost_coaching_impact_analysis.py --no-tuning defaults.
OPTIMAL_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 2,
    'gamma': 0,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'subsample': 0.7,
    'colsample_bytree': 1.0,
    'min_child_weight': 5,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'verbosity': 0,
    'n_jobs': -1,
}

# Pipeline 1's honest test RMSE (split-then-impute, random coach 80/20) is the
# source of the reported WAR SE. Read it from Pipeline 1's own output file so it
# can never silently drift from the validation run; fall back to the known value
# (with a warning) if Pipeline 1 has not been run.
PIPELINE1_IMPACT_FILE = Path(
    'data/final/leakage_tests/imputation_t20/'
    'coaching_impact_analysis_imputation_test.csv'
)
PIPELINE1_TEST_RMSE_FALLBACK = 0.1552  # ±2.48 wins per 16-game season

OUT_DIR = Path('data/final/pipeline2')
RESULTS_FILE = OUT_DIR / 'coaching_impact_analysis_pipeline2.csv'
HIGH_IMPACT_FILE = OUT_DIR / 'high_impact_coaches_pipeline2.csv'
COACH_STATS_FILE = OUT_DIR / 'coach_career_impact_stats_pipeline2.csv'
IMPORTANCE_FILE = OUT_DIR / 'feature_importance_pipeline2.csv'

# Canonical root copies — the single source of truth consumed by every
# downstream analysis (background/decade, persistence, trajectories, SHAP,
# Ridge comparison) and the paper. Pipeline 2 is the authoritative producer;
# the legacy xgboost_coaching_impact_analysis.py is deprecated (see its header).
ROOT_DIR = Path('data/final')
ROOT_RESULTS_FILE = ROOT_DIR / 'coaching_impact_analysis.csv'
ROOT_HIGH_IMPACT_FILE = ROOT_DIR / 'high_impact_coaches.csv'
ROOT_COACH_STATS_FILE = ROOT_DIR / 'coach_career_impact_stats.csv'
ROOT_IMPORTANCE_FILE = ROOT_DIR / 'feature_importance_coaching_analysis.csv'

LOG_DIR = Path('analysis/outputs/logs')


def load_pipeline1_test_rmse():
    """Return Pipeline 1's test RMSE, read from its impact CSV (WAR_SE / 16).

    Reading it at runtime keeps the reported WAR SE in sync with the actual
    validation run rather than a hand-copied constant. Falls back to the known
    value with a warning if Pipeline 1 has not been run.
    """
    if PIPELINE1_IMPACT_FILE.exists():
        war_se = pd.read_csv(PIPELINE1_IMPACT_FILE)['WAR_SE'].iloc[0]
        return float(war_se) / 16.0
    print(f'WARNING: {PIPELINE1_IMPACT_FILE} not found; falling back to '
          f'PIPELINE1_TEST_RMSE_FALLBACK = {PIPELINE1_TEST_RMSE_FALLBACK}')
    return PIPELINE1_TEST_RMSE_FALLBACK


def train_full_data(X, y, pipeline1_test_rmse):
    """Fit XGBoost with optimal hyperparameters on the full dataset."""
    print(f'\nTraining XGBoost on full dataset ({X.shape[0]} rows, '
          f'{X.shape[1]} features)...')
    print('Hyperparameters (fixed at known optima):')
    for k, v in OPTIMAL_PARAMS.items():
        if k not in ('objective', 'random_state', 'verbosity', 'n_jobs'):
            print(f'  {k}: {v}')

    model = xgb.XGBRegressor(**OPTIMAL_PARAMS)
    model.fit(X, y)

    # In-sample metrics (for reference; not used as primary uncertainty)
    y_pred = model.predict(X)
    metrics = {
        'r2': r2_score(y, y_pred),
        'mse': mean_squared_error(y, y_pred),
        'mae': mean_absolute_error(y, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
    }
    print(f'\nIn-sample fit (NOT generalization metric):')
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f'\nReported WAR_SE uses Pipeline 1 test RMSE = {pipeline1_test_rmse:.4f}')
    print(f'  (±{pipeline1_test_rmse * 16:.2f} wins per 16-game season)')

    return model, y_pred, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline 2: Full-data WAR estimation (no train/test split)'
    )
    parser.add_argument('--with-av', action='store_true',
                        help='Include AV features (default: exclude, matches primary)')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = LOG_DIR / f'pipeline2_war_estimation_log_{timestamp}.txt'

    tee = TeeOutput(str(log_filename))
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        pipeline1_test_rmse = load_pipeline1_test_rmse()
        print('=' * 80)
        print('PIPELINE 2: FULL-DATA WAR ESTIMATION')
        print(f'Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Data file: {DATA_FILE}')
        print(f'Output dir: {OUT_DIR}')
        print('Strategy: train on ALL rows with fixed optimal hyperparameters.')
        print(f'WAR_SE source: Pipeline 1 test RMSE = {pipeline1_test_rmse:.4f} '
              f'(±{pipeline1_test_rmse * 16:.2f} wins/season)')
        print('=' * 80)

        # Load data and identify coaching features
        exclude_av = not args.with_av
        X, y, team_year_info, full_df = load_and_prepare_data(
            DATA_FILE, exclude_av=exclude_av
        )
        coach_features = identify_coach_features(X)

        # Compute replacement (median) coach feature values
        replacement_values = calculate_replacement_features(
            X, coach_features, team_year_info
        )
        X_replacement = create_replacement_dataset(
            X, coach_features, replacement_values
        )

        # Train on full dataset
        model, y_pred_actual, in_sample_metrics = train_full_data(
            X, y, pipeline1_test_rmse
        )

        # Predict win% with replacement coach features
        print('\nGenerating predictions with replacement-level coaching...')
        y_pred_replacement = model.predict(X_replacement)

        # Compute WAR using Pipeline 1's test RMSE for honest SE
        results, high_impact_coaches = analyze_coaching_impact(
            y, y_pred_actual, y_pred_replacement, team_year_info,
            test_rmse=pipeline1_test_rmse,
        )

        # Career-level rankings and feature importance
        coach_stats = analyze_coach_rankings(results)
        importance_df = plot_feature_importance_comparison(
            model, X, coach_features
        )

        # Save provenance copies under pipeline2/
        results.to_csv(RESULTS_FILE, index=False)
        if len(high_impact_coaches) > 0:
            high_impact_coaches.to_csv(HIGH_IMPACT_FILE, index=False)
        coach_stats.to_csv(COACH_STATS_FILE)
        importance_df.to_csv(IMPORTANCE_FILE, index=False)

        # Save canonical root files (single source of truth for all downstream
        # analyses and the paper). This replaces the previous manual copy step.
        results.to_csv(ROOT_RESULTS_FILE, index=False)
        if len(high_impact_coaches) > 0:
            high_impact_coaches.to_csv(ROOT_HIGH_IMPACT_FILE, index=False)
        coach_stats.to_csv(ROOT_COACH_STATS_FILE)
        importance_df.to_csv(ROOT_IMPORTANCE_FILE, index=False)
        print('\nCanonical root files written (authoritative):')
        for f in (ROOT_RESULTS_FILE, ROOT_HIGH_IMPACT_FILE,
                  ROOT_COACH_STATS_FILE, ROOT_IMPORTANCE_FILE):
            print(f'  {f}')

        print('\n' + '=' * 80)
        print('PIPELINE 2 COMPLETE')
        print('=' * 80)
        print(f"\nIn-sample R² (full data fit): {in_sample_metrics['r2']:.4f}")
        print(f"In-sample RMSE: {in_sample_metrics['rmse']:.4f}")
        print(f'\nHonest validation metrics (from Pipeline 1):')
        print(f'  Test R²: ~0.43 (split-then-impute, random coach 80/20)')
        print(f'  Test RMSE: {pipeline1_test_rmse:.4f}')
        print(f'  WAR SE: ±{pipeline1_test_rmse * 16:.2f} wins per 16-game season')
        print(f'\nWAR distribution:')
        print(f"  Mean: {results['Coaching_WAR'].mean():.4f}")
        print(f"  Std:  {results['Coaching_WAR'].std():.4f}")
        print(f"  Max:  {results['Coaching_WAR'].max():.4f}")
        print(f"  Min:  {results['Coaching_WAR'].min():.4f}")
        print(f'\nResults saved to: {OUT_DIR}/')
        print(f'Log: {log_filename}')

    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f'\nDone. Log: {log_filename}')


if __name__ == '__main__':
    main()
