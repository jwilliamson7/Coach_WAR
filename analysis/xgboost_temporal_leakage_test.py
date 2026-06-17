"""
Temporal-leakage robustness test (reviewer concern #5).

The primary analysis splits coaches randomly across the full 1970-2024 span,
which means the model can train on (e.g.) 2024 seasons and be tested on 1985
seasons. The reviewer is concerned that league-evolution features make this an
unrealistic generalization test.

This script re-runs the XGBoost coaching-impact pipeline with a CHRONOLOGICAL
split: train on seasons with Year <= 2014, test on Year >= 2015. This allows
within-coach correlation (a coach's 2010 season may be in train and 2018 season
in test) but directly addresses the era-mixing concern.

Cutoff = 2014 gives ~80.5%/19.5% train/test, matching the primary 80/20 ratio.

Outputs live in data/final/leakage_tests/temporal/ to keep the primary
analysis results untouched.

Usage:
    python analysis/xgboost_temporal_leakage_test.py
    python analysis/xgboost_temporal_leakage_test.py --no-tuning
    python analysis/xgboost_temporal_leakage_test.py --cutoff 2010
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

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


DEFAULT_DATA_FILE = 'data/final/imputed_final_data.csv'  # primary imputed file
DEFAULT_CUTOFF = 2014

LOG_DIR = Path('analysis/outputs/logs')


def io_paths(out_subdir):
    out_dir = Path(f'data/final/leakage_tests/{out_subdir}')
    return {
        'out_dir': out_dir,
        'results': out_dir / 'coaching_impact_analysis_temporal_test.csv',
        'high_impact': out_dir / 'high_impact_coaches_temporal_test.csv',
        'coach_stats': out_dir / 'coach_career_impact_stats_temporal_test.csv',
        'importance': out_dir / 'feature_importance_temporal_test.csv',
    }


def train_and_predict_chronological(X, y, team_year_info, cutoff_year,
                                     use_tuning=True, cv_folds=5, n_iter=200,
                                     random_state=42):
    """Train XGBoost with a chronological train/test split.

    Mirrors xgboost_coaching_impact_analysis.train_and_predict but replaces
    the coach-based split with a year-based split.
    """
    # --- Chronological split based on Year ---
    if 'Year' not in team_year_info.columns:
        raise ValueError('Year column required for chronological split')

    years = team_year_info['Year'].reset_index(drop=True)
    train_mask = (years <= cutoff_year).values
    test_mask = ~train_mask

    print(f'\nChronological split (cutoff = {cutoff_year}):')
    print(f'  Training: years {int(years.min())}-{cutoff_year}, '
          f'{train_mask.sum()} rows ({train_mask.mean()*100:.1f}%)')
    print(f'  Test: years {cutoff_year + 1}-{int(years.max())}, '
          f'{test_mask.sum()} rows ({test_mask.mean()*100:.1f}%)')

    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)

    X_train = X_reset[train_mask].copy()
    X_test = X_reset[test_mask].copy()
    y_train = y_reset[train_mask].copy()
    y_test = y_reset[test_mask].copy()

    print(f'  X_train shape: {X_train.shape}')
    print(f'  X_test shape:  {X_test.shape}')

    if use_tuning:
        print(f'\nHyperparameter tuning: {n_iter} iter, {cv_folds} CV folds...')
        param_dist = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
            'max_depth': [2, 3, 4, 5],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1.0, 1.5, 2.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0, 1.5, 2.0],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        )
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        print(f'\nBest CV R²: {random_search.best_score_:.4f}')
        print('Best parameters:')
        for k, v in random_search.best_params_.items():
            print(f'  {k}: {v}')
    else:
        print('\nUsing default hyperparameters (no tuning)...')
        # Same defaults as primary script
        model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=2, gamma=0,
            reg_alpha=0.1, reg_lambda=0.1, subsample=0.7,
            colsample_bytree=1.0, min_child_weight=5,
            objective='reg:squarederror', random_state=42,
            verbosity=0, n_jobs=-1,
        )
        model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = {
        'mse': mean_squared_error(y_train, y_train_pred),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred),
    }
    test_metrics = {
        'mse': mean_squared_error(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'r2': r2_score(y_test, y_test_pred),
    }

    print('\n' + '=' * 60)
    print('TRAIN/TEST PERFORMANCE (chronological split)')
    print('=' * 60)
    print(f"{'Metric':<10} {'Train':<12} {'Test':<12} {'Diff':<12}")
    for k, label in [('r2', 'R²'), ('mse', 'MSE'), ('mae', 'MAE')]:
        if k == 'mse':
            tr = np.sqrt(train_metrics[k])
            te = np.sqrt(test_metrics[k])
            print(f"{'RMSE':<10} {tr:<12.4f} {te:<12.4f} {te - tr:<12.4f}")
        else:
            tr = train_metrics[k]
            te = test_metrics[k]
            print(f"{label:<10} {tr:<12.4f} {te:<12.4f} {te - tr:<12.4f}")

    # Predict on full dataset for WAR computation
    y_pred_full = model.predict(X_reset)
    return model, y_pred_full, train_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Temporal-leakage robustness test (chronological split)'
    )
    parser.add_argument('--no-tuning', action='store_true',
                        help='Skip hyperparameter tuning, use known optimal defaults')
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--n-iter', type=int, default=200)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--cutoff', type=int, default=DEFAULT_CUTOFF,
                        help=f'Year cutoff for chronological split '
                             f'(default: {DEFAULT_CUTOFF})')
    parser.add_argument('--with-av', action='store_true')
    parser.add_argument('--data-file', default=DEFAULT_DATA_FILE,
                        help='Imputed data file to use (default: primary impute-then-split)')
    parser.add_argument('--out-subdir', default='temporal',
                        help='Subdirectory under data/final/leakage_tests/ '
                             'for outputs (default: temporal)')
    args = parser.parse_args()

    DATA_FILE = args.data_file
    paths = io_paths(args.out_subdir)
    OUT_DIR = paths['out_dir']
    RESULTS_FILE = paths['results']
    HIGH_IMPACT_FILE = paths['high_impact']
    COACH_STATS_FILE = paths['coach_stats']
    IMPORTANCE_FILE = paths['importance']

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = LOG_DIR / f'temporal_leakage_test_{args.out_subdir}_log_{timestamp}.txt'

    tee = TeeOutput(str(log_filename))
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print('=' * 80)
        print('TEMPORAL LEAKAGE ROBUSTNESS TEST')
        print(f'Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Data file: {DATA_FILE}')
        print(f'Chronological split cutoff: year <= {args.cutoff} (train) '
              f'/ year > {args.cutoff} (test)')
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

        coach_features = identify_coach_features(X)
        replacement_values = calculate_replacement_features(
            X, coach_features, team_year_info
        )
        X_replacement = create_replacement_dataset(
            X, coach_features, replacement_values
        )

        print('\nTraining XGBoost with chronological split...')
        use_tuning = not args.no_tuning
        model_actual, y_pred_actual, train_metrics, test_metrics = \
            train_and_predict_chronological(
                X, y, team_year_info,
                cutoff_year=args.cutoff,
                use_tuning=use_tuning,
                cv_folds=args.cv_folds,
                n_iter=args.n_iter,
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

        results.to_csv(RESULTS_FILE, index=False)
        if len(high_impact_coaches) > 0:
            high_impact_coaches.to_csv(HIGH_IMPACT_FILE, index=False)
        coach_stats.to_csv(COACH_STATS_FILE)
        importance_df.to_csv(IMPORTANCE_FILE, index=False)

        print('\n' + '=' * 80)
        print('TEMPORAL LEAKAGE TEST COMPLETE')
        print('=' * 80)
        print(f'\nKey metrics:')
        print(f"  Train R² (year<={args.cutoff}): {train_metrics['r2']:.4f}")
        print(f"  Test R²  (year>{args.cutoff}):  {test_metrics['r2']:.4f}")
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
