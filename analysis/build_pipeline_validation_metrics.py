"""
Build the source-of-truth validation metrics CSV documenting both pipelines.

Pipeline 1 (validation): split-then-impute, random coach 80/20, no tuning.
  Source: analysis/outputs/logs/imputation_leakage_test_log_*.txt
          and data/final/leakage_tests/imputation/*.csv

Pipeline 2 (estimation): full data, train on all, no train/test split.
  Source: analysis/outputs/logs/pipeline2_war_estimation_log_*.txt
          and data/final/pipeline2/*.csv

Output: data/final/pipeline_validation_metrics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


PIPE1_IMPACT = Path('data/final/leakage_tests/imputation_t20/coaching_impact_analysis_imputation_test.csv')
PIPE2_IMPACT = Path('data/final/pipeline2/coaching_impact_analysis_pipeline2.csv')

ASSIGNMENTS = Path('data/final/imputed_split_then_imputed_assignments_t20.csv')
COMBINED_IMPACT = Path('data/final/leakage_tests/temporal_combined/coaching_impact_analysis_temporal_test.csv')

OUT_CSV = Path('data/final/pipeline_validation_metrics.csv')


# Optimal hyperparameters (200-iter RandomizedSearchCV, 5-fold CV)
OPTIMAL_HP = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 2,
    'gamma': 0,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'subsample': 0.7,
    'colsample_bytree': 1.0,
    'min_child_weight': 5,
}


def metrics_from_impact(impact_df, mask=None):
    """Compute R², RMSE, MAE on a subset of the impact CSV."""
    if mask is not None:
        df = impact_df[mask]
    else:
        df = impact_df
    y_true = df['Actual_Win_Pct']
    y_pred = df['Predicted_With_Coach']
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        'n': len(df),
        'r2': r2,
        'rmse': rmse,
        'rmse_wins_per_season': rmse * 16,
    }


def main():
    pipe1 = pd.read_csv(PIPE1_IMPACT)
    pipe2 = pd.read_csv(PIPE2_IMPACT)
    assignments = pd.read_csv(ASSIGNMENTS)
    combined = pd.read_csv(COMBINED_IMPACT)

    # Pipeline 1 metrics broken out by train/test
    pipe1_with_split = pipe1.merge(
        assignments[['Team', 'Year', 'split']], on=['Team', 'Year'], how='left'
    )
    p1_train = metrics_from_impact(pipe1_with_split,
                                    mask=(pipe1_with_split['split'] == 'train'))
    p1_test = metrics_from_impact(pipe1_with_split,
                                   mask=(pipe1_with_split['split'] == 'test'))
    p1_full = metrics_from_impact(pipe1)

    # Pipeline 2: in-sample metrics on full data
    p2_full = metrics_from_impact(pipe2)

    # Combined leakage test: chronological + split-then-impute
    # Split is implicit by year <= 2014 vs > 2014
    combined['_split'] = np.where(combined['Year'] <= 2014, 'train', 'test')
    c_train = metrics_from_impact(combined, mask=(combined['_split'] == 'train'))
    c_test = metrics_from_impact(combined, mask=(combined['_split'] == 'test'))

    rows = [
        # Pipeline 1 — validation
        {
            'pipeline': 'Pipeline 1 (validation)',
            'description': 'Split-then-impute, random coach 80/20',
            'subset': 'train',
            'n': p1_train['n'],
            'r2': p1_train['r2'],
            'rmse': p1_train['rmse'],
            'rmse_wins_per_16game_season': p1_train['rmse_wins_per_season'],
            'data_file': PIPE1_IMPACT.name,
        },
        {
            'pipeline': 'Pipeline 1 (validation)',
            'description': 'Split-then-impute, random coach 80/20',
            'subset': 'test',
            'n': p1_test['n'],
            'r2': p1_test['r2'],
            'rmse': p1_test['rmse'],
            'rmse_wins_per_16game_season': p1_test['rmse_wins_per_season'],
            'data_file': PIPE1_IMPACT.name,
        },
        # Pipeline 2 — estimation
        {
            'pipeline': 'Pipeline 2 (estimation)',
            'description': 'Full-data SVD impute, train on all rows (in-sample)',
            'subset': 'full (in-sample)',
            'n': p2_full['n'],
            'r2': p2_full['r2'],
            'rmse': p2_full['rmse'],
            'rmse_wins_per_16game_season': p2_full['rmse_wins_per_season'],
            'data_file': PIPE2_IMPACT.name,
        },
        # Combined leakage test (chronological + split-then-impute)
        {
            'pipeline': 'Combined leakage test',
            'description': 'Split-then-impute, chronological cutoff=2014',
            'subset': 'train (year<=2014)',
            'n': c_train['n'],
            'r2': c_train['r2'],
            'rmse': c_train['rmse'],
            'rmse_wins_per_16game_season': c_train['rmse_wins_per_season'],
            'data_file': COMBINED_IMPACT.name,
        },
        {
            'pipeline': 'Combined leakage test',
            'description': 'Split-then-impute, chronological cutoff=2014',
            'subset': 'test (year>2014)',
            'n': c_test['n'],
            'r2': c_test['r2'],
            'rmse': c_test['rmse'],
            'rmse_wins_per_16game_season': c_test['rmse_wins_per_season'],
            'data_file': COMBINED_IMPACT.name,
        },
    ]

    df = pd.DataFrame(rows)
    # Round for readability
    for col in ['r2', 'rmse', 'rmse_wins_per_16game_season']:
        df[col] = df[col].round(4)

    df.to_csv(OUT_CSV, index=False)

    # Append a one-row hyperparameter summary to a sidecar CSV
    hp_path = OUT_CSV.with_name('pipeline_hyperparameters.csv')
    hp_df = pd.DataFrame([{
        'note': 'Fixed optimal hyperparameters used by both pipelines '
                '(from 200-iter RandomizedSearchCV, 5-fold CV).',
        **OPTIMAL_HP,
        'objective': 'reg:squarederror',
        'random_state': 42,
    }])
    hp_df.to_csv(hp_path, index=False)

    print(f'Wrote {OUT_CSV}')
    with pd.option_context('display.width', 200,
                           'display.max_colwidth', 60):
        print(df.to_string(index=False))
    print(f'\nHyperparameters: {hp_path}')


if __name__ == '__main__':
    main()
