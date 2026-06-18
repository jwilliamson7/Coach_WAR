"""
Leakage-free Ridge generalization benchmark.

The paper reports that a CV-tuned Ridge regression generalizes far less well than
XGBoost on held-out coaches (test R^2 ~ 0.15 vs ~0.43). This script produces that
Ridge number on EXACTLY the same basis as the XGBoost Pipeline 1 (validation)
result, so the comparison is apples-to-apples:

  * Same data: data/final/imputed_split_then_imputed.csv, where the SVD imputation
    was fit on the training coaches and applied to the held-out coaches separately
    (no test information leaks into the imputation). This is the same split-then-
    impute file the XGBoost validation pipeline uses.
  * Same partition: the coach-based 80/20 split recorded in
    imputed_split_then_imputed_assignments.csv (train 1338 / test 299).
  * Same feature set: 365 predictors with Approximate Value (AV) features excluded,
    matching the XGBoost default.

Contrast with analysis/ridge_coaching_comparison.py, which trains Ridge on the
full-data (impute-BEFORE-split) file imputed_final_data.csv. That script's Spearman
rho = 0.81 ranking-robustness check is valid (rankings are insensitive to the
imputation leakage), but its test R^2 (~0.66) is inflated by the same leakage we
removed from XGBoost and is therefore NOT a fair generalization benchmark.

Standardization is fit on the training subset only and applied to the test subset,
mirroring the leakage-free protocol. Ridge's alpha is selected by 5-fold CV on the
training subset.

Output: analysis/outputs/csv/ridge_generalization_metrics.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / 'data/final/imputed_split_then_imputed.csv'
ASSIGN = ROOT / 'data/final/imputed_split_then_imputed_assignments.csv'
XGB_METRICS = ROOT / 'data/final/pipeline_validation_metrics.csv'
OUT = ROOT / 'analysis/outputs/csv/ridge_generalization_metrics.csv'

METADATA_COLS = ['Team', 'Year']
TARGET_COL = 'Win_Pct'
GAMES_PER_SEASON = 16  # WAR is standardized to a 16-game season throughout the paper
ALPHA_GRID = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
CV_FOLDS = 5
RANDOM_STATE = 42


def load_features():
    """Load the split-then-impute data, join the train/test assignment, and
    return (X_train, y_train, X_test, y_test, feature_cols)."""
    df = pd.read_csv(DATA)
    assign = pd.read_csv(ASSIGN)[['Team', 'Year', 'split']]
    df = df.merge(assign, on=['Team', 'Year'], how='left')

    n_missing = df['split'].isna().sum()
    if n_missing:
        raise ValueError(f'{n_missing} rows have no train/test assignment; '
                         'data and assignment files are out of sync.')

    drop = METADATA_COLS + [TARGET_COL, 'split']
    feature_cols = [c for c in df.columns if c not in drop]
    av_cols = [c for c in feature_cols if 'AV' in c or 'Approximate_Value' in c]
    feature_cols = [c for c in feature_cols if c not in av_cols]
    print(f'Features: {len(feature_cols)} (excluded {len(av_cols)} AV features)')

    tr = df['split'] == 'train'
    te = df['split'] == 'test'
    print(f'Split (coach-based 80/20): train {tr.sum()} / test {te.sum()}')

    X_train = df.loc[tr, feature_cols].values
    X_test = df.loc[te, feature_cols].values
    y_train = df.loc[tr, TARGET_COL].values
    y_test = df.loc[te, TARGET_COL].values
    return X_train, y_train, X_test, y_test, feature_cols


def tune_and_evaluate(X_train, y_train, X_test, y_test):
    """Fit the scaler on train only, CV-tune Ridge alpha on train, evaluate on test."""
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    search = GridSearchCV(
        Ridge(random_state=RANDOM_STATE),
        {'alpha': ALPHA_GRID},
        cv=CV_FOLDS,
        scoring='r2',
    )
    search.fit(X_train_s, y_train)
    model = search.best_estimator_

    def metrics(X, y):
        pred = model.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        return float(r2_score(y, pred)), rmse

    train_r2, train_rmse = metrics(X_train_s, y_train)
    test_r2, test_rmse = metrics(X_test_s, y_test)
    return {
        'best_alpha': search.best_params_['alpha'],
        'cv_r2': float(search.best_score_),
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
    }


def xgboost_test_r2():
    """Read XGBoost Pipeline 1 (validation) held-out test R^2 for comparison."""
    try:
        m = pd.read_csv(XGB_METRICS)
        row = m[(m['pipeline'].str.startswith('Pipeline 1')) & (m['subset'] == 'test')]
        if len(row):
            return float(row.iloc[0]['r2'])
    except (FileNotFoundError, KeyError):
        pass
    return None


def main():
    print('=' * 78)
    print('LEAKAGE-FREE RIDGE GENERALIZATION BENCHMARK')
    print('=' * 78)
    X_train, y_train, X_test, y_test, feature_cols = load_features()
    res = tune_and_evaluate(X_train, y_train, X_test, y_test)

    res['test_rmse_wins_per_16game_season'] = res['test_rmse'] * GAMES_PER_SEASON
    xgb_r2 = xgboost_test_r2()

    print('\n' + '-' * 78)
    print(f"Best alpha (5-fold CV): {res['best_alpha']}  (CV R^2 = {res['cv_r2']:.4f})")
    print(f"Ridge train R^2:        {res['train_r2']:.4f}")
    print(f"Ridge TEST R^2:         {res['test_r2']:.4f}  "
          f"(RMSE {res['test_rmse']:.4f} = "
          f"{res['test_rmse_wins_per_16game_season']:.2f} wins/16-game season)")
    if xgb_r2 is not None:
        print(f"XGBoost Pipeline 1 TEST R^2 (for comparison): {xgb_r2:.4f}")
        print(f"\nGeneralization gap: XGBoost {xgb_r2:.2f} vs Ridge {res['test_r2']:.2f} "
              "on identical held-out coaches.")
    print('-' * 78)

    out = pd.DataFrame([{
        'model': 'Ridge (CV-tuned, leakage-free)',
        'basis': 'split-then-impute, coach-based 80/20, 365 feats (AV excluded)',
        'n_train': len(y_train),
        'n_test': len(y_test),
        **res,
        'xgboost_pipeline1_test_r2': xgb_r2,
    }])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f'\nSaved: {OUT}')
    print('=' * 78)


if __name__ == '__main__':
    main()
