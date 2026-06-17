"""
Build leakage-free imputed dataset for sensitivity analysis.

Reviewer concern (#4): in the primary pipeline, SVD matrix completion is run on
the FULL dataset and then split into train/test for XGBoost tuning. This means
information from test rows could leak into the imputed values used for training.

This driver replicates the coach-based train/test split from
xgboost_coaching_impact_analysis.py (seed=42, test_size=0.2), then runs SVD
imputation SEPARATELY on each subset so neither one's imputation is influenced
by the other. StandardScaler is fit on the training imputed data and applied to
the test imputed data (standard ML practice).

Output: data/final/imputed_split_then_imputed.csv
        data/final/imputed_split_then_imputed_assignments.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svd_imputation import SVDImputer  # noqa: E402


RAW_DATA = Path('data/final/combined_final_dataset.csv')
COACH_DATA = Path('data/processed/Coaching/team_year_head_coaches.csv')

RANDOM_STATE = 42
N_COMPONENTS = 50


def out_paths(split_mode, test_size, cutoff):
    """Build output paths keyed by split mode."""
    if split_mode == 'coach':
        suffix = f'_t{int(round(test_size * 100)):02d}'
    else:  # chronological
        suffix = f'_chrono{cutoff}'
    return (
        Path(f'data/final/imputed_split_then_imputed{suffix}.csv'),
        Path(f'data/final/imputed_split_then_imputed_assignments{suffix}.csv'),
    )


def coach_split(df, test_size):
    """Random split by Primary_Coach (matches primary analysis)."""
    print(f'\nLoading coach assignments from {COACH_DATA}...')
    coach_df = pd.read_csv(COACH_DATA)
    stratify_df = df[['Team', 'Year']].merge(
        coach_df[['Team', 'Year', 'Primary_Coach']],
        on=['Team', 'Year'],
        how='left',
    )
    coach_seasons = stratify_df.groupby('Primary_Coach').size()
    unique_coaches = coach_seasons.index.tolist()
    print(f'  Unique coaches: {len(unique_coaches)}')
    coaches_train, coaches_test = train_test_split(
        unique_coaches, test_size=test_size, random_state=RANDOM_STATE
    )
    print(f'  Train coaches: {len(coaches_train)}')
    print(f'  Test coaches:  {len(coaches_test)}')
    train_mask = (
        stratify_df['Primary_Coach'].isin(coaches_train)
        | stratify_df['Primary_Coach'].isna()
    )
    test_mask = stratify_df['Primary_Coach'].isin(coaches_test)
    return train_mask.values, test_mask.values, stratify_df['Primary_Coach'].values


def chronological_split(df, cutoff):
    """Year-based split: train on Year <= cutoff, test on Year > cutoff."""
    train_mask = (df['Year'] <= cutoff).values
    test_mask = (df['Year'] > cutoff).values
    print(f'\nChronological split at year {cutoff}:')
    print(f'  Train years: {int(df["Year"].min())}-{cutoff}')
    print(f'  Test years:  {cutoff + 1}-{int(df["Year"].max())}')
    return train_mask, test_mask, np.array([None] * len(df))


def main():
    parser = argparse.ArgumentParser(
        description='Build leakage-free imputed dataset (split-then-impute).'
    )
    parser.add_argument('--split-mode', choices=['coach', 'chronological'],
                        default='coach',
                        help='How to split before imputing (default: coach)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of coaches assigned to test '
                             '(coach mode only, default: 0.2)')
    parser.add_argument('--cutoff', type=int, default=2014,
                        help='Year cutoff for chronological split '
                             '(chronological mode only, default: 2014)')
    args = parser.parse_args()

    OUT_DATA, OUT_ASSIGNMENTS = out_paths(
        args.split_mode, args.test_size, args.cutoff
    )
    print('=' * 80)
    print('BUILD SPLIT-THEN-IMPUTE DATASET (imputation leakage sensitivity)')
    print(f'Split mode: {args.split_mode}')
    if args.split_mode == 'coach':
        print(f'Test size: {args.test_size}')
    else:
        print(f'Cutoff year: {args.cutoff}')
    print('=' * 80)

    # --- Load raw data ---
    print(f'\nLoading raw data from {RAW_DATA}...')
    df = pd.read_csv(RAW_DATA)
    print(f'  Raw shape: {df.shape}')

    initial = len(df)
    df = df[df['Win_Pct'].notna()].reset_index(drop=True)
    print(f'  After dropping missing Win_Pct: {len(df)} rows '
          f'({initial - len(df)} dropped)')

    # --- Compute split based on mode ---
    if args.split_mode == 'coach':
        train_mask, test_mask, primary_coach = coach_split(df, args.test_size)
    else:
        train_mask, test_mask, primary_coach = chronological_split(df, args.cutoff)

    print(f'  Train rows: {train_mask.sum()}')
    print(f'  Test rows:  {test_mask.sum()}')
    assert (train_mask & test_mask).sum() == 0, 'Train and test masks overlap'
    assert (train_mask | test_mask).sum() == len(df), 'Some rows unassigned'

    # --- Identify feature columns ---
    feature_cols = [c for c in df.columns if c not in ['Team', 'Year', 'Win_Pct']]
    norm_cols = [c for c in feature_cols if c.endswith('_Norm')]
    non_norm_cols = [c for c in feature_cols if not c.endswith('_Norm')]
    print(f'\nFeatures: {len(feature_cols)} total '
          f'({len(norm_cols)} normalized, {len(non_norm_cols)} non-normalized)')

    # --- SVD impute train and test SEPARATELY ---
    X_train_raw = df.loc[train_mask, feature_cols].copy()
    X_test_raw = df.loc[test_mask, feature_cols].copy()

    train_missing = X_train_raw.isnull().sum().sum()
    test_missing = X_test_raw.isnull().sum().sum()
    print(f'  Train missing values: {train_missing:,} '
          f'({train_missing / X_train_raw.size * 100:.1f}%)')
    print(f'  Test missing values:  {test_missing:,} '
          f'({test_missing / X_test_raw.size * 100:.1f}%)')

    print(f'\n--- Imputing TRAIN set ({X_train_raw.shape}) ---')
    train_imputer = SVDImputer(n_components=N_COMPONENTS, verbose=True)
    X_train_imputed = train_imputer.fit_transform(X_train_raw)

    print(f'\n--- Imputing TEST set ({X_test_raw.shape}) ---')
    test_imputer = SVDImputer(n_components=N_COMPONENTS, verbose=True)
    X_test_imputed = test_imputer.fit_transform(X_test_raw)

    # --- Standardize non-normalized features: fit on train, apply to test ---
    print(f'\n--- Standardizing {len(non_norm_cols)} non-normalized features '
          '(fit on train, apply to test) ---')
    scaler = StandardScaler()
    X_train_imputed[non_norm_cols] = scaler.fit_transform(
        X_train_imputed[non_norm_cols]
    )
    X_test_imputed[non_norm_cols] = scaler.transform(
        X_test_imputed[non_norm_cols]
    )

    # --- Recombine into single DataFrame, preserving original row order ---
    print('\n--- Recombining train + test back into single DataFrame ---')
    df_imputed = df.copy()
    df_imputed.loc[train_mask, feature_cols] = X_train_imputed.values
    df_imputed.loc[test_mask, feature_cols] = X_test_imputed.values

    missing_after = df_imputed[feature_cols].isnull().sum().sum()
    print(f'  Missing values after imputation: {missing_after}')
    assert missing_after == 0, 'Missing values remain after imputation'

    # --- Save ---
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    df_imputed.to_csv(OUT_DATA, index=False)
    print(f'\n  Saved imputed data: {OUT_DATA}')
    print(f'  Final shape: {df_imputed.shape}')

    # Save split assignments as a separate file so the XGBoost test script can
    # verify the split is reproduced identically.
    assignments = df[['Team', 'Year']].copy()
    assignments['split'] = np.where(train_mask, 'train',
                                    np.where(test_mask, 'test', 'unassigned'))
    assignments['Primary_Coach'] = primary_coach
    assignments.to_csv(OUT_ASSIGNMENTS, index=False)
    print(f'  Saved split assignments: {OUT_ASSIGNMENTS}')

    print('\n' + '=' * 80)
    print('DONE')
    print('=' * 80)


if __name__ == '__main__':
    main()
