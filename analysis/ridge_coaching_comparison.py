"""
Ridge Regression Coaching Impact Analysis
Compares regularized linear regression to XGBoost approach
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
from datetime import datetime

def load_and_prepare_data(data_path, exclude_av=False):
    """Load data and separate features from target."""
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Separate features and target
    metadata_cols = ['Team', 'Year']
    target_col = 'Win_Pct'

    # Get feature columns (everything except metadata and target)
    feature_cols = [col for col in df.columns if col not in metadata_cols + [target_col]]

    # Optionally exclude AV features
    if exclude_av:
        av_cols = [col for col in feature_cols if 'AV' in col or 'Approximate_Value' in col]
        if av_cols:
            feature_cols = [col for col in feature_cols if col not in av_cols]
            df = df.drop(columns=av_cols)
            print(f"Excluded {len(av_cols)} AV features")

    X = df[feature_cols].values
    y = df[target_col].values
    team_year_info = df[metadata_cols].copy()

    print(f"Data shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y, team_year_info, feature_cols

def identify_coaching_features(feature_cols):
    """Identify which features are coaching-related."""
    coaching_suffixes = ['__oc_Norm', '__dc_Norm', '__hc_Norm', '__opp__hc_Norm']
    coaching_features = ['num_times_hc', 'num_yr_col_pos', 'num_yr_col_coor',
                         'num_yr_col_hc', 'num_yr_nfl_pos', 'num_yr_nfl_coor', 'num_yr_nfl_hc']

    coaching_cols = []
    for i, col in enumerate(feature_cols):
        if any(col.endswith(suffix) for suffix in coaching_suffixes):
            coaching_cols.append(i)
        elif col in coaching_features:
            coaching_cols.append(i)

    print(f"\nFound {len(coaching_cols)} coaching-related features")
    return coaching_cols

def calculate_replacement_level(X, team_year_info, coaching_cols, feature_cols):
    """Calculate replacement-level (median) values for coaching features."""
    print("\nCalculating replacement-level coach features...")

    # Add coach identifier to team_year_info
    df_with_features = team_year_info.copy()
    for i, col in enumerate(feature_cols):
        df_with_features[col] = X[:, i]

    # Calculate median across all coaches for each coaching feature
    replacement_values = np.zeros(len(coaching_cols))
    for idx, col_idx in enumerate(coaching_cols):
        col_name = feature_cols[col_idx]
        replacement_values[idx] = df_with_features[col_name].median()

    print(f"Calculated replacement values for {len(coaching_cols)} features")

    # Create replacement dataset
    X_replacement = X.copy()
    for idx, col_idx in enumerate(coaching_cols):
        X_replacement[:, col_idx] = replacement_values[idx]

    return X_replacement

def coach_based_split(team_year_info, test_size=0.2, random_state=42):
    """Split data by coaches to prevent data leakage."""
    # Get unique coaches (assuming coach name is in Team-Year pattern or we need to load it)
    # For simplicity, we'll do a random coach-based split
    # This is a simplified version - the full version would need coach names

    print("\nPerforming coach-based split...")
    print(f"Note: Using simplified year-based split as proxy for coach-based split")

    np.random.seed(random_state)
    years = team_year_info['Year'].unique()
    test_years = np.random.choice(years, size=int(len(years) * test_size), replace=False)

    train_mask = ~team_year_info['Year'].isin(test_years)
    test_mask = team_year_info['Year'].isin(test_years)

    train_idx = team_year_info[train_mask].index
    test_idx = team_year_info[test_mask].index

    print(f"Training samples: {len(train_idx)} ({100*len(train_idx)/len(team_year_info):.1f}%)")
    print(f"Test samples: {len(test_idx)} ({100*len(test_idx)/len(team_year_info):.1f}%)")

    return train_idx, test_idx

def train_ridge_model(X_train, y_train):
    """Train Ridge regression with hyperparameter tuning."""
    print("\nTraining Ridge regression with hyperparameter tuning...")

    # Standardize features for Ridge
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define parameter grid for Ridge
    param_dist = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
    }

    ridge = Ridge(random_state=42)

    # RandomizedSearchCV with fewer iterations for speed
    random_search = RandomizedSearchCV(
        ridge,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train_scaled, y_train)

    print(f"\nBest CV R² score: {random_search.best_score_:.4f}")
    print("Best parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    return random_search.best_estimator_, scaler

def evaluate_model(model, scaler, X_train, y_train, X_test, y_test):
    """Evaluate model performance on train and test sets."""
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("\n" + "="*60)
    print("TRAIN/TEST PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Metric':<12} {'Training':<15} {'Test':<15} {'Difference':<15}")
    print("-"*60)
    print(f"{'R²':<12} {train_r2:<15.4f} {test_r2:<15.4f} {train_r2-test_r2:<15.4f}")
    print(f"{'RMSE':<12} {train_rmse:<15.4f} {test_rmse:<15.4f} {test_rmse-train_rmse:<15.4f}")
    print(f"{'MAE':<12} {train_mae:<15.4f} {test_mae:<15.4f} {test_mae-train_mae:<15.4f}")

    if train_r2 - test_r2 > 0.2:
        print("\nWARNING: Possible overfitting detected (R² difference: {:.4f})".format(train_r2 - test_r2))

    return train_r2, test_r2

def analyze_coaching_impact(model, scaler, X, X_replacement, y, team_year_info):
    """Analyze coaching impact by comparing actual vs replacement predictions."""
    print("\nAnalyzing coaching impact...")

    X_scaled = scaler.transform(X)
    X_replacement_scaled = scaler.transform(X_replacement)

    y_pred_actual = model.predict(X_scaled)
    y_pred_replacement = model.predict(X_replacement_scaled)

    coaching_war = y - y_pred_replacement  # Actual win% - Replacement prediction

    print("\nCoaching WAR Statistics:")
    print(f"Mean coaching WAR: {coaching_war.mean():.4f}")
    print(f"Median coaching WAR: {np.median(coaching_war):.4f}")
    print(f"Std deviation: {coaching_war.std():.4f}")
    print(f"Min WAR: {coaching_war.min():.4f}")
    print(f"Max WAR: {coaching_war.max():.4f}")

    # Convert to games (16-game season)
    war_games = coaching_war * 16
    print(f"\nIn 16-game season terms:")
    print(f"Mean: {war_games.mean():.2f} games")
    print(f"Median: {np.median(war_games):.2f} games")

    # Create results dataframe
    results_df = team_year_info.copy()
    results_df['Actual_Win_Pct'] = y
    results_df['Predicted_Replacement'] = y_pred_replacement
    results_df['Coaching_WAR'] = coaching_war
    results_df['Coaching_WAR_Games'] = war_games

    return results_df, war_games

def compare_war_estimates(ridge_results):
    """Compare Ridge WAR estimates to XGBoost WAR estimates."""
    print("\n" + "="*80)
    print("WAR ESTIMATE CORRELATION ANALYSIS")
    print("="*80)

    # Load XGBoost results
    try:
        xgb_results = pd.read_csv('data/final/coaching_impact_analysis.csv')

        # Check for duplicates
        ridge_dupes = ridge_results.duplicated(['Team', 'Year']).sum()
        xgb_dupes = xgb_results.duplicated(['Team', 'Year']).sum()

        if ridge_dupes > 0 or xgb_dupes > 0:
            print(f"\nNote: Found {ridge_dupes} duplicate Team-Year in Ridge, {xgb_dupes} in XGBoost")
            print("These represent mid-season coaching changes. Averaging WAR values for each team-year.")

            # Average WAR values for duplicate Team-Years
            ridge_results = ridge_results.groupby(['Team', 'Year'], as_index=False).agg({
                'Actual_Win_Pct': 'first',  # Same for all duplicates
                'Predicted_Replacement': 'mean',  # Average prediction
                'Coaching_WAR': 'mean',  # Average WAR
                'Coaching_WAR_Games': 'mean'  # Average WAR in games
            })

            xgb_results = xgb_results.groupby(['Team', 'Year'], as_index=False).agg({
                'Coaching_WAR': 'mean'  # Average WAR
            })

        # Rename XGBoost column before merge to avoid conflict
        xgb_results_renamed = xgb_results[['Team', 'Year', 'Coaching_WAR']].copy()
        xgb_results_renamed = xgb_results_renamed.rename(columns={'Coaching_WAR': 'XGB_WAR'})

        # Merge on Team and Year
        merged = ridge_results.merge(
            xgb_results_renamed,
            on=['Team', 'Year'],
            how='inner'
        )

        # Rename Ridge column for clarity
        merged = merged.rename(columns={'Coaching_WAR_Games': 'Ridge_WAR_Games'})

        # Convert XGBoost WAR to games
        merged['XGB_WAR_Games'] = merged['XGB_WAR'] * 16

        # Calculate correlation
        from scipy.stats import pearsonr, spearmanr

        pearson_r, pearson_p = pearsonr(merged['Ridge_WAR_Games'], merged['XGB_WAR_Games'])
        spearman_r, spearman_p = spearmanr(merged['Ridge_WAR_Games'], merged['XGB_WAR_Games'])

        print(f"\nCorrelation between Ridge and XGBoost WAR estimates:")
        print(f"  Sample size: {len(merged)} team-seasons")
        print(f"  Pearson correlation: {pearson_r:.4f} (p={pearson_p:.4e})")
        print(f"  Spearman correlation: {spearman_r:.4f} (p={spearman_p:.4e})")

        # Calculate mean absolute difference
        mad = np.abs(merged['Ridge_WAR_Games'] - merged['XGB_WAR_Games']).mean()
        print(f"  Mean absolute difference: {mad:.2f} games")

        # Calculate percentage agreement on direction (positive vs negative)
        same_sign = ((merged['Ridge_WAR_Games'] > 0) == (merged['XGB_WAR_Games'] > 0)).sum()
        pct_agreement = 100 * same_sign / len(merged)
        print(f"  Agreement on direction (±): {pct_agreement:.1f}%")

        # Show top disagreements
        merged['Difference'] = merged['XGB_WAR_Games'] - merged['Ridge_WAR_Games']
        merged['Abs_Difference'] = np.abs(merged['Difference'])

        print("\nTop 5 Disagreements (XGBoost higher than Ridge):")
        top_diff = merged.nlargest(5, 'Difference')[['Team', 'Year', 'Ridge_WAR_Games', 'XGB_WAR_Games', 'Difference']]
        for _, row in top_diff.iterrows():
            print(f"  {row['Team']} {int(row['Year'])}: Ridge={row['Ridge_WAR_Games']:+.2f}, XGB={row['XGB_WAR_Games']:+.2f}, Diff={row['Difference']:+.2f}")

        print("\nTop 5 Disagreements (Ridge higher than XGBoost):")
        bottom_diff = merged.nsmallest(5, 'Difference')[['Team', 'Year', 'Ridge_WAR_Games', 'XGB_WAR_Games', 'Difference']]
        for _, row in bottom_diff.iterrows():
            print(f"  {row['Team']} {int(row['Year'])}: Ridge={row['Ridge_WAR_Games']:+.2f}, XGB={row['XGB_WAR_Games']:+.2f}, Diff={row['Difference']:+.2f}")

        return merged

    except FileNotFoundError:
        print("\nWARNING: Could not load XGBoost results for comparison.")
        print("XGBoost results file not found: data/final/coaching_impact_analysis.csv")
        return None

def compare_to_xgboost():
    """Display XGBoost results for comparison."""
    print("\n" + "="*60)
    print("COMPARISON TO XGBOOST RESULTS")
    print("="*60)
    print("\nXGBoost Performance (from log):")
    print("  Train R²: 0.8646")
    print("  Test R²: 0.6383")
    print("  Gap: 0.2264")
    print("  Mean WAR: 0.20 games")
    print("  Median WAR: 0.19 games")
    print("  Coaching feature importance: 49%")

def main():
    print("="*80)
    print("RIDGE REGRESSION COACHING IMPACT ANALYSIS")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Comparing regularized linear regression to XGBoost approach")
    print("="*80)

    # Load data (use same file as XGBoost, exclude AV features)
    data_path = 'data/final/imputed_final_data.csv'
    exclude_av = True  # Match XGBoost default behavior
    X, y, team_year_info, feature_cols = load_and_prepare_data(data_path, exclude_av=exclude_av)

    # Identify coaching features
    coaching_cols = identify_coaching_features(feature_cols)

    # Calculate replacement level
    X_replacement = calculate_replacement_level(X, team_year_info, coaching_cols, feature_cols)

    # Split data
    train_idx, test_idx = coach_based_split(team_year_info, test_size=0.2, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train model
    model, scaler = train_ridge_model(X_train, y_train)

    # Evaluate model
    train_r2, test_r2 = evaluate_model(model, scaler, X_train, y_train, X_test, y_test)

    # Analyze coaching impact on full dataset
    results_df, war_games = analyze_coaching_impact(model, scaler, X, X_replacement, y, team_year_info)

    # Save results
    output_path = 'analysis/outputs/ridge_coaching_impact_analysis.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Compare WAR estimates between Ridge and XGBoost
    merged = compare_war_estimates(results_df)

    # Compare to XGBoost
    compare_to_xgboost()

    print("\n" + "="*80)
    print("RIDGE REGRESSION SUMMARY")
    print("="*80)
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Gap: {train_r2 - test_r2:.4f}")
    print(f"Mean WAR: {war_games.mean():.2f} games")
    print(f"Median WAR: {np.median(war_games):.2f} games")

    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
