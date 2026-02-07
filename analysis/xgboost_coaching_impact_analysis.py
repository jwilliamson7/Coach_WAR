"""
XGBoost Coaching Impact Analysis Script
Compares XGBoost predictions using actual coach features versus replacement-level (average) coach features.
This quantifies the impact of individual coaching characteristics on team performance.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import uniform, randint, t as t_dist
import argparse
import warnings
import sys
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class TeeOutput:
    """Tee output to both stdout and a file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

def load_and_prepare_data(filepath, exclude_av=False):
    """Load the combined dataset and prepare features."""
    print("Loading data...")
    df = pd.read_csv(filepath)

    # Ensure Win_Pct is the target
    if 'Win_Pct' not in df.columns:
        raise ValueError("Win_Pct column not found in dataset")

    # Separate features and target
    X = df.drop(['Win_Pct'], axis=1)
    y = df['Win_Pct']

    # Optionally exclude AV (Approximate Value) features
    if exclude_av:
        av_cols = [col for col in X.columns if 'AV' in col or 'Approximate_Value' in col]
        if av_cols:
            X = X.drop(columns=av_cols)
            print(f"Excluded {len(av_cols)} AV features")
    
    # Store team and year for later analysis
    if 'Team' in X.columns and 'Year' in X.columns:
        team_year_info = df[['Team', 'Year']].copy()
        # Remove Team and Year from features (metadata only, not predictors)
        if X['Team'].dtype == 'object':
            X = X.drop(['Team', 'Year'], axis=1)
        else:
            # If Team was already numeric (shouldn't happen), just drop Year
            X = X.drop(['Year'], axis=1)
    else:
        team_year_info = pd.DataFrame(index=df.index)
    
    # Convert any remaining object columns to numeric if possible
    for col in X.select_dtypes(include=['object']).columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            # If conversion fails, drop the column
            X = X.drop([col], axis=1)
            print(f"Dropped non-numeric column: {col}")
    
    # Handle missing values
    X = X.fillna(0)
    
    print(f"Data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, team_year_info, df

def identify_coach_features(X):
    """Identify coaching-related features in the dataset."""
    print("\nIdentifying coaching features...")
    
    coach_features = []
    
    # 1. Coaching performance metrics (normalized features from coaching data)
    coaching_suffixes = ['_oc_Norm', '_dc_Norm', '_hc_Norm', '_opp__oc_Norm', '_opp__dc_Norm', '_opp__hc_Norm']
    
    for col in X.columns:
        # Check if column ends with any coaching suffix
        if any(col.endswith(suffix) for suffix in coaching_suffixes):
            coach_features.append(col)
    
    # 2. Coach tenure and experience metrics
    # These are characteristics of the coaching hire itself
    tenure_patterns = [
        'num_times_hc',      # Number of times as head coach
        'num_yr_col_pos',    # Years of college position coaching
        'num_yr_col_coor',   # Years of college coordinator experience  
        'num_yr_col_hc',     # Years of college head coaching
        'num_yr_nfl_pos',    # Years of NFL position coaching
        'num_yr_nfl_coor',   # Years of NFL coordinator experience
        'num_yr_nfl_hc'      # Years of NFL head coaching
    ]
    
    for col in X.columns:
        # Add exact matches for tenure/experience columns
        if col in tenure_patterns and col not in coach_features:
            coach_features.append(col)
    
    # 3. Also check for any other coach-specific columns that might exist
    # But exclude player experience metrics
    additional_patterns = ['Head_Coach', 'Coordinator', 'HC_Experience', 'OC_Experience', 'DC_Experience']
    
    for col in X.columns:
        if any(pattern in col for pattern in additional_patterns):
            # Exclude player/roster/starter experience metrics
            if not any(x in col.lower() for x in ['roster', 'starter', 'avg_', 'stddev_']):
                if col not in coach_features:  # Avoid duplicates
                    coach_features.append(col)
    
    print(f"Found {len(coach_features)} coaching-related features")
    
    # Display breakdown of feature types
    performance_features = [f for f in coach_features if any(f.endswith(s) for s in coaching_suffixes)]
    tenure_features = [f for f in coach_features if f in tenure_patterns]
    
    print(f"  - Performance metrics: {len(performance_features)}")
    print(f"  - Tenure/experience metrics: {len(tenure_features)}")
    
    # Display sample of identified features
    if len(coach_features) > 0:
        print("\nSample coaching features identified:")
        for feat in coach_features[:10]:
            print(f"  - {feat}")
        if len(coach_features) > 10:
            print(f"  ... and {len(coach_features) - 10} more")
    
    return coach_features

def calculate_replacement_features(X, coach_features, team_year_info):
    """Calculate average (replacement-level) values for coach features using coach-level averaging."""
    print("\nCalculating replacement-level coach features...")
    print("Using coach-level averaging (each coach weighted equally regardless of tenure)...")
    
    # Load coach data to group by coach
    try:
        coach_df = pd.read_csv('data/processed/Coaching/team_year_head_coaches.csv')

        # Start with team_year_info and add features
        if 'Team' in team_year_info.columns and 'Year' in team_year_info.columns:
            combined_df = team_year_info.reset_index(drop=True).copy()
        else:
            print("Warning: No Team/Year columns in team_year_info. Cannot group by coach.")
            raise ValueError("Missing Team/Year columns")

        # Add coaching features to the dataframe
        for feature in coach_features:
            if feature in X.columns:
                combined_df[feature] = X[feature].values

        # Merge with coach information
        combined_df = combined_df.merge(
            coach_df[['Team', 'Year', 'Primary_Coach']],
            on=['Team', 'Year'],
            how='left'
        )
        
        # Remove rows with missing coach information
        before_count = len(combined_df)
        combined_df = combined_df.dropna(subset=['Primary_Coach'])
        after_count = len(combined_df)
        print(f"Found coach data for {after_count} of {before_count} team-years")
        
    except Exception as e:
        print(f"Warning: Could not load coach data ({e}). Falling back to team-year median.")
        replacement_values = {}
        for feature in coach_features:
            if feature in X.columns:
                replacement_values[feature] = X[feature].median()
        return replacement_values
    
    # Calculate coach career averages for each feature
    replacement_values = {}
    for feature in coach_features:
        if feature in combined_df.columns:
            # Group by coach and calculate mean for each coach
            coach_averages = combined_df.groupby('Primary_Coach')[feature].mean()
            
            # Take median of coach averages (each coach weighted equally)
            replacement_values[feature] = coach_averages.median()
            
            print(f"  {feature}: {len(coach_averages)} coaches, replacement = {replacement_values[feature]:.3f}")
    
    print(f"\nCalculated replacement values for {len(replacement_values)} features")
    print("Each coach's career average was weighted equally in replacement calculation")
    
    # Show sample of replacement values
    print("\nSample replacement values:")
    sample_features = list(replacement_values.keys())[:5]
    for feat in sample_features:
        print(f"  {feat}: {replacement_values[feat]:.3f}")
    
    return replacement_values

def create_replacement_dataset(X, coach_features, replacement_values):
    """Create dataset with coach features replaced by average values."""
    print("\nCreating replacement-level dataset...")
    
    # Create a copy of the original data
    X_replacement = X.copy()
    
    # Replace coaching features with average values
    for feature in coach_features:
        if feature in X_replacement.columns:
            X_replacement[feature] = replacement_values[feature]
    
    print(f"Replaced {len(coach_features)} coaching features with replacement-level values")
    
    return X_replacement

def train_and_predict(X, y, team_year_info, use_tuning=True, cv_folds=5, n_iter=50, test_size=0.2, random_state=42):
    """Train XGBoost model with coach-based stratification to prevent data leakage."""
    
    # Load coach data for stratification
    try:
        coach_df = pd.read_csv('data/processed/Coaching/team_year_head_coaches.csv')
        
        # Create combined dataframe with coach information
        if 'Team' in team_year_info.columns and 'Year' in team_year_info.columns:
            stratify_df = team_year_info.reset_index(drop=True).copy()
            stratify_df = stratify_df.merge(
                coach_df[['Team', 'Year', 'Primary_Coach']], 
                on=['Team', 'Year'], 
                how='left'
            )
            
            # Get unique coaches and their total seasons
            coach_seasons = stratify_df.groupby('Primary_Coach').size()
            print(f"\nFound {len(coach_seasons)} unique coaches")
            print(f"Coach season distribution: min={coach_seasons.min()}, max={coach_seasons.max()}, mean={coach_seasons.mean():.1f}")
            
            # Assign coaches to train/test sets (not individual seasons)
            unique_coaches = coach_seasons.index.tolist()
            coaches_train, coaches_test = train_test_split(
                unique_coaches, test_size=test_size, random_state=random_state
            )
            
            # Create train/test masks based on coach assignments
            train_mask = stratify_df['Primary_Coach'].isin(coaches_train)
            test_mask = stratify_df['Primary_Coach'].isin(coaches_test)
            
            # Handle rows with missing coach data (assign to train)
            missing_coach_mask = stratify_df['Primary_Coach'].isna()
            train_mask = train_mask | missing_coach_mask
            
            print(f"Coach-based split:")
            print(f"  Training coaches: {len(coaches_train)} (seasons: {train_mask.sum()})")
            print(f"  Test coaches: {len(coaches_test)} (seasons: {test_mask.sum()})")
            print(f"  Missing coach data: {missing_coach_mask.sum()} (assigned to train)")
            
        else:
            print("Warning: No Team/Year columns available. Using random split.")
            raise ValueError("Missing Team/Year for coach stratification")
            
    except Exception as e:
        print(f"Warning: Could not perform coach-based stratification ({e}). Using random split.")
        # Fallback to random split
        train_mask = np.random.RandomState(42).random(len(X)) >= test_size
        test_mask = ~train_mask
    
    # Apply masks to create train/test sets
    # Reset index to ensure alignment
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    X_train = X_reset[train_mask].copy()
    X_test = X_reset[test_mask].copy()
    y_train = y_reset[train_mask].copy()
    y_test = y_reset[test_mask].copy()
    
    print(f"\nFinal split sizes:")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    if use_tuning:
        print(f"\nPerforming hyperparameter tuning with RandomizedSearchCV...")
        print(f"CV folds: {cv_folds}, Iterations: {n_iter}")
        
        # Define hyperparameter search space with discrete options
        param_dist = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
            'max_depth': [2, 3, 4, 5],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1.0, 1.5, 2.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0, 1.5, 2.0],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        
        # Base model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=random_state,
            verbosity=0,
            n_jobs=-1
        )

        # RandomizedSearchCV on training data only
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            random_state=random_state,
            verbose=1
        )
        
        # Fit the random search on training data
        random_search.fit(X_train, y_train)

        # Get best model
        model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = random_search.best_score_

        print(f"\nBest CV R² score (training): {best_score:.4f}")
        print("Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

        # Print detailed CV fold results for best model
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION FOLD PERFORMANCE (BEST MODEL)")
        print(f"{'='*80}")

        # Get the index of the best model in cv_results_
        best_idx = random_search.best_index_

        # Extract fold scores for the best model
        fold_scores = []
        for fold_num in range(cv_folds):
            score_key = f'split{fold_num}_test_score'
            if score_key in random_search.cv_results_:
                fold_scores.append(random_search.cv_results_[score_key][best_idx])

        if fold_scores:
            print(f"\n{'Fold':<10} {'R² Score':<15}")
            print("-" * 25)
            for i, score in enumerate(fold_scores, 1):
                print(f"{'Fold ' + str(i):<10} {score:<15.4f}")
            print("-" * 25)
            print(f"{'Mean':<10} {np.mean(fold_scores):<15.4f}")
            print(f"{'Std Dev':<10} {np.std(fold_scores):<15.4f}")
            print(f"{'Min':<10} {np.min(fold_scores):<15.4f}")
            print(f"{'Max':<10} {np.max(fold_scores):<15.4f}")
        else:
            print("Could not extract individual fold scores from CV results")
            
    else:
        print("\nUsing default hyperparameters (no tuning)...")
        # Default hyperparameters (from 200-iteration RandomizedSearchCV, 5-fold CV)
        model_params = {
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
            'n_jobs': -1
        }
        
        # Create and train model on training data
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train, y_train)
    
    # Generate predictions for train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate performance metrics
    train_metrics = {
        'mse': mean_squared_error(y_train, y_train_pred),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred)
    }
    
    test_metrics = {
        'mse': mean_squared_error(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'r2': r2_score(y_test, y_test_pred)
    }
    
    # Print performance comparison
    print(f"\n{'='*60}")
    print("TRAIN/TEST PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'Training':<15} {'Test':<15} {'Difference':<15}")
    print("-" * 60)
    print(f"{'R²':<12} {train_metrics['r2']:<15.4f} {test_metrics['r2']:<15.4f} {test_metrics['r2'] - train_metrics['r2']:<15.4f}")
    print(f"{'RMSE':<12} {np.sqrt(train_metrics['mse']):<15.4f} {np.sqrt(test_metrics['mse']):<15.4f} {np.sqrt(test_metrics['mse']) - np.sqrt(train_metrics['mse']):<15.4f}")
    print(f"{'MAE':<12} {train_metrics['mae']:<15.4f} {test_metrics['mae']:<15.4f} {test_metrics['mae'] - train_metrics['mae']:<15.4f}")
    
    # Check for overfitting
    r2_diff = train_metrics['r2'] - test_metrics['r2']
    if r2_diff > 0.1:
        print(f"\nWARNING: Possible overfitting detected (R² difference: {r2_diff:.4f})")
    elif r2_diff > 0.05:
        print(f"\nCAUTION: Moderate train/test performance gap (R² difference: {r2_diff:.4f})")
    else:
        print(f"\nGood generalization (R² difference: {r2_diff:.4f})")
    
    # Make predictions on full dataset for coaching analysis
    y_pred_full = model.predict(X)
    
    return model, y_pred_full, train_metrics, test_metrics

def analyze_coaching_impact(y_true, y_pred_actual, y_pred_replacement, team_year_info, test_rmse=None):
    """Analyze the impact of coaching by comparing actual vs replacement predictions."""
    print("\nAnalyzing coaching impact...")

    # Compute global WAR SE from test set RMSE (converted to 16-game wins)
    if test_rmse is not None:
        war_se = test_rmse * 16  # Convert win percentage RMSE to 16-game season wins
        print(f"Model residual standard error: ±{war_se:.2f} wins per season (from test RMSE = {test_rmse:.4f})")
    else:
        war_se = np.nan

    # Create results dataframe
    # Key metric: Coaching WAR = Actual Win% - Replacement Prediction
    results = pd.DataFrame({
        'Team': team_year_info.get('Team', 'Unknown'),
        'Year': team_year_info.get('Year', 0),
        'Actual_Win_Pct': y_true.values,
        'Predicted_With_Coach': y_pred_actual,
        'Predicted_Replacement': y_pred_replacement,
        'Coaching_WAR': y_true.values - y_pred_replacement,  # Primary metric: Actual - Replacement prediction
        'Predicted_Impact': y_pred_actual - y_pred_replacement,  # Secondary metric: Model's predicted difference
        'Prediction_Error_Coach': y_true.values - y_pred_actual,
        'Prediction_Error_Replacement': y_true.values - y_pred_replacement,
        'WAR_SE': war_se  # Global model SE in wins (same for all rows)
    })
    
    # Load head coach data
    try:
        coach_df = pd.read_csv('data/processed/Coaching/team_year_head_coaches.csv')
        # Merge coach information
        results = results.merge(
            coach_df[['Team', 'Year', 'Primary_Coach', 'Combined_Coach']], 
            on=['Team', 'Year'], 
            how='left'
        )
    except:
        print("Warning: Could not load coach data")
        results['Primary_Coach'] = 'N/A'
        results['Combined_Coach'] = 'N/A'
    
    # Calculate percentiles for coaching WAR
    results['WAR_Percentile'] = results['Coaching_WAR'].rank(pct=True) * 100
    
    # Sort by coaching WAR (largest positive impact first)
    results = results.sort_values('Coaching_WAR', ascending=False)
    
    # Calculate statistics
    print(f"\nCoaching WAR Statistics:")
    print(f"Mean coaching WAR: {results['Coaching_WAR'].mean():.4f}")
    print(f"Median coaching WAR: {results['Coaching_WAR'].median():.4f}")
    print(f"Std deviation: {results['Coaching_WAR'].std():.4f}")
    print(f"Min WAR: {results['Coaching_WAR'].min():.4f}")
    print(f"Max WAR: {results['Coaching_WAR'].max():.4f}")

    # Convert to 16-game season terms for interpretability
    print(f"\nIn 16-game season terms:")
    print(f"Mean: {results['Coaching_WAR'].mean() * 16:.2f} games")
    print(f"Median: {results['Coaching_WAR'].median() * 16:.2f} games")

    print(f"\nPredicted Impact Statistics:")
    print(f"Mean predicted impact: {results['Predicted_Impact'].mean():.4f}")
    print(f"Median predicted impact: {results['Predicted_Impact'].median():.4f}")
    print(f"Std deviation: {results['Predicted_Impact'].std():.4f}")
    print(f"Min impact: {results['Predicted_Impact'].min():.4f}")
    print(f"Max impact: {results['Predicted_Impact'].max():.4f}")
    
    # Model performance comparison
    mse_actual = mean_squared_error(y_true, y_pred_actual)
    mse_replacement = mean_squared_error(y_true, y_pred_replacement)
    mae_actual = mean_absolute_error(y_true, y_pred_actual)
    mae_replacement = mean_absolute_error(y_true, y_pred_replacement)
    r2_actual = r2_score(y_true, y_pred_actual)
    r2_replacement = r2_score(y_true, y_pred_replacement)
    
    print(f"\n{'='*80}")
    print("MODEL PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Metric':<20} {'With Actual Coach':<20} {'With Replacement':<20} {'Difference':<15}")
    print("-" * 75)
    print(f"{'MSE':<20} {mse_actual:<20.6f} {mse_replacement:<20.6f} {mse_replacement - mse_actual:<15.6f}")
    print(f"{'MAE':<20} {mae_actual:<20.6f} {mae_replacement:<20.6f} {mae_replacement - mae_actual:<15.6f}")
    print(f"{'R² Score':<20} {r2_actual:<20.4f} {r2_replacement:<20.4f} {r2_replacement - r2_actual:<15.4f}")
    print(f"{'RMSE':<20} {np.sqrt(mse_actual):<20.6f} {np.sqrt(mse_replacement):<20.6f} {np.sqrt(mse_replacement) - np.sqrt(mse_actual):<15.6f}")

    # Identify coaches with highest positive WAR for saving to file
    threshold_percentile = 95
    threshold_war = 0.05

    high_impact_coaches = results[
        (results['WAR_Percentile'] >= threshold_percentile) |
        (results['Coaching_WAR'] > threshold_war)
    ].copy()

    # Show top 10 positive WAR
    print(f"\n{'='*80}")
    print(f"TOP 10 POSITIVE COACHING WAR")
    print(f"{'='*80}")
    
    top_10 = results.head(10)
    print(f"\n{'Team':<6} {'Year':<6} {'Coach':<25} {'Actual':<8} {'Replacement':<12} {'WAR':<8} {'Pred Impact'}")
    print("-" * 85)
    for idx, row in top_10.iterrows():
        if pd.isna(row['Primary_Coach']):
            coach_name = 'N/A'
        else:
            coach_name = row['Primary_Coach'][:23] if len(row['Primary_Coach']) > 23 else row['Primary_Coach']
        print(f"{row['Team']:<6} {int(row['Year']):<6} {coach_name:<25} {row['Actual_Win_Pct']:.3f}    "
              f"{row['Predicted_Replacement']:.3f}        {row['Coaching_WAR']:+.3f}    {row['Predicted_Impact']:+.3f}")
    
    # Show bottom 10 (negative WAR)
    print(f"\n{'='*80}")
    print(f"TOP 10 NEGATIVE COACHING WAR")
    print(f"{'='*80}")
    
    bottom_10 = results.tail(10).iloc[::-1]  # Reverse to show worst first
    print(f"\n{'Team':<6} {'Year':<6} {'Coach':<25} {'Actual':<8} {'Replacement':<12} {'WAR':<8} {'Pred Impact'}")
    print("-" * 85)
    for idx, row in bottom_10.iterrows():
        if pd.isna(row['Primary_Coach']):
            coach_name = 'N/A'
        else:
            coach_name = row['Primary_Coach'][:23] if len(row['Primary_Coach']) > 23 else row['Primary_Coach']
        print(f"{row['Team']:<6} {int(row['Year']):<6} {coach_name:<25} {row['Actual_Win_Pct']:.3f}    "
              f"{row['Predicted_Replacement']:.3f}        {row['Coaching_WAR']:+.3f}    {row['Predicted_Impact']:+.3f}")
    
    return results, high_impact_coaches

def analyze_coach_rankings(results):
    """Analyze coaching impact by individual coaches across their careers."""
    print(f"\n{'='*80}")
    print("COACH CAREER IMPACT ANALYSIS (WAR: Actual Win% - Replacement Prediction)")
    print(f"{'='*80}")

    # Group by coach and calculate statistics
    coach_stats = results.groupby('Primary_Coach').agg({
        'Coaching_WAR': ['mean', 'std', 'count', 'sum'],  # Primary WAR metric
        'Predicted_Impact': ['mean', 'sum'],  # Secondary metric (predicted difference)
        'Actual_Win_Pct': 'mean',
        'Predicted_With_Coach': 'mean',
        'Predicted_Replacement': 'mean'
    }).round(4)

    # Flatten column names
    coach_stats.columns = ['_'.join(col).strip() for col in coach_stats.columns.values]
    coach_stats = coach_stats.rename(columns={
        'Coaching_WAR_mean': 'Avg_WAR',
        'Coaching_WAR_std': 'WAR_StdDev',
        'Coaching_WAR_count': 'Seasons',
        'Coaching_WAR_sum': 'Total_WAR',
        'Predicted_Impact_mean': 'Avg_Pred_Impact',
        'Predicted_Impact_sum': 'Total_Pred_Impact',
        'Actual_Win_Pct_mean': 'Avg_Actual_Win',
        'Predicted_With_Coach_mean': 'Avg_Pred_Coach',
        'Predicted_Replacement_mean': 'Avg_Pred_Replace'
    })

    # Filter for coaches with at least 3 seasons
    coach_stats = coach_stats[coach_stats['Seasons'] >= 3]

    # Compute 95% confidence intervals for career average WAR
    # SE = std / sqrt(N), CI = mean ± t_{N-1, 0.025} × SE
    # WAR values are in win-percentage units; convert to 16-game wins for CI
    coach_stats['SE'] = (coach_stats['WAR_StdDev'] / np.sqrt(coach_stats['Seasons'])) * 16
    t_crit = coach_stats['Seasons'].apply(lambda n: t_dist.ppf(0.975, df=n - 1))
    coach_stats['CI_Lower'] = coach_stats['Avg_WAR'] * 16 - t_crit * coach_stats['SE']
    coach_stats['CI_Upper'] = coach_stats['Avg_WAR'] * 16 + t_crit * coach_stats['SE']

    # Round CI columns
    coach_stats['SE'] = coach_stats['SE'].round(2)
    coach_stats['CI_Lower'] = coach_stats['CI_Lower'].round(2)
    coach_stats['CI_Upper'] = coach_stats['CI_Upper'].round(2)

    # Sort by average WAR (actual vs replacement)
    coach_stats = coach_stats.sort_values('Avg_WAR', ascending=False)

    print(f"\nTop 15 Coaches by Average WAR (min 3 seasons):")
    print(f"\n{'Coach':<25} {'Avg WAR':<10} {'Seasons':<9} {'Total WAR':<11} {'95% CI':<18} {'Avg Win%'}")
    print("-" * 90)

    for coach, row in coach_stats.head(15).iterrows():
        if pd.notna(coach) and coach != 'N/A':
            coach_name = coach[:23] if len(coach) > 23 else coach
            avg_war_games = row['Avg_WAR'] * 16
            ci_str = f"[{row['CI_Lower']:+.1f}, {row['CI_Upper']:+.1f}]"
            print(f"{coach_name:<25} {avg_war_games:+.1f}      {int(row['Seasons']):<9} "
                  f"{row['Total_WAR'] * 16:+.1f}       {ci_str:<18} {row['Avg_Actual_Win']:.3f}")

    print(f"\nBottom 15 Coaches by Average WAR (min 3 seasons):")
    print(f"\n{'Coach':<25} {'Avg WAR':<10} {'Seasons':<9} {'Total WAR':<11} {'95% CI':<18} {'Avg Win%'}")
    print("-" * 90)

    for coach, row in coach_stats.tail(15).iterrows():
        if pd.notna(coach) and coach != 'N/A':
            coach_name = coach[:23] if len(coach) > 23 else coach
            avg_war_games = row['Avg_WAR'] * 16
            ci_str = f"[{row['CI_Lower']:+.1f}, {row['CI_Upper']:+.1f}]"
            print(f"{coach_name:<25} {avg_war_games:+.1f}      {int(row['Seasons']):<9} "
                  f"{row['Total_WAR'] * 16:+.1f}       {ci_str:<18} {row['Avg_Actual_Win']:.3f}")

    return coach_stats

def plot_feature_importance_comparison(model_actual, X_actual, coach_features, top_n=20):
    """Compare feature importance with focus on coaching features."""
    print(f"\n{'='*80}")
    print(f"FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Get feature importances
    importance = model_actual.feature_importances_
    features = X_actual.columns
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance,
        'Is_Coach_Feature': [f in coach_features for f in features]
    }).sort_values('Importance', ascending=False)
    
    # Calculate coaching feature statistics
    coach_importance = importance_df[importance_df['Is_Coach_Feature']]['Importance'].sum()
    total_importance = importance_df['Importance'].sum()
    coach_pct = (coach_importance / total_importance) * 100 if total_importance > 0 else 0
    
    print(f"\nCoaching Features Importance:")
    print(f"  Total importance of coaching features: {coach_importance:.4f}")
    print(f"  Percentage of total importance: {coach_pct:.2f}%")
    print(f"  Number of coaching features: {sum(importance_df['Is_Coach_Feature'])}")
    
    # Display top features with coaching indicator
    print(f"\nTop {top_n} Features by Importance:")
    print(f"\n{'Rank':<6} {'Feature':<50} {'Importance':<12} {'Coach Feature'}")
    print("-" * 80)
    
    for i, (idx, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
        coach_indicator = "Yes" if row['Is_Coach_Feature'] else ""
        print(f"{i:<6} {row['Feature']:<50} {row['Importance']:.6f}    {coach_indicator}")
    
    # Show top coaching features specifically
    top_coach_features = importance_df[importance_df['Is_Coach_Feature']].head(10)
    if len(top_coach_features) > 0:
        print(f"\n{'='*80}")
        print("TOP COACHING FEATURES BY IMPORTANCE")
        print(f"{'='*80}")
        print(f"\n{'Rank':<6} {'Feature':<50} {'Importance'}")
        print("-" * 70)
        
        for i, (idx, row) in enumerate(top_coach_features.iterrows(), 1):
            print(f"{i:<6} {row['Feature']:<50} {row['Importance']:.6f}")
    
    return importance_df

def save_results(results, high_impact_coaches, coach_stats, importance_df):
    """Save analysis results to CSV files."""
    print("\nSaving results...")
    
    # Save full results
    results_file = 'data/final/coaching_impact_analysis.csv'
    results.to_csv(results_file, index=False)
    print(f"Full results saved to: {results_file}")
    
    # Save high impact coaches
    if len(high_impact_coaches) > 0:
        high_impact_file = 'data/final/high_impact_coaches.csv'
        high_impact_coaches.to_csv(high_impact_file, index=False)
        print(f"High impact coaches saved to: {high_impact_file}")
    
    # Save coach career statistics
    coach_stats_file = 'data/final/coach_career_impact_stats.csv'
    coach_stats.to_csv(coach_stats_file)
    print(f"Coach career statistics saved to: {coach_stats_file}")
    
    # Save feature importance with coaching indicator
    importance_file = 'data/final/feature_importance_coaching_analysis.csv'
    importance_df.to_csv(importance_file, index=False)
    print(f"Feature importance saved to: {importance_file}")

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='XGBoost Coaching Impact Analysis')
    parser.add_argument('--with-av', action='store_true', 
                       help='Include AV (Approximate Value) features in analysis')
    parser.add_argument('--no-tuning', action='store_true',
                       help='Skip hyperparameter tuning and use default parameters')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds for tuning (default: 5)')
    parser.add_argument('--n-iter', type=int, default=50,
                       help='Number of iterations for RandomizedSearchCV (default: 50)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    args = parser.parse_args()
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'analysis/outputs/logs/coaching_analysis_log_{timestamp}.txt'
    
    # Set up output redirection to both console and file
    tee = TeeOutput(log_filename)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        # Use single imputed dataset file
        filepath = 'data/final/imputed_final_data.csv'
        exclude_av = not args.with_av  # Exclude AV features unless --with-av flag is used
        dataset_type = "with ALL features (including AV)" if args.with_av else "WITHOUT AV features"

        print("="*80)
        print("XGBOOST COACHING IMPACT ANALYSIS")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using dataset {dataset_type}")
        print("Comparing predictions with actual vs replacement-level coaching")
        if not args.no_tuning:
            print(f"Hyperparameter tuning: ENABLED ({args.n_iter} iterations, {args.cv_folds} CV folds)")
        else:
            print("Hyperparameter tuning: DISABLED (using default parameters)")
        print(f"Log file: {log_filename}")
        print("="*80)

        # Load and prepare data
        X, y, team_year_info, full_df = load_and_prepare_data(filepath, exclude_av=exclude_av)
        
        # Identify coaching features
        coach_features = identify_coach_features(X)
        
        # Calculate replacement-level values
        replacement_values = calculate_replacement_features(X, coach_features, team_year_info)
        
        # Create replacement dataset
        X_replacement = create_replacement_dataset(X, coach_features, replacement_values)
        
        # Train model with actual data
        print("\nTraining model with actual coaching data...")
        use_tuning = not args.no_tuning
        model_actual, y_pred_actual, train_metrics, test_metrics = train_and_predict(
            X, y, team_year_info, use_tuning=use_tuning, cv_folds=args.cv_folds, n_iter=args.n_iter, random_state=args.random_state)
        
        # Generate predictions with replacement-level coaching
        print("\nGenerating predictions with replacement-level coaching...")
        y_pred_replacement = model_actual.predict(X_replacement)
        
        # Analyze coaching impact (pass test RMSE for global model SE)
        test_rmse = np.sqrt(test_metrics['mse'])
        results, high_impact_coaches = analyze_coaching_impact(
            y, y_pred_actual, y_pred_replacement, team_year_info, test_rmse=test_rmse
        )
        
        # Analyze individual coach rankings
        coach_stats = analyze_coach_rankings(results)
        
        # Analyze feature importance
        importance_df = plot_feature_importance_comparison(model_actual, X, coach_features)
        
        # Save results
        save_results(results, high_impact_coaches, coach_stats, importance_df)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print("\nKey findings have been saved to data/final/")
        print("- coaching_impact_analysis.csv: Full analysis of coaching impact")
        print("- high_impact_coaches.csv: Coaches with highest positive impact")
        print("- coach_career_impact_stats.csv: Career statistics for each coach")
        print("- feature_importance_coaching_analysis.csv: Feature importance with coaching indicators")
        
        # Summary statistics
        print(f"\nSummary:")
        print(f"- Dataset used: {dataset_type}")
        print(f"- Train R²: {train_metrics['r2']:.4f}, Test R²: {test_metrics['r2']:.4f}")
        print(f"- Model residual SE: ±{test_rmse * 16:.2f} wins per season")
        print(f"- Coaching features account for {len(coach_features)} of {len(X.columns)} total features")
        print(f"- Average coaching WAR (Actual - Replacement): {results['Coaching_WAR'].mean():.4f}")
        print(f"- Maximum positive coaching WAR: {results['Coaching_WAR'].max():.4f}")
        print(f"- Maximum negative coaching WAR: {results['Coaching_WAR'].min():.4f}")
        print(f"- Average predicted impact: {results['Predicted_Impact'].mean():.4f}")
        print(f"- Log saved to: {log_filename}")
        
    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        tee.close()
        print(f"\nAnalysis complete. Log saved to: {log_filename}")

if __name__ == "__main__":
    main()