"""
XGBoost Coaching Impact Analysis Script - Year-Specific Replacement
Compares XGBoost predictions using actual coach features versus year-specific median coach features.
This uses year-specific replacement levels rather than overall career averages.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import uniform, randint
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

def load_and_prepare_data(filepath):
    """Load the combined dataset and prepare features."""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Ensure Win_Pct is the target
    if 'Win_Pct' not in df.columns:
        raise ValueError("Win_Pct column not found in dataset")
    
    # Separate features and target
    X = df.drop(['Win_Pct'], axis=1)
    y = df['Win_Pct']
    
    # Store team and year for later analysis
    if 'Team' in X.columns and 'Year' in X.columns:
        team_year_info = df[['Team', 'Year']].copy()
        # Remove Team from features if it's a string column
        if X['Team'].dtype == 'object':
            X = X.drop(['Team'], axis=1)
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

def calculate_year_specific_replacement_features(X, coach_features, team_year_info):
    """Calculate year-specific median values for coach features."""
    print("\nCalculating year-specific replacement-level coach features...")
    print("Using year-specific medians (contemporary peer comparison)...")
    
    # Ensure we have Year information
    if 'Year' not in team_year_info.columns:
        print("Error: No Year column available for year-specific calculation")
        raise ValueError("Missing Year column")
    
    # Create combined dataframe with features and year
    combined_df = team_year_info.reset_index(drop=True).copy()
    
    # Add coaching features to the dataframe
    for feature in coach_features:
        if feature in X.columns:
            combined_df[feature] = X[feature].values
    
    # Calculate year-specific medians for each feature
    replacement_values_by_year = {}
    
    unique_years = sorted(combined_df['Year'].unique())
    print(f"Calculating replacement values for {len(unique_years)} years ({min(unique_years)}-{max(unique_years)})")
    
    for feature in coach_features:
        if feature in combined_df.columns:
            feature_by_year = {}
            
            for year in unique_years:
                year_data = combined_df[combined_df['Year'] == year]
                year_median = year_data[feature].median()
                feature_by_year[year] = year_median
            
            replacement_values_by_year[feature] = feature_by_year
            
            # Show sample years for this feature
            sample_years = list(feature_by_year.keys())[:5]
            sample_values = [feature_by_year[y] for y in sample_years]
            print(f"  {feature}: {len(feature_by_year)} years, sample {sample_years}: {sample_values}")
    
    print(f"\nCalculated year-specific replacement values for {len(replacement_values_by_year)} features")
    print("Each team-year will be compared to that year's median coach performance")
    
    return replacement_values_by_year

def create_year_specific_replacement_dataset(X, coach_features, replacement_values_by_year, team_year_info):
    """Create dataset with coach features replaced by year-specific median values."""
    print("\nCreating year-specific replacement-level dataset...")
    
    # Create a copy of the original data
    X_replacement = X.copy()
    
    # Replace coaching features with year-specific median values
    years = team_year_info['Year'].values
    features_replaced = 0
    
    for feature in coach_features:
        if feature in X_replacement.columns and feature in replacement_values_by_year:
            # Replace each row with that year's median value
            for i, year in enumerate(years):
                if year in replacement_values_by_year[feature]:
                    X_replacement.iloc[i, X_replacement.columns.get_loc(feature)] = replacement_values_by_year[feature][year]
            features_replaced += 1
    
    print(f"Replaced {features_replaced} coaching features with year-specific median values")
    
    return X_replacement

def train_and_predict(X, y, team_year_info, use_tuning=True, cv_folds=5, n_iter=50, test_size=0.2):
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
                unique_coaches, test_size=test_size, random_state=42
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
            'n_estimators': [50, 75, 100, 150, 200, 250, 300],
            'learning_rate': [0.01, 0.03, 0.05],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0, 1.5, 2.0],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        
        # Base model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
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
            random_state=42,
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
            
    else:
        print("\nUsing default hyperparameters (no tuning)...")
        # Default hyperparameters
        model_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'gamma': 0.01,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
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

def analyze_coaching_impact(y_true, y_pred_actual, y_pred_replacement, team_year_info):
    """Analyze the impact of coaching by comparing actual vs replacement predictions."""
    print("\nAnalyzing coaching impact...")
    
    # Create results dataframe
    # Key metric: Coaching WAR = Actual Win% - Year-Specific Replacement Prediction
    results = pd.DataFrame({
        'Team': team_year_info.get('Team', 'Unknown'),
        'Year': team_year_info.get('Year', 0),
        'Actual_Win_Pct': y_true.values,
        'Predicted_With_Coach': y_pred_actual,
        'Predicted_Replacement': y_pred_replacement,
        'Coaching_WAR': y_true.values - y_pred_replacement,  # Primary metric: Actual - Year-specific replacement prediction
        'Predicted_Impact': y_pred_actual - y_pred_replacement,  # Secondary metric: Model's predicted difference
        'Prediction_Error_Coach': y_true.values - y_pred_actual,
        'Prediction_Error_Replacement': y_true.values - y_pred_replacement
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
    print(f"\nCoaching WAR Statistics (vs Year-Specific Medians):")
    print(f"Mean coaching WAR: {results['Coaching_WAR'].mean():.4f}")
    print(f"Std deviation: {results['Coaching_WAR'].std():.4f}")
    print(f"Min WAR: {results['Coaching_WAR'].min():.4f}")
    print(f"Max WAR: {results['Coaching_WAR'].max():.4f}")
    
    print(f"\nPredicted Impact Statistics:")
    print(f"Mean predicted impact: {results['Predicted_Impact'].mean():.4f}")
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
    print("MODEL PERFORMANCE COMPARISON (vs Year-Specific Replacement)")
    print(f"{'='*80}")
    print(f"\n{'Metric':<20} {'With Actual Coach':<20} {'With Year Replacement':<20} {'Difference':<15}")
    print("-" * 75)
    print(f"{'MSE':<20} {mse_actual:<20.6f} {mse_replacement:<20.6f} {mse_replacement - mse_actual:<15.6f}")
    print(f"{'MAE':<20} {mae_actual:<20.6f} {mae_replacement:<20.6f} {mae_replacement - mae_actual:<15.6f}")
    print(f"{'R² Score':<20} {r2_actual:<20.4f} {r2_replacement:<20.4f} {r2_replacement - r2_actual:<15.4f}")
    print(f"{'RMSE':<20} {np.sqrt(mse_actual):<20.6f} {np.sqrt(mse_replacement):<20.6f} {np.sqrt(mse_replacement) - np.sqrt(mse_actual):<15.6f}")
    
    return results, None

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='XGBoost Coaching Impact Analysis - Year-Specific')
    parser.add_argument('--with-av', action='store_true', 
                       help='Include AV (Approximate Value) features in analysis')
    parser.add_argument('--no-tuning', action='store_true',
                       help='Skip hyperparameter tuning and use default parameters')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds for tuning (default: 5)')
    parser.add_argument('--n-iter', type=int, default=50,
                       help='Number of iterations for RandomizedSearchCV (default: 50)')
    args = parser.parse_args()
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'analysis/outputs/logs/coaching_analysis_year_specific_log_{timestamp}.txt'
    
    # Set up output redirection to both console and file
    tee = TeeOutput(log_filename)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        # Select dataset based on argument (default is without AV)
        if args.with_av:
            filepath = 'data/final/imputed_final_data.csv'
            dataset_type = "with ALL features (including AV)"
        else:
            filepath = 'data/final/imputed_final_data_no_AV.csv'
            dataset_type = "WITHOUT AV features"
        
        print("="*80)
        print("XGBOOST COACHING IMPACT ANALYSIS - YEAR-SPECIFIC REPLACEMENT")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using dataset {dataset_type}")
        print("Comparing predictions with actual vs YEAR-SPECIFIC MEDIAN coaching")
        if not args.no_tuning:
            print(f"Hyperparameter tuning: ENABLED ({args.n_iter} iterations, {args.cv_folds} CV folds)")
        else:
            print("Hyperparameter tuning: DISABLED (using default parameters)")
        print(f"Log file: {log_filename}")
        print("="*80)
    
        # Load and prepare data
        X, y, team_year_info, full_df = load_and_prepare_data(filepath)
        
        # Identify coaching features
        coach_features = identify_coach_features(X)
        
        # Calculate year-specific replacement-level values
        replacement_values_by_year = calculate_year_specific_replacement_features(X, coach_features, team_year_info)
        
        # Create year-specific replacement dataset
        X_replacement = create_year_specific_replacement_dataset(X, coach_features, replacement_values_by_year, team_year_info)
        
        # Train model with actual data
        print("\nTraining model with actual coaching data...")
        use_tuning = not args.no_tuning
        model_actual, y_pred_actual, train_metrics, test_metrics = train_and_predict(
            X, y, team_year_info, use_tuning=use_tuning, cv_folds=args.cv_folds, n_iter=args.n_iter)
        
        # Generate predictions with year-specific replacement-level coaching
        print("\nGenerating predictions with year-specific replacement-level coaching...")
        y_pred_replacement = model_actual.predict(X_replacement)
        
        # Analyze coaching impact
        results, _ = analyze_coaching_impact(
            y, y_pred_actual, y_pred_replacement, team_year_info
        )
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - YEAR-SPECIFIC VERSION")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Summary statistics
        print(f"\nSummary:")
        print(f"- Dataset used: {dataset_type}")
        print(f"- Replacement method: Year-specific medians")
        print(f"- Train R²: {train_metrics['r2']:.4f}, Test R²: {test_metrics['r2']:.4f}")
        print(f"- Coaching features: {len(coach_features)} of {len(X.columns)} total features")
        print(f"- Average coaching WAR (vs year-specific): {results['Coaching_WAR'].mean():.4f}")
        print(f"- Maximum positive coaching WAR: {results['Coaching_WAR'].max():.4f}")
        print(f"- Maximum negative coaching WAR: {results['Coaching_WAR'].min():.4f}")
        print(f"- Average predicted impact: {results['Predicted_Impact'].mean():.4f}")
        print(f"- Log saved to: {log_filename}")
        
        # Calculate final R2 on full dataset
        final_r2_actual = r2_score(y, y_pred_actual)
        final_r2_replacement = r2_score(y, y_pred_replacement)
        
        print(f"\nFinal R² Scores on Full Dataset:")
        print(f"- With actual coaching: {final_r2_actual:.4f}")
        print(f"- With year-specific replacement: {final_r2_replacement:.4f}")
        print(f"- Difference: {final_r2_actual - final_r2_replacement:.4f}")
        
    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        tee.close()
        print(f"\nAnalysis complete. Log saved to: {log_filename}")

if __name__ == "__main__":
    main()