"""
XGBoost Coaching Impact Analysis Script
Compares XGBoost predictions using actual coach features versus replacement-level (average) coach features.
This quantifies the impact of individual coaching characteristics on team performance.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import warnings
warnings.filterwarnings('ignore')

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

def train_and_predict(X, y, model_params=None):
    """Train XGBoost model and generate predictions."""
    
    if model_params is None:
        # Default hyperparameters
        model_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'gamma': 0.01,
            'reg_lambda': 0,
            'subsample': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'verbosity': 0
        }
    
    # Create and train model
    model = xgb.XGBRegressor(**model_params)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    return model, y_pred

def analyze_coaching_impact(y_true, y_pred_actual, y_pred_replacement, team_year_info):
    """Analyze the impact of coaching by comparing actual vs replacement predictions."""
    print("\nAnalyzing coaching impact...")
    
    # Create results dataframe
    results = pd.DataFrame({
        'Team': team_year_info.get('Team', 'Unknown'),
        'Year': team_year_info.get('Year', 0),
        'Actual_Win_Pct': y_true.values,
        'Predicted_With_Coach': y_pred_actual,
        'Predicted_Replacement': y_pred_replacement,
        'Coaching_Impact': y_pred_actual - y_pred_replacement,
        'Actual_vs_Replacement': y_true.values - y_pred_replacement,
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
    
    # Calculate percentiles for coaching impact
    results['Impact_Percentile'] = results['Coaching_Impact'].rank(pct=True) * 100
    
    # Sort by coaching impact (largest positive impact first)
    results = results.sort_values('Coaching_Impact', ascending=False)
    
    # Calculate statistics
    print(f"\nCoaching Impact Statistics:")
    print(f"Mean coaching impact: {results['Coaching_Impact'].mean():.4f}")
    print(f"Std deviation: {results['Coaching_Impact'].std():.4f}")
    print(f"Min impact: {results['Coaching_Impact'].min():.4f}")
    print(f"Max impact: {results['Coaching_Impact'].max():.4f}")
    
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
    
    # Identify coaches with highest positive impact
    threshold_percentile = 95
    threshold_impact = 0.05
    
    high_impact_coaches = results[
        (results['Impact_Percentile'] >= threshold_percentile) | 
        (results['Coaching_Impact'] > threshold_impact)
    ].copy()
    
    print(f"\n{'='*80}")
    print(f"COACHES WITH HIGHEST POSITIVE IMPACT")
    print(f"(Top 5% or impact > 0.05)")
    print(f"{'='*80}")
    
    if len(high_impact_coaches) > 0:
        print(f"\nFound {len(high_impact_coaches)} team-years with significant positive coaching impact:\n")
        
        # Display top impact coaches
        for idx, row in high_impact_coaches.head(20).iterrows():
            coach_display = row['Primary_Coach'] if pd.notna(row['Primary_Coach']) else 'N/A'
            print(f"{row['Team']} ({int(row['Year'])}) - Coach: {coach_display}")
            print(f"  Predicted with coach: {row['Predicted_With_Coach']:.3f}")
            print(f"  Predicted replacement: {row['Predicted_Replacement']:.3f}")
            print(f"  Coaching impact: {row['Coaching_Impact']:+.3f} ({row['Impact_Percentile']:.1f} percentile)")
            print(f"  Actual Win%: {row['Actual_Win_Pct']:.3f}")
            print()
    
    # Show top 10 positive impact
    print(f"\n{'='*80}")
    print(f"TOP 10 POSITIVE COACHING IMPACTS")
    print(f"{'='*80}")
    
    top_10 = results.head(10)
    print(f"\n{'Team':<6} {'Year':<6} {'Coach':<25} {'With Coach':<12} {'Replacement':<12} {'Impact':<10} {'Actual'}")
    print("-" * 85)
    for idx, row in top_10.iterrows():
        if pd.isna(row['Primary_Coach']):
            coach_name = 'N/A'
        else:
            coach_name = row['Primary_Coach'][:23] if len(row['Primary_Coach']) > 23 else row['Primary_Coach']
        print(f"{row['Team']:<6} {int(row['Year']):<6} {coach_name:<25} {row['Predicted_With_Coach']:.3f}        "
              f"{row['Predicted_Replacement']:.3f}        {row['Coaching_Impact']:+.3f}      {row['Actual_Win_Pct']:.3f}")
    
    # Show bottom 10 (negative impact)
    print(f"\n{'='*80}")
    print(f"TOP 10 NEGATIVE COACHING IMPACTS")
    print(f"{'='*80}")
    
    bottom_10 = results.tail(10).iloc[::-1]  # Reverse to show worst first
    print(f"\n{'Team':<6} {'Year':<6} {'Coach':<25} {'With Coach':<12} {'Replacement':<12} {'Impact':<10} {'Actual'}")
    print("-" * 85)
    for idx, row in bottom_10.iterrows():
        if pd.isna(row['Primary_Coach']):
            coach_name = 'N/A'
        else:
            coach_name = row['Primary_Coach'][:23] if len(row['Primary_Coach']) > 23 else row['Primary_Coach']
        print(f"{row['Team']:<6} {int(row['Year']):<6} {coach_name:<25} {row['Predicted_With_Coach']:.3f}        "
              f"{row['Predicted_Replacement']:.3f}        {row['Coaching_Impact']:+.3f}      {row['Actual_Win_Pct']:.3f}")
    
    return results, high_impact_coaches

def analyze_coach_rankings(results):
    """Analyze coaching impact by individual coaches across their careers."""
    print(f"\n{'='*80}")
    print("COACH CAREER IMPACT ANALYSIS (WAR: Actual Win% - Replacement Prediction)")
    print(f"{'='*80}")
    
    # Group by coach and calculate statistics
    coach_stats = results.groupby('Primary_Coach').agg({
        'Actual_vs_Replacement': ['mean', 'std', 'count', 'sum'],  # Primary WAR metric
        'Coaching_Impact': ['mean', 'sum'],  # Secondary metric (predicted difference)
        'Actual_Win_Pct': 'mean',
        'Predicted_With_Coach': 'mean',
        'Predicted_Replacement': 'mean'
    }).round(4)
    
    # Flatten column names
    coach_stats.columns = ['_'.join(col).strip() for col in coach_stats.columns.values]
    coach_stats = coach_stats.rename(columns={
        'Actual_vs_Replacement_mean': 'Avg_WAR',
        'Actual_vs_Replacement_std': 'WAR_StdDev',
        'Actual_vs_Replacement_count': 'Seasons',
        'Actual_vs_Replacement_sum': 'Total_WAR',
        'Coaching_Impact_mean': 'Avg_Pred_Impact',
        'Coaching_Impact_sum': 'Total_Pred_Impact',
        'Actual_Win_Pct_mean': 'Avg_Actual_Win',
        'Predicted_With_Coach_mean': 'Avg_Pred_Coach',
        'Predicted_Replacement_mean': 'Avg_Pred_Replace'
    })
    
    # Filter for coaches with at least 3 seasons
    coach_stats = coach_stats[coach_stats['Seasons'] >= 3]
    
    # Sort by average WAR (actual vs replacement)
    coach_stats = coach_stats.sort_values('Avg_WAR', ascending=False)
    
    print(f"\nTop 15 Coaches by Average WAR (min 3 seasons):")
    print(f"\n{'Coach':<30} {'Avg WAR':<12} {'Seasons':<10} {'Total WAR':<12} {'Avg Win%'}")
    print("-" * 80)
    
    for coach, row in coach_stats.head(15).iterrows():
        if pd.notna(coach) and coach != 'N/A':
            coach_name = coach[:28] if len(coach) > 28 else coach
            print(f"{coach_name:<30} {row['Avg_WAR']:+.4f}      {int(row['Seasons']):<10} "
                  f"{row['Total_WAR']:+.4f}      {row['Avg_Actual_Win']:.3f}")
    
    print(f"\nBottom 15 Coaches by Average WAR (min 3 seasons):")
    print(f"\n{'Coach':<30} {'Avg WAR':<12} {'Seasons':<10} {'Total WAR':<12} {'Avg Win%'}")
    print("-" * 80)
    
    for coach, row in coach_stats.tail(15).iterrows():
        if pd.notna(coach) and coach != 'N/A':
            coach_name = coach[:28] if len(coach) > 28 else coach
            print(f"{coach_name:<30} {row['Avg_WAR']:+.4f}      {int(row['Seasons']):<10} "
                  f"{row['Total_WAR']:+.4f}      {row['Avg_Actual_Win']:.3f}")
    
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
    args = parser.parse_args()
    
    # Select dataset based on argument (default is without AV)
    if args.with_av:
        filepath = 'data/final/imputed_final_data.csv'
        dataset_type = "with ALL features (including AV)"
    else:
        filepath = 'data/final/imputed_final_data_no_AV.csv'
        dataset_type = "WITHOUT AV features"
    
    print("="*80)
    print("XGBOOST COACHING IMPACT ANALYSIS")
    print(f"Using dataset {dataset_type}")
    print("Comparing predictions with actual vs replacement-level coaching")
    print("="*80)
    
    # Load and prepare data
    X, y, team_year_info, full_df = load_and_prepare_data(filepath)
    
    # Identify coaching features
    coach_features = identify_coach_features(X)
    
    # Calculate replacement-level values
    replacement_values = calculate_replacement_features(X, coach_features, team_year_info)
    
    # Create replacement dataset
    X_replacement = create_replacement_dataset(X, coach_features, replacement_values)
    
    # Train model with actual data
    print("\nTraining model with actual coaching data...")
    model_actual, y_pred_actual = train_and_predict(X, y)
    
    # Generate predictions with replacement-level coaching
    print("\nGenerating predictions with replacement-level coaching...")
    y_pred_replacement = model_actual.predict(X_replacement)
    
    # Analyze coaching impact
    results, high_impact_coaches = analyze_coaching_impact(
        y, y_pred_actual, y_pred_replacement, team_year_info
    )
    
    # Analyze individual coach rankings
    coach_stats = analyze_coach_rankings(results)
    
    # Analyze feature importance
    importance_df = plot_feature_importance_comparison(model_actual, X, coach_features)
    
    # Save results
    save_results(results, high_impact_coaches, coach_stats, importance_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey findings have been saved to data/final/")
    print("- coaching_impact_analysis.csv: Full analysis of coaching impact")
    print("- high_impact_coaches.csv: Coaches with highest positive impact")
    print("- coach_career_impact_stats.csv: Career statistics for each coach")
    print("- feature_importance_coaching_analysis.csv: Feature importance with coaching indicators")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"- Dataset used: {dataset_type}")
    print(f"- Coaching features account for {len(coach_features)} of {len(X.columns)} total features")
    print(f"- Average coaching WAR (Actual - Replacement): {results['Actual_vs_Replacement'].mean():.4f}")
    print(f"- Maximum positive coaching WAR: {results['Actual_vs_Replacement'].max():.4f}")
    print(f"- Maximum negative coaching WAR: {results['Actual_vs_Replacement'].min():.4f}")
    print(f"- Average predicted impact: {results['Coaching_Impact'].mean():.4f}")

if __name__ == "__main__":
    main()