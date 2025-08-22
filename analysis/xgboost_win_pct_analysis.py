"""
XGBoost Win Percentage Analysis Script
Trains an XGBoost model on the complete coaching dataset to predict Win_Pct
and identifies teams that significantly outperformed predictions.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    
    return X, y, team_year_info

def train_xgboost_model(X, y):
    """Train XGBoost model with specified hyperparameters."""
    print("\nTraining XGBoost model...")
    
    # Specified hyperparameters
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'gamma': 0.01,
        'reg_lambda': 0,
        'subsample': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'verbosity': 1
    }
    
    # Create and train model
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    
    # Make predictions on the same data (100% training)
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\nModel Performance (on full dataset):")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    
    return model, y_pred

def analyze_predictions(y_true, y_pred, team_year_info):
    """Analyze prediction differences and identify outperformers."""
    print("\nAnalyzing predictions...")
    
    # Create results dataframe
    results = pd.DataFrame({
        'Team': team_year_info.get('Team', 'Unknown'),
        'Year': team_year_info.get('Year', 0),
        'Actual_Win_Pct': y_true.values,
        'Predicted_Win_Pct': y_pred,
        'Difference': y_true.values - y_pred
    })
    
    # Load head coach data
    coach_df = pd.read_csv('data/processed/Coaching/team_year_head_coaches.csv')
    
    # Merge coach information
    results = results.merge(
        coach_df[['Team', 'Year', 'Primary_Coach', 'Combined_Coach']], 
        on=['Team', 'Year'], 
        how='left'
    )
    
    # Calculate percentiles for differences
    results['Difference_Percentile'] = results['Difference'].rank(pct=True) * 100
    
    # Sort by difference (largest outperformance first)
    results = results.sort_values('Difference', ascending=False)
    
    # Calculate statistics
    print(f"\nPrediction Difference Statistics:")
    print(f"Mean difference: {results['Difference'].mean():.4f}")
    print(f"Std deviation: {results['Difference'].std():.4f}")
    print(f"Min difference: {results['Difference'].min():.4f}")
    print(f"Max difference: {results['Difference'].max():.4f}")
    
    # Identify significant outperformers (top 5% or difference > 0.15)
    threshold_percentile = 95
    threshold_difference = 0.15
    
    outperformers = results[
        (results['Difference_Percentile'] >= threshold_percentile) | 
        (results['Difference'] > threshold_difference)
    ].copy()
    
    print(f"\n{'='*80}")
    print(f"TEAMS THAT SIGNIFICANTLY OUTPERFORMED PREDICTIONS")
    print(f"(Top 5% or difference > 0.15)")
    print(f"{'='*80}")
    
    if len(outperformers) > 0:
        print(f"\nFound {len(outperformers)} teams that significantly outperformed predictions:\n")
        
        # Display outperformers
        for idx, row in outperformers.iterrows():
            coach_display = row['Primary_Coach'] if pd.notna(row['Primary_Coach']) else 'N/A'
            print(f"{row['Team']} ({int(row['Year'])}) - Coach: {coach_display}")
            print(f"  Actual Win%: {row['Actual_Win_Pct']:.3f}")
            print(f"  Predicted Win%: {row['Predicted_Win_Pct']:.3f}")
            print(f"  Outperformed by: {row['Difference']:.3f} ({row['Difference_Percentile']:.1f} percentile)")
            if pd.notna(row['Primary_Coach']) and pd.notna(row['Combined_Coach']) and row['Primary_Coach'] != row['Combined_Coach']:
                print(f"  (Full season: {row['Combined_Coach']})")
            print()
    
    # Also show top 10 outperformers regardless of threshold
    print(f"\n{'='*80}")
    print(f"TOP 10 TEAMS THAT OUTPERFORMED PREDICTIONS")
    print(f"{'='*80}")
    
    top_10 = results.head(10)
    print(f"\n{'Team':<6} {'Year':<6} {'Coach':<25} {'Actual':<8} {'Predicted':<10} {'Difference':<10} {'Percentile'}")
    print("-" * 95)
    for idx, row in top_10.iterrows():
        if pd.isna(row['Primary_Coach']):
            coach_name = 'N/A'
        else:
            coach_name = row['Primary_Coach'][:23] if len(row['Primary_Coach']) > 23 else row['Primary_Coach']
        print(f"{row['Team']:<6} {int(row['Year']):<6} {coach_name:<25} {row['Actual_Win_Pct']:.3f}    {row['Predicted_Win_Pct']:.3f}      "
              f"{row['Difference']:+.3f}      {row['Difference_Percentile']:.1f}%")
    
    # Show bottom 10 (underperformers)
    print(f"\n{'='*80}")
    print(f"TOP 10 TEAMS THAT UNDERPERFORMED PREDICTIONS")
    print(f"{'='*80}")
    
    bottom_10 = results.tail(10).iloc[::-1]  # Reverse to show worst first
    print(f"\n{'Team':<6} {'Year':<6} {'Coach':<25} {'Actual':<8} {'Predicted':<10} {'Difference':<10} {'Percentile'}")
    print("-" * 95)
    for idx, row in bottom_10.iterrows():
        if pd.isna(row['Primary_Coach']):
            coach_name = 'N/A'
        else:
            coach_name = row['Primary_Coach'][:23] if len(row['Primary_Coach']) > 23 else row['Primary_Coach']
        print(f"{row['Team']:<6} {int(row['Year']):<6} {coach_name:<25} {row['Actual_Win_Pct']:.3f}    {row['Predicted_Win_Pct']:.3f}      "
              f"{row['Difference']:+.3f}      {row['Difference_Percentile']:.1f}%")
    
    return results, outperformers

def analyze_coach_performance(results):
    """Aggregate results by coach and rank by average win percentage increase."""
    print(f"\n{'='*80}")
    print(f"COACH-LEVEL PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Filter out rows with missing coach data
    coach_results = results[results['Primary_Coach'].notna()].copy()
    
    if len(coach_results) == 0:
        print("No coach data available for analysis.")
        return pd.DataFrame()
    
    # Group by coach and calculate statistics
    coach_stats = coach_results.groupby('Primary_Coach').agg({
        'Difference': ['count', 'mean', 'std', 'min', 'max'],
        'Actual_Win_Pct': 'mean',
        'Predicted_Win_Pct': 'mean'
    }).round(4)
    
    # Flatten column names
    coach_stats.columns = ['Games', 'Avg_Difference', 'Std_Difference', 'Min_Difference', 'Max_Difference',
                          'Avg_Actual_Win_Pct', 'Avg_Predicted_Win_Pct']
    
    # Filter coaches with at least 2 seasons for meaningful analysis (exclude one-year coaches)
    min_games = 2
    coach_stats = coach_stats[coach_stats['Games'] >= min_games].copy()
    
    # Sort by average difference (highest positive impact first)
    coach_stats = coach_stats.sort_values('Avg_Difference', ascending=False)
    
    print(f"\nCoach performance analysis (coaches with {min_games}+ seasons):")
    print(f"Found {len(coach_stats)} coaches with sufficient data\n")
    
    # Display top 20 coaches with highest average win percentage increase
    print(f"TOP 20 COACHES - HIGHEST AVERAGE WIN PERCENTAGE INCREASE:")
    print("-" * 120)
    print(f"{'Coach Name':<25} {'Games':<6} {'Avg +/-':<8} {'Std Dev':<8} {'Min':<8} {'Max':<8} {'Actual Win%':<12} {'Predicted Win%'}")
    print("-" * 120)
    
    for idx, (coach, row) in enumerate(coach_stats.head(20).iterrows(), 1):
        coach_display = coach[:23] if len(coach) > 23 else coach
        print(f"{coach_display:<25} {int(row['Games']):<6} {row['Avg_Difference']:+.4f}   {row['Std_Difference']:.4f}   "
              f"{row['Min_Difference']:+.3f}   {row['Max_Difference']:+.3f}   {row['Avg_Actual_Win_Pct']:.3f}        {row['Avg_Predicted_Win_Pct']:.3f}")
    
    # Display bottom 20 coaches (largest negative impact)
    print(f"\nBOTTOM 20 COACHES - LARGEST NEGATIVE WIN PERCENTAGE IMPACT:")
    print("-" * 120)
    print(f"{'Coach Name':<25} {'Games':<6} {'Avg +/-':<8} {'Std Dev':<8} {'Min':<8} {'Max':<8} {'Actual Win%':<12} {'Predicted Win%'}")
    print("-" * 120)
    
    for idx, (coach, row) in enumerate(coach_stats.tail(20).iterrows(), 1):
        coach_display = coach[:23] if len(coach) > 23 else coach
        print(f"{coach_display:<25} {int(row['Games']):<6} {row['Avg_Difference']:+.4f}   {row['Std_Difference']:.4f}   "
              f"{row['Min_Difference']:+.3f}   {row['Max_Difference']:+.3f}   {row['Avg_Actual_Win_Pct']:.3f}        {row['Avg_Predicted_Win_Pct']:.3f}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"COACH PERFORMANCE SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total coaches analyzed: {len(coach_stats)}")
    print(f"Average win percentage increase across all coaches: {coach_stats['Avg_Difference'].mean():+.4f}")
    print(f"Standard deviation of coach impacts: {coach_stats['Avg_Difference'].std():.4f}")
    print(f"Best coach average impact: {coach_stats['Avg_Difference'].max():+.4f} ({coach_stats['Avg_Difference'].idxmax()})")
    print(f"Worst coach average impact: {coach_stats['Avg_Difference'].min():+.4f} ({coach_stats['Avg_Difference'].idxmin()})")
    
    # Calculate coaches above/below expectations
    positive_coaches = len(coach_stats[coach_stats['Avg_Difference'] > 0])
    negative_coaches = len(coach_stats[coach_stats['Avg_Difference'] < 0])
    print(f"\nCoaches performing above model predictions: {positive_coaches} ({positive_coaches/len(coach_stats)*100:.1f}%)")
    print(f"Coaches performing below model predictions: {negative_coaches} ({negative_coaches/len(coach_stats)*100:.1f}%)")
    
    return coach_stats

def plot_feature_importance(model, X, top_n=20):
    """Display top feature importances."""
    print(f"\n{'='*80}")
    print(f"TOP {top_n} FEATURE IMPORTANCES")
    print(f"{'='*80}")
    
    # Get feature importances
    importance = model.feature_importances_
    features = X.columns
    
    # Create dataframe and sort
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Display top features
    print(f"\n{'Rank':<6} {'Feature':<50} {'Importance':<10}")
    print("-" * 70)
    for i, row in importance_df.head(top_n).iterrows():
        print(f"{importance_df.index.get_loc(i)+1:<6} {row['Feature']:<50} {row['Importance']:.6f}")
    
    return importance_df

def save_results(results, outperformers, importance_df, coach_stats=None):
    """Save analysis results to CSV files."""
    print("\nSaving results...")
    
    # Save full results
    results_file = 'data/final/xgboost_predictions_full.csv'
    results.to_csv(results_file, index=False)
    print(f"Full results saved to: {results_file}")
    
    # Save outperformers
    if len(outperformers) > 0:
        outperformers_file = 'data/final/xgboost_outperformers.csv'
        outperformers.to_csv(outperformers_file, index=False)
        print(f"Outperformers saved to: {outperformers_file}")
    
    # Save feature importance
    importance_file = 'data/final/xgboost_feature_importance.csv'
    importance_df.to_csv(importance_file, index=False)
    print(f"Feature importance saved to: {importance_file}")
    
    # Save coach statistics
    if coach_stats is not None and len(coach_stats) > 0:
        coach_file = 'data/final/coach_win_pct_analysis.csv'
        coach_stats.to_csv(coach_file)  # Keep index (coach names)
        print(f"Coach analysis saved to: {coach_file}")

def main():
    """Main execution function."""
    print("="*80)
    print("XGBOOST WIN PERCENTAGE ANALYSIS")
    print("Training on 100% of data to identify coaching outperformance")
    print("="*80)
    
    # Load and prepare data
    filepath = 'data/final/imputed_final_data.csv'
    X, y, team_year_info = load_and_prepare_data(filepath)
    
    # Train model
    model, y_pred = train_xgboost_model(X, y)
    
    # Analyze predictions
    results, outperformers = analyze_predictions(y, y_pred, team_year_info)
    
    # Analyze coach-level performance
    coach_stats = analyze_coach_performance(results)
    
    # Display feature importance
    importance_df = plot_feature_importance(model, X, top_n=20)
    
    # Save results
    save_results(results, outperformers, importance_df, coach_stats)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey findings have been saved to data/final/")
    print("- xgboost_predictions_full.csv: All predictions and differences")
    print("- xgboost_outperformers.csv: Teams that significantly outperformed")
    print("- xgboost_feature_importance.csv: Feature importance rankings")
    print("- coach_win_pct_analysis.csv: Coach-level performance aggregation")

if __name__ == "__main__":
    main()