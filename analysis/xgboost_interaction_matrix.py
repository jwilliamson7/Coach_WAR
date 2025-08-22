"""
XGBoost Feature Interaction Matrix Analysis
Creates bucketed interaction matrices showing model predictions across two features

Usage:
    python xgboost_interaction_matrix.py feature1 feature2
    
Example:
    python xgboost_interaction_matrix.py Avg_Starter_AV Avg_Starter_AV_QB
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# Create output directories if they don't exist
OUTPUT_DIR = 'analysis/interaction_matrices'
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')
PNG_DIR = os.path.join(OUTPUT_DIR, 'png')

for directory in [OUTPUT_DIR, CSV_DIR, PNG_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_model_and_data(filepath='data/final/imputed_final_data.csv'):
    """Load data and train XGBoost model."""
    print("Loading data and training model...")
    df = pd.read_csv(filepath)
    
    # Ensure Win_Pct is the target
    if 'Win_Pct' not in df.columns:
        raise ValueError("Win_Pct column not found in dataset")
    
    # Separate features and target
    X = df.drop(['Win_Pct'], axis=1)
    y = df['Win_Pct']
    
    # Store team and year for reference
    if 'Team' in X.columns and 'Year' in X.columns:
        team_year_info = df[['Team', 'Year']].copy()
        if X['Team'].dtype == 'object':
            X = X.drop(['Team'], axis=1)
    else:
        team_year_info = pd.DataFrame(index=df.index)
    
    # Convert object columns to numeric
    for col in X.select_dtypes(include=['object']).columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            X = X.drop([col], axis=1)
    
    # Handle missing values
    X = X.fillna(0)
    
    # Train model with specified hyperparameters
    params = {
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
    
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    
    # Calculate model performance
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"Model R² Score: {r2:.4f}")
    print(f"Model RMSE: {rmse:.4f}")
    
    return model, X, y, team_year_info

def create_interaction_matrix(model, X, feature1, feature2, n_bins1=10, n_bins2=10, 
                            percentile_based=True, show_counts=True):
    """
    Create an interaction matrix showing model predictions across two features.
    
    Parameters:
    -----------
    model : trained XGBoost model
    X : DataFrame of features
    feature1 : str, name of first feature (x-axis)
    feature2 : str, name of second feature (y-axis)
    n_bins1 : int, number of bins for feature1
    n_bins2 : int, number of bins for feature2
    percentile_based : bool, if True use percentile-based bins, else equal-width
    show_counts : bool, if True show sample counts in each cell
    
    Returns:
    --------
    interaction_matrix : DataFrame with predicted win percentages
    count_matrix : DataFrame with sample counts per cell
    """
    
    print(f"\nCreating interaction matrix for {feature1} vs {feature2}")
    
    # Check features exist
    if feature1 not in X.columns:
        raise ValueError(f"{feature1} not found in features")
    if feature2 not in X.columns:
        raise ValueError(f"{feature2} not found in features")
    
    # Create bins for each feature
    if percentile_based:
        # Use percentiles for more balanced bins
        percentiles1 = np.linspace(0, 100, n_bins1 + 1)
        percentiles2 = np.linspace(0, 100, n_bins2 + 1)
        bins1 = np.percentile(X[feature1].dropna(), percentiles1)
        bins2 = np.percentile(X[feature2].dropna(), percentiles2)
        # Ensure unique bins
        bins1 = np.unique(bins1)
        bins2 = np.unique(bins2)
    else:
        # Equal-width bins
        bins1 = np.linspace(X[feature1].min(), X[feature1].max(), n_bins1 + 1)
        bins2 = np.linspace(X[feature2].min(), X[feature2].max(), n_bins2 + 1)
    
    # Adjust bins slightly to include all values
    bins1[0] -= 0.001
    bins1[-1] += 0.001
    bins2[0] -= 0.001
    bins2[-1] += 0.001
    
    # Create bin labels with [x, y) notation
    # First and last bins get special treatment
    labels1 = []
    for i in range(len(bins1)-1):
        if i == 0:
            # First bin includes lower bound
            labels1.append(f"[{bins1[i]+0.001:.1f}, {bins1[i+1]:.1f})")
        elif i == len(bins1)-2:
            # Last bin includes upper bound
            labels1.append(f"[{bins1[i]:.1f}, {bins1[i+1]-0.001:.1f}]")
        else:
            # Middle bins: [lower, upper)
            labels1.append(f"[{bins1[i]:.1f}, {bins1[i+1]:.1f})")
    
    labels2 = []
    for i in range(len(bins2)-1):
        if i == 0:
            # First bin includes lower bound
            labels2.append(f"[{bins2[i]+0.001:.1f}, {bins2[i+1]:.1f})")
        elif i == len(bins2)-2:
            # Last bin includes upper bound
            labels2.append(f"[{bins2[i]:.1f}, {bins2[i+1]-0.001:.1f}]")
        else:
            # Middle bins: [lower, upper)
            labels2.append(f"[{bins2[i]:.1f}, {bins2[i+1]:.1f})")
    
    # Bin the features
    X_copy = X.copy()
    X_copy[f'{feature1}_bin'] = pd.cut(X[feature1], bins=bins1, labels=labels1, include_lowest=True)
    X_copy[f'{feature2}_bin'] = pd.cut(X[feature2], bins=bins2, labels=labels2, include_lowest=True)
    
    # Initialize matrices
    interaction_matrix = pd.DataFrame(index=labels2[::-1], columns=labels1)
    count_matrix = pd.DataFrame(index=labels2[::-1], columns=labels1)
    
    # Calculate predictions for each bin combination
    for i, label1 in enumerate(labels1):
        for j, label2 in enumerate(labels2[::-1]):  # Reverse for proper display
            # Get samples in this bin combination
            mask = (X_copy[f'{feature1}_bin'] == label1) & (X_copy[f'{feature2}_bin'] == label2)
            
            if mask.sum() > 0:
                # Get median values for this bin combination
                X_subset = X[mask]
                X_median = X_subset.median().to_frame().T
                
                # Ensure all columns are present
                for col in X.columns:
                    if col not in X_median.columns:
                        X_median[col] = X[col].median()
                
                # Reorder columns to match original
                X_median = X_median[X.columns]
                
                # Set the specific features to bin midpoints
                bin1_mid = (bins1[i] + bins1[i+1]) / 2
                bin2_mid = (bins2[len(labels2)-j-1] + bins2[len(labels2)-j]) / 2
                X_median[feature1] = bin1_mid
                X_median[feature2] = bin2_mid
                
                # Predict
                pred = model.predict(X_median)[0]
                interaction_matrix.loc[label2, label1] = pred
                count_matrix.loc[label2, label1] = mask.sum()
            else:
                interaction_matrix.loc[label2, label1] = np.nan
                count_matrix.loc[label2, label1] = 0
    
    # Convert to numeric
    interaction_matrix = interaction_matrix.astype(float)
    count_matrix = count_matrix.astype(int)
    
    return interaction_matrix, count_matrix

def plot_interaction_matrix(interaction_matrix, count_matrix, feature1, feature2, 
                           save_path=None, show_counts=True, figsize=(20, 10)):
    """
    Plot the interaction matrix as a heatmap with sample size visualization.
    
    Parameters:
    -----------
    interaction_matrix : DataFrame with predicted values
    count_matrix : DataFrame with sample counts
    feature1 : str, name of first feature
    feature2 : str, name of second feature
    save_path : str, optional path to save figure
    show_counts : bool, whether to show sample counts in cells
    figsize : tuple, figure size
    """
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # LEFT PLOT: Predictions only (no sample counts)
    # Create annotations with just predictions
    annot_matrix = pd.DataFrame(index=interaction_matrix.index, 
                               columns=interaction_matrix.columns)
    for i in interaction_matrix.index:
        for j in interaction_matrix.columns:
            val = interaction_matrix.loc[i, j]
            if pd.notna(val):
                annot_matrix.loc[i, j] = f"{val:.3f}"
            else:
                annot_matrix.loc[i, j] = ""
    
    # Create predictions heatmap
    sns.heatmap(interaction_matrix, 
                annot=annot_matrix,
                fmt='',
                cmap='RdYlGn',
                center=0.5,
                vmin=0.2,
                vmax=0.8,
                cbar_kws={'label': 'Predicted Win Percentage'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax1)
    
    ax1.set_title(f'Win Percentage Predictions\n{feature1} vs {feature2}', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel(feature1, fontsize=10, labelpad=10)
    ax1.set_ylabel(feature2, fontsize=10, labelpad=15)
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # RIGHT PLOT: Sample sizes
    # Create custom colormap from white to purple
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['white', '#E8D8F5', '#D1B3E6', '#BA8DD7', '#A368C8', '#8B42B9', '#741CAA', '#5C009B']
    n_bins = 100
    cmap_purple = LinearSegmentedColormap.from_list('white_purple', colors, N=n_bins)
    
    # Create sample size annotations
    annot_counts = count_matrix.astype(str)
    annot_counts[count_matrix == 0] = ''
    
    # Plot sample sizes
    max_count = count_matrix.max().max()
    sns.heatmap(count_matrix,
                annot=annot_counts,
                fmt='',
                cmap=cmap_purple,
                vmin=0,
                vmax=max_count,
                cbar_kws={'label': 'Sample Size'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax2)
    
    ax2.set_title(f'Sample Sizes per Cell\n{feature1} vs {feature2}', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel(feature1, fontsize=10, labelpad=10)
    ax2.set_ylabel('', fontsize=10, labelpad=15)  # No y-label since it's same as left plot
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    # Overall title
    fig.suptitle(f'Interaction Analysis: {feature1} × {feature2}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Don't show the plot - just save it
    plt.close(fig)

def analyze_feature_interaction(model, X, feature1, feature2, n_bins1=8, n_bins2=8):
    """
    Complete analysis of feature interaction with statistics.
    
    Parameters:
    -----------
    model : trained model
    X : feature DataFrame
    feature1 : str, first feature name
    feature2 : str, second feature name
    n_bins1 : int, number of bins for feature1
    n_bins2 : int, number of bins for feature2
    """
    
    print("="*80)
    print(f"FEATURE INTERACTION ANALYSIS")
    print(f"{feature1} vs {feature2}")
    print("="*80)
    
    # Feature statistics
    print(f"\n{feature1} Statistics:")
    print(f"  Mean: {X[feature1].mean():.3f}")
    print(f"  Std: {X[feature1].std():.3f}")
    print(f"  Min: {X[feature1].min():.3f}")
    print(f"  Max: {X[feature1].max():.3f}")
    
    print(f"\n{feature2} Statistics:")
    print(f"  Mean: {X[feature2].mean():.3f}")
    print(f"  Std: {X[feature2].std():.3f}")
    print(f"  Min: {X[feature2].min():.3f}")
    print(f"  Max: {X[feature2].max():.3f}")
    
    # Calculate correlation
    correlation = X[[feature1, feature2]].corr().iloc[0, 1]
    print(f"\nFeature Correlation: {correlation:.3f}")
    
    # Create interaction matrix
    interaction_matrix, count_matrix = create_interaction_matrix(
        model, X, feature1, feature2, n_bins1, n_bins2, percentile_based=True
    )
    
    # Plot the matrix
    # Clean feature names for filenames (remove special characters)
    clean_feat1 = feature1.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    clean_feat2 = feature2.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    save_path = os.path.join(PNG_DIR, f'interaction_{clean_feat1}_vs_{clean_feat2}.png')
    plot_interaction_matrix(interaction_matrix, count_matrix, feature1, feature2,
                          save_path=save_path)
    
    # Analyze the interaction effect
    print("\nInteraction Effect Analysis:")
    
    # Find cells with highest and lowest predictions
    flat_predictions = interaction_matrix.values.flatten()
    flat_predictions = flat_predictions[~np.isnan(flat_predictions)]
    
    if len(flat_predictions) > 0:
        print(f"  Prediction Range: {flat_predictions.min():.3f} to {flat_predictions.max():.3f}")
        print(f"  Prediction Spread: {flat_predictions.max() - flat_predictions.min():.3f}")
        print(f"  Mean Prediction: {flat_predictions.mean():.3f}")
        
        # Find best and worst combinations
        max_idx = np.unravel_index(np.nanargmax(interaction_matrix.values), 
                                  interaction_matrix.shape)
        min_idx = np.unravel_index(np.nanargmin(interaction_matrix.values), 
                                  interaction_matrix.shape)
        
        print(f"\nBest Combination:")
        print(f"  {feature1}: {interaction_matrix.columns[max_idx[1]]}")
        print(f"  {feature2}: {interaction_matrix.index[max_idx[0]]}")
        print(f"  Predicted Win%: {interaction_matrix.iloc[max_idx]:.3f}")
        print(f"  Sample Count: {count_matrix.iloc[max_idx]}")
        
        print(f"\nWorst Combination:")
        print(f"  {feature1}: {interaction_matrix.columns[min_idx[1]]}")
        print(f"  {feature2}: {interaction_matrix.index[min_idx[0]]}")
        print(f"  Predicted Win%: {interaction_matrix.iloc[min_idx]:.3f}")
        print(f"  Sample Count: {count_matrix.iloc[min_idx]}")
    
    # Save matrices to CSV
    # Clean feature names for filenames (remove special characters)
    clean_feat1 = feature1.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    clean_feat2 = feature2.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    interaction_csv = os.path.join(CSV_DIR, f'interaction_matrix_{clean_feat1}_vs_{clean_feat2}.csv')
    count_csv = os.path.join(CSV_DIR, f'interaction_counts_{clean_feat1}_vs_{clean_feat2}.csv')
    
    interaction_matrix.to_csv(interaction_csv)
    count_matrix.to_csv(count_csv)
    print(f"\nMatrices saved to {OUTPUT_DIR}/")
    
    return interaction_matrix, count_matrix

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate interaction matrix for two features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python xgboost_interaction_matrix.py Avg_Starter_AV Avg_Starter_AV_QB
  python xgboost_interaction_matrix.py DeadCap_Pct Avg_Starter_AV
  python xgboost_interaction_matrix.py num_times_hc "PF (Points For)__oc_Norm"
        """
    )
    
    parser.add_argument('feature1', 
                       help='First feature name (x-axis)')
    parser.add_argument('feature2', 
                       help='Second feature name (y-axis)')
    parser.add_argument('--bins1', type=int, default=8,
                       help='Number of bins for feature1 (default: 8)')
    parser.add_argument('--bins2', type=int, default=8,
                       help='Number of bins for feature2 (default: 8)')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    print("="*80)
    print("XGBOOST INTERACTION MATRIX ANALYSIS")
    print("="*80)
    
    # Load model and data
    model, X, y, team_year_info = load_model_and_data()
    
    # Check if features exist
    if args.feature1 not in X.columns:
        print(f"ERROR: Feature '{args.feature1}' not found in dataset")
        print("\nAvailable features:")
        for i, col in enumerate(sorted(X.columns), 1):
            print(f"  {i:3d}. {col}")
        sys.exit(1)
        
    if args.feature2 not in X.columns:
        print(f"ERROR: Feature '{args.feature2}' not found in dataset")
        print("\nAvailable features:")
        for i, col in enumerate(sorted(X.columns), 1):
            print(f"  {i:3d}. {col}")
        sys.exit(1)
    
    # Run the analysis
    analyze_feature_interaction(
        model, X, args.feature1, args.feature2, 
        n_bins1=args.bins1, n_bins2=args.bins2
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()