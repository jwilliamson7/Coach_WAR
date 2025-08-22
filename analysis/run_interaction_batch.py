"""
Batch run interaction matrix analyses for selected feature pairs
"""

import pandas as pd
import numpy as np
from xgboost_interaction_matrix import load_model_and_data, analyze_feature_interaction

def main():
    """Run batch interaction analyses."""
    
    # Load model and data once
    print("="*80)
    print("BATCH INTERACTION MATRIX ANALYSIS")
    print("="*80)
    
    model, X, y, team_year_info = load_model_and_data()
    
    # Define feature pairs to analyze
    feature_pairs = [
        # Original 8 pairs
        # (feature1, feature2, n_bins1, n_bins2, description)
        ('Avg_Starter_Experience', 'Avg_Starter_AV', 8, 8, 
         'Player Experience vs Performance'),
        
        ('Avg_Starter_AV_QB', 'PF (Points For)__oc_Norm', 8, 8,
         'QB Quality vs Offensive Coordinator Scoring'),
        
        ('Avg_Starter_AV_QB', 'NY/A Passing__oc_Norm', 8, 8,
         'QB Quality vs OC Passing Efficiency'),
        
        ('OL_Players_New', 'Avg_Starter_AV_OL', 7, 8,
         'O-Line Turnover vs O-Line Performance'),
        
        ('QB_Avg_Games_Missed_Pct', 'PF (Points For)__oc_Norm', 7, 8,
         'QB Availability vs OC Scoring'),
        
        ('num_times_hc', 'Avg_Starter_AV', 6, 8,
         'Head Coach Tenure Count vs Team Performance'),
        
        ('Avg_Starter_Age_QB', 'Avg_Starter_AV_QB', 8, 8,
         'QB Age vs QB Performance'),
        
        # New 7 pairs - Top Priority and Surprising Potential
        ('DeadCap_Pct', 'Avg_Starter_AV', 8, 8,
         'Dead Money vs Team Talent'),
        
        ('QB_Pct', 'Avg_Starter_AV_QB', 8, 8,
         'QB Salary Investment vs QB Performance'),
        
        
        ('Year', 'NY/A Passing__oc_Norm', 8, 8,
         'Evolution of Passing Offense Over Time'),
        
        ('num_yr_col_hc', 'num_yr_nfl_hc', 7, 7,
         'College vs NFL Head Coaching Background'),
        
        ('OL_Avg_Games_Missed_Pct', 'Avg_Starter_AV_OL', 8, 8,
         'O-Line Injury Rate vs Performance Quality'),
    ]
    
    # Process each pair
    results_summary = []
    
    for i, (feat1, feat2, bins1, bins2, desc) in enumerate(feature_pairs, 1):
        print(f"\n{'='*80}")
        print(f"Analysis {i}/{len(feature_pairs)}: {desc}")
        print(f"{'='*80}")
        
        try:
            # Check if features exist
            if feat1 not in X.columns:
                print(f"WARNING: {feat1} not found in dataset. Skipping...")
                continue
            if feat2 not in X.columns:
                print(f"WARNING: {feat2} not found in dataset. Skipping...")
                continue
            
            # Run analysis
            interaction_matrix, count_matrix = analyze_feature_interaction(
                model, X, feat1, feat2, n_bins1=bins1, n_bins2=bins2
            )
            
            # Collect summary statistics
            flat_predictions = interaction_matrix.values.flatten()
            flat_predictions = flat_predictions[~pd.isnan(flat_predictions)]
            
            if len(flat_predictions) > 0:
                results_summary.append({
                    'Feature1': feat1,
                    'Feature2': feat2, 
                    'Description': desc,
                    'Min_Prediction': flat_predictions.min(),
                    'Max_Prediction': flat_predictions.max(),
                    'Spread': flat_predictions.max() - flat_predictions.min(),
                    'Mean_Prediction': flat_predictions.mean(),
                    'Correlation': X[[feat1, feat2]].corr().iloc[0, 1]
                })
                
        except Exception as e:
            print(f"ERROR processing {feat1} vs {feat2}: {str(e)}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH ANALYSIS SUMMARY")
    print("="*80)
    
    if results_summary:
        import pandas as pd
        summary_df = pd.DataFrame(results_summary)
        summary_df = summary_df.sort_values('Spread', ascending=False)
        
        print("\nInteraction Effects Ranked by Prediction Spread:")
        print("-"*80)
        for idx, row in summary_df.iterrows():
            print(f"\n{row['Description']}:")
            print(f"  Features: {row['Feature1']} vs {row['Feature2']}")
            print(f"  Prediction Range: {row['Min_Prediction']:.3f} to {row['Max_Prediction']:.3f}")
            print(f"  Spread: {row['Spread']:.3f}")
            print(f"  Correlation: {row['Correlation']:.3f}")
        
        # Save summary
        summary_df.to_csv('analysis/interaction_matrices/csv/batch_analysis_summary.csv', index=False)
        print(f"\nSummary saved to analysis/interaction_matrices/csv/batch_analysis_summary.csv")
    
    print("\n" + "="*80)
    print("BATCH ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nProcessed {len(results_summary)} interaction pairs")
    print("All matrices and visualizations saved to analysis/interaction_matrices/")

if __name__ == "__main__":
    main()