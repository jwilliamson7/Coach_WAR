"""
Simple script to remove all AV (Approximate Value) metrics from the dataset.
Creates a new dataset without AV features for analysis.
"""

import pandas as pd

def remove_av_features():
    """Remove all AV features from imputed_final_data.csv"""
    
    print("Loading imputed_final_data.csv...")
    df = pd.read_csv('data/final/imputed_final_data.csv')
    
    print(f"Original dataset shape: {df.shape}")
    
    # Find all columns containing "AV" in the name
    av_columns = [col for col in df.columns if 'AV' in col]
    
    print(f"Found {len(av_columns)} AV features to remove:")
    for col in av_columns:
        print(f"  - {col}")
    
    # Remove AV columns
    df_no_av = df.drop(columns=av_columns)
    
    print(f"New dataset shape: {df_no_av.shape}")
    print(f"Features removed: {len(av_columns)}")
    print(f"Features remaining: {df_no_av.shape[1]}")
    
    # Save the new dataset
    output_file = 'data/final/imputed_final_data_no_AV.csv'
    df_no_av.to_csv(output_file, index=False)
    
    print(f"\nDataset without AV features saved to: {output_file}")
    
    return df_no_av, av_columns

if __name__ == "__main__":
    df_no_av, removed_columns = remove_av_features()