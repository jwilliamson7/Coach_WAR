"""
Summarize the interaction analysis results
"""

import pandas as pd
import os

# Results from the batch run
results = [
    {
        'Feature1': 'Avg_Starter_Experience',
        'Feature2': 'Avg_Starter_AV',
        'Description': 'Player Experience vs Performance',
        'Min_Pred': 0.181,
        'Max_Pred': 0.762,
        'Spread': 0.581,
        'Correlation': 0.328,
        'Best_Combo': 'High Exp (5.5-8.0) + High AV (8.6-11.2)',
        'Worst_Combo': 'Mid Exp (4.2-4.5) + Low AV (3.5-5.5)'
    },
    {
        'Feature1': 'Avg_Starter_AV_QB',
        'Feature2': 'PF (Points For)__oc_Norm',
        'Description': 'QB Quality vs OC Scoring',
        'Min_Pred': 0.272,
        'Max_Pred': 0.706,
        'Spread': 0.435,
        'Correlation': 0.007,
        'Best_Combo': 'Elite QB (15-25) + High OC Scoring (0.6-2.9)',
        'Worst_Combo': 'Poor QB (-3-5) + Low OC Scoring (-2.2-0)'
    },
    {
        'Feature1': 'Avg_Starter_AV_QB',
        'Feature2': 'NY/A Passing__oc_Norm',
        'Description': 'QB Quality vs OC Passing Efficiency',
        'Min_Pred': 0.274,
        'Max_Pred': 0.714,
        'Spread': 0.441,
        'Correlation': 0.021,
        'Best_Combo': 'Elite QB (15-25) + Low OC Pass Eff (-2.4-0)',
        'Worst_Combo': 'Poor QB (-3-5) + Low OC Pass Eff (-2.4-0)'
    },
    {
        'Feature1': 'OL_Players_New',
        'Feature2': 'Avg_Starter_AV_OL',
        'Description': 'O-Line Turnover vs Performance',
        'Min_Pred': 0.239,
        'Max_Pred': 0.737,
        'Spread': 0.498,
        'Correlation': -0.211,
        'Best_Combo': 'Low Turnover (2-3) + High OL AV (9.4-13.8)',
        'Worst_Combo': 'High Turnover (5-13) + Low OL AV (2-5)'
    },
    {
        'Feature1': 'QB_Avg_Games_Missed_Pct',
        'Feature2': 'PF (Points For)__oc_Norm',
        'Description': 'QB Availability vs OC Scoring',
        'Min_Pred': 0.270,
        'Max_Pred': 0.608,
        'Spread': 0.338,
        'Correlation': -0.003,
        'Best_Combo': 'Healthy QB (-6-0%) + Mid OC Scoring (0-0.6)',
        'Worst_Combo': 'Injured QB (44-69%) + High OC Scoring (0.6-2.9)'
    },
    {
        'Feature1': 'num_times_hc',
        'Feature2': 'Avg_Starter_AV',
        'Description': 'HC Tenure Count vs Team Performance',
        'Min_Pred': 0.183,
        'Max_Pred': 0.756,
        'Spread': 0.573,
        'Correlation': 0.101,
        'Best_Combo': 'Second-time HC (1-2) + High AV (8.6-11.2)',
        'Worst_Combo': 'First-time HC (0-1) + Low AV (3.5-5.5)'
    },
    {
        'Feature1': 'Avg_Starter_Age',
        'Feature2': 'Avg_Starter_Experience',
        'Description': 'Player Age vs Experience',
        'Min_Pred': 0.250,
        'Max_Pred': 0.737,
        'Spread': 0.486,
        'Correlation': 0.961,
        'Best_Combo': 'Age 27.2-27.5 + Exp 5.0-5.5',
        'Worst_Combo': 'Age 27.5-27.8 + Exp 3.9-4.2'
    },
    {
        'Feature1': 'Avg_Starter_Age_QB',
        'Feature2': 'Avg_Starter_AV_QB',
        'Description': 'QB Age vs QB Performance',
        'Min_Pred': 0.255,
        'Max_Pred': 0.759,
        'Spread': 0.504,
        'Correlation': 0.089,
        'Best_Combo': 'Prime Age QB (27-28) + Elite AV (15-25)',
        'Worst_Combo': 'Young QB (21-24) + Poor AV (-3-5)'
    }
]

# Create DataFrame
df = pd.DataFrame(results)

# Sort by spread
df = df.sort_values('Spread', ascending=False)

print("="*100)
print("INTERACTION MATRIX ANALYSIS SUMMARY")
print("="*100)
print("\nRanked by Prediction Spread (Largest Effect Size):\n")

for idx, row in df.iterrows():
    print(f"{row['Description']}")
    print(f"  Features: {row['Feature1']} vs {row['Feature2']}")
    print(f"  Win% Range: {row['Min_Pred']:.1%} to {row['Max_Pred']:.1%} (Spread: {row['Spread']:.1%})")
    print(f"  Correlation: {row['Correlation']:.3f}")
    print(f"  Best: {row['Best_Combo']}")
    print(f"  Worst: {row['Worst_Combo']}")
    print()

# Key insights
print("="*100)
print("KEY INSIGHTS")
print("="*100)

print("\n1. LARGEST INTERACTION EFFECTS:")
top3 = df.nlargest(3, 'Spread')
for idx, row in top3.iterrows():
    print(f"   - {row['Description']}: {row['Spread']:.1%} spread")

print("\n2. SURPRISING FINDINGS:")
print("   - QB quality + OC passing efficiency shows NEGATIVE synergy")
print("     (Elite QBs do better with LOWER OC passing efficiency scores)")
print("   - QB injuries hurt MORE when OC scoring is high")
print("     (Perhaps backup QBs can't execute sophisticated offenses)")
print("   - Age and Experience are 96% correlated but still show interaction effects")

print("\n3. COACHING INSIGHTS:")
print("   - Second-time head coaches (num_times_hc=1-2) perform best")
print("   - O-Line continuity matters: Low turnover + high performance = 73.7% win rate")

print("\n4. CORRELATION VS INTERACTION:")
low_corr = df[abs(df['Correlation']) < 0.1]
print(f"   - {len(low_corr)} pairs have near-zero correlation but strong interaction effects")
print("     (Features can be independent but still interact in predictions)")

# Save summary
output_file = 'analysis/interaction_matrices/csv/interaction_summary.csv'
df.to_csv(output_file, index=False)
print(f"\nSummary saved to {output_file}")