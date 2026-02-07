"""
2024 NFL Coaches WAR Analysis - PNG Version
Shows the distribution of coaches who coached in 2024 based on their career WAR and length.
Saves as a PNG image instead of HTML.
"""

import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from figure_utils import configure_matplotlib_fonts

def create_2024_coaches_plot(font_family='serif'):
    """Create scatter plot for coaches who coached in 2024 and save as PNG."""

    # Configure matplotlib fonts
    configure_matplotlib_fonts(font_family)

    # Load the coaching impact analysis data
    print("Loading coaching impact analysis data...")
    impact_df = pd.read_csv('data/final/coaching_impact_analysis.csv')
    
    # Find coaches who coached in 2024
    coaches_2024 = impact_df[impact_df['Year'] == 2024]['Primary_Coach'].unique()
    print(f"Found {len(coaches_2024)} coaches who coached in 2024")
    
    # Calculate career statistics for all coaches
    print("Calculating career statistics for all coaches...")
    coach_stats = impact_df.groupby('Primary_Coach').agg({
        'Coaching_WAR': ['mean', 'std', 'count', 'sum'],
        'Predicted_Impact': ['mean', 'sum'],
        'Actual_Win_Pct': 'mean',
        'Predicted_With_Coach': 'mean',
        'Predicted_Replacement': 'mean'
    }).round(4)
    
    # Flatten column names
    coach_stats.columns = ['Avg_WAR', 'WAR_StdDev', 'Seasons', 'Total_WAR', 
                          'Avg_Pred_Impact', 'Total_Pred_Impact', 'Avg_Actual_Win', 
                          'Avg_Pred_Coach', 'Avg_Pred_Replace']
    coach_stats = coach_stats.reset_index()
    
    # Filter to coaches who coached in 2024
    df_2024 = coach_stats[coach_stats['Primary_Coach'].isin(coaches_2024)].copy()
    
    # Clean up coach names (remove any extra whitespace)
    df_2024['Primary_Coach'] = df_2024['Primary_Coach'].str.strip()
    
    # Convert WAR from percentage points to games (multiply by 16)
    df_2024['Avg_WAR_Games'] = df_2024['Avg_WAR'] * 16
    df_2024['Total_WAR_Games'] = df_2024['Total_WAR'] * 16
    
    # Add index numbers for each coach
    df_2024 = df_2024.reset_index(drop=True)
    df_2024['Coach_Index'] = range(1, len(df_2024) + 1)
    
    # Mark fired coaches
    fired_coaches = ['Dennis Allen', 'Mike McCarthy', 'Antonio Pierce', 
                     'Doug Pederson', 'Robert Saleh', 'Matt Eberflus', 'Jerod Mayo']
    df_2024['Status'] = df_2024['Primary_Coach'].apply(lambda x: 'Fired' if x in fired_coaches else 'Retained')
    
    print(f"\n2024 Coaches Career Statistics:")
    print(f"Total coaches: {len(df_2024)}")
    print(f"Average career length: {df_2024['Seasons'].mean():.1f} seasons")
    print(f"Average WAR per season: {df_2024['Avg_WAR_Games'].mean():+.2f} games")
    print(f"WAR range: {df_2024['Avg_WAR_Games'].min():+.2f} to {df_2024['Avg_WAR_Games'].max():+.2f} games")
    print(f"Season range: {df_2024['Seasons'].min()} to {df_2024['Seasons'].max()}")
    
    # Print fired vs retained statistics
    fired_df = df_2024[df_2024['Status'] == 'Fired']
    retained_df = df_2024[df_2024['Status'] == 'Retained']
    
    print(f"\n2024 Coaching Changes:")
    print(f"Fired: {len(fired_df)} coaches")
    print(f"Retained: {len(retained_df)} coaches")
    print(f"\nFired coaches average WAR: {fired_df['Avg_WAR_Games'].mean():+.2f} games/season")
    print(f"Retained coaches average WAR: {retained_df['Avg_WAR_Games'].mean():+.2f} games/season")
    print(f"\nFired coaches:")
    for _, coach in fired_df.iterrows():
        print(f"  - {coach['Primary_Coach']}: {coach['Avg_WAR_Games']:+.1f} games/season ({coach['Seasons']} seasons)")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 22))  # Tall figure to accommodate legend below
    
    # Create color mapping for fired vs retained
    colors = df_2024['Status'].map({'Fired': '#DC143C', 'Retained': '#1E90FF'})  # Crimson red for fired, dodger blue for retained
    
    # Create scatter plot with status-based colors
    scatter = ax.scatter(df_2024['Avg_WAR_Games'], df_2024['Seasons'], 
                        c=colors, s=800, alpha=0.9, edgecolors='black', linewidth=1.5)
    
    # Add index numbers to each point
    for _, row in df_2024.iterrows():
        ax.text(row['Avg_WAR_Games'], row['Seasons'], str(int(row['Coach_Index'])),
                ha='center', va='center', fontsize=16, fontweight='bold',
                color='black')
    
    # Add reference lines at median values
    median_war = df_2024['Avg_WAR_Games'].median()
    median_seasons = df_2024['Seasons'].median()
    
    ax.axvline(x=median_war, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axhline(y=median_seasons, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add median labels (increased font size)
    ax.text(median_war + 0.05, ax.get_ylim()[1] * 0.95, f'Median WAR: {median_war:.2f} games',
            rotation=90, va='top', ha='left', fontsize=18)
    ax.text(ax.get_xlim()[1] * 0.95, median_seasons + 0.5, f'Median Career: {median_seasons:.0f} seasons',
            rotation=0, va='bottom', ha='right', fontsize=18)

    # Customize the plot
    ax.set_xlabel('Average WAR per Season (in games)', fontsize=22, fontweight='bold')
    ax.set_ylabel('Career Length (Number of Seasons)', fontsize=22, fontweight='bold')
    
    # Style the plot
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    
    # Add legend for status colors instead of colorbar
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1E90FF', edgecolor='black', label='Retained'),
                       Patch(facecolor='#DC143C', edgecolor='black', label='Fired')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=18, title='Coach Status',
              title_fontsize=18, frameon=True, fancybox=True, shadow=True)

    # Set font for tick labels
    ax.tick_params(axis='both', labelsize=16)
    
    # Create legend below the plot
    legend_df = df_2024[['Coach_Index', 'Primary_Coach', 'Avg_WAR_Games', 'Seasons', 'Status']].sort_values('Coach_Index')
    
    # Create legend text in 4 columns
    coaches_per_col = (len(legend_df) + 3) // 4  # Round up division
    legend_lines = []
    
    # Prepare legend data in columns (without WAR values, larger font)
    cols = [[] for _ in range(4)]
    for i, (_, row) in enumerate(legend_df.iterrows()):
        col_idx = i // coaches_per_col
        if col_idx < 4:  # Safety check
            cols[col_idx].append(f"{int(row['Coach_Index']):2d}. {row['Primary_Coach']}")
    
    # Create legend text
    max_rows = max(len(col) for col in cols)
    legend_text = "COACH INDEX LEGEND\n\n"

    for row_idx in range(max_rows):
        line_parts = []
        for col_idx in range(4):
            if row_idx < len(cols[col_idx]):
                line_parts.append(f"{cols[col_idx][row_idx]:<25}")
            else:
                line_parts.append(" " * 25)
        legend_text += "".join(line_parts) + "\n"

    # Add legend as text below the plot, centered on the x-axis of the plot
    # Get the position of the axes to center on it
    ax_pos = ax.get_position()
    ax_center_x = (ax_pos.x0 + ax_pos.x1) / 2  # Center of the axes

    fig.text(ax_center_x, 0.36, legend_text, ha='center', va='top', fontsize=16,
             family='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
             edgecolor="black", alpha=0.9))

    # Adjust layout to prevent cutoff with less buffer
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.40)  # Adjusted buffer for legend
    
    # Print index to coach mapping for reference
    print(f"\nCoach Index Reference:")
    print("=" * 60)
    for _, row in legend_df.iterrows():
        war_indicator = "+" if row['Avg_WAR_Games'] >= 0 else ""
        print(f"{int(row['Coach_Index']):2d}. {row['Primary_Coach']:<25} ({war_indicator}{row['Avg_WAR_Games']:.1f} games/season)")
    
    # Save the plot
    output_file = 'analysis/outputs/png/coach_2024_matrix.png'
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none', pad_inches=0.5)
    print(f"\nPlot saved as: {output_file}")
    print("High-resolution PNG image ready for use in presentations or documents.")
    
    plt.close()  # Close the figure to free memory
    
    return df_2024

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create 2024 NFL coaches WAR analysis')
    parser.add_argument('--font', type=str, default='serif',
                       help='Font family to use (default: serif)')
    args = parser.parse_args()

    print("Creating 2024 NFL coaches WAR analysis as PNG...")
    data = create_2024_coaches_plot(font_family=args.font)
    print(f"\nAnalysis complete! Check the PNG file for the visualization.")