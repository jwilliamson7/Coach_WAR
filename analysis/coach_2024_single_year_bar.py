"""
2024 NFL Coaches Single Year WAR - Horizontal Bar Chart
Shows 2024 WAR performance for all coaches who coached in 2024, ordered by performance.
Color coded: Red for fired coaches, Blue for retained coaches.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_2024_single_year_bar_chart():
    """Create horizontal bar chart for 2024 single year WAR performance."""
    
    # Load the coaching impact analysis data
    print("Loading coaching impact analysis data...")
    impact_df = pd.read_csv('data/final/coaching_impact_analysis.csv')
    
    # Filter to 2024 data only
    df_2024 = impact_df[impact_df['Year'] == 2024].copy()
    print(f"Found {len(df_2024)} coaches who coached in 2024")
    
    # Clean up coach names
    df_2024['Primary_Coach'] = df_2024['Primary_Coach'].str.strip()
    
    # Convert WAR from percentage points to games (multiply by 16)
    df_2024['WAR_Games_2024'] = df_2024['Coaching_WAR'] * 16
    
    # Mark fired coaches
    fired_coaches = ['Dennis Allen', 'Mike McCarthy', 'Antonio Pierce', 
                     'Doug Pederson', 'Robert Saleh', 'Matt Eberflus', 'Jerod Mayo']
    df_2024['Status'] = df_2024['Primary_Coach'].apply(lambda x: 'Fired' if x in fired_coaches else 'Retained')
    
    # Sort by WAR performance (highest to lowest)
    df_2024 = df_2024.sort_values('WAR_Games_2024', ascending=True)  # ascending=True for horizontal bars (bottom to top)
    
    print(f"\n2024 Single Season WAR Performance:")
    print(f"Best performing coach: {df_2024.iloc[-1]['Primary_Coach']} ({df_2024.iloc[-1]['WAR_Games_2024']:+.1f} games)")
    print(f"Worst performing coach: {df_2024.iloc[0]['Primary_Coach']} ({df_2024.iloc[0]['WAR_Games_2024']:+.1f} games)")
    print(f"Average 2024 WAR: {df_2024['WAR_Games_2024'].mean():+.2f} games")
    
    # Count fired vs retained
    fired_df = df_2024[df_2024['Status'] == 'Fired']
    retained_df = df_2024[df_2024['Status'] == 'Retained']
    
    print(f"\n2024 Performance by Status:")
    print(f"Fired coaches average: {fired_df['WAR_Games_2024'].mean():+.2f} games")
    print(f"Retained coaches average: {retained_df['WAR_Games_2024'].mean():+.2f} games")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 16))
    
    # Create color mapping
    colors = df_2024['Status'].map({'Fired': '#DC143C', 'Retained': '#1E90FF'})  # Crimson red for fired, dodger blue for retained
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(df_2024)), df_2024['WAR_Games_2024'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Set y-tick labels to coach names
    ax.set_yticks(range(len(df_2024)))
    ax.set_yticklabels(df_2024['Primary_Coach'], fontsize=14, family='Cambria')
    
    # Remove awkward spacing above and below bars
    ax.set_ylim(-0.5, len(df_2024) - 0.5)
    
    # Add a vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('2024 WAR (Wins Above Replacement)', fontsize=20, fontweight='bold', family='Cambria')
    
    # Style the plot
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_facecolor('white')
    
    # Set font for tick labels
    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_xticklabels():
        label.set_family('Cambria')
    
    # Add legend for status colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1E90FF', edgecolor='black', label='Retained'),
                       Patch(facecolor='#DC143C', edgecolor='black', label='Fired')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=16, title='Coach Status', 
              title_fontsize=16, frameon=True, fancybox=True, shadow=True)
    
    # Add value labels on bars for easier reading
    for i, (_, row) in enumerate(df_2024.iterrows()):
        value = row['WAR_Games_2024']
        # Position text to the right of positive bars, left of negative bars
        if value >= 0:
            x_pos = value + 0.1
            ha = 'left'
        else:
            x_pos = value - 0.1
            ha = 'right'
        
        ax.text(x_pos, i, f'{value:+.1f}', 
                va='center', ha=ha, fontsize=12, fontweight='bold', family='Cambria')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = 'analysis/outputs/png/coach_2024_single_year_bar.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nPlot saved as: {output_file}")
    
    # Print detailed results
    print("\n2024 WAR Performance (ordered worst to best):")
    print("=" * 60)
    for _, row in df_2024.iterrows():
        status_symbol = "*" if row['Status'] == 'Fired' else " "
        print(f"{status_symbol} {row['Primary_Coach']:<25} {row['WAR_Games_2024']:+6.1f} games ({row['Status']})")
    
    plt.close()  # Close the figure to free memory
    
    return df_2024

if __name__ == "__main__":
    print("Creating 2024 NFL coaches single year WAR bar chart...")
    data = create_2024_single_year_bar_chart()
    print(f"\nAnalysis complete! Check the PNG file for the visualization.")
    print("Red bars = Fired coaches, Blue bars = Retained coaches")