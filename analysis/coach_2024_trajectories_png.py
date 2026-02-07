"""
2024 NFL Coaches Career WAR Trajectories - PNG Version
Shows cumulative WAR (in games) over entire career tenure for coaches who coached in 2024.
Includes terminal markers for coaches who were fired.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from figure_utils import configure_matplotlib_fonts

def create_2024_trajectories_plot(font_family='serif'):
    """Create line chart showing cumulative WAR trajectories for 2024 coaches."""

    # Configure matplotlib fonts
    configure_matplotlib_fonts(font_family)

    # Load the coaching impact analysis data
    print("Loading coaching impact analysis data...")
    impact_df = pd.read_csv('data/final/coaching_impact_analysis.csv')
    
    # Find coaches who coached in 2024
    coaches_2024 = impact_df[impact_df['Year'] == 2024]['Primary_Coach'].unique()
    print(f"Found {len(coaches_2024)} coaches who coached in 2024")
    
    # Mark fired coaches
    fired_coaches = ['Dennis Allen', 'Mike McCarthy', 'Antonio Pierce', 
                     'Doug Pederson', 'Robert Saleh', 'Matt Eberflus', 'Jerod Mayo']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Color palette for lines - cycle through if needed
    retained_colors = ['#1E90FF', '#32CD32', '#FF6347', '#FFD700', '#9370DB', 
                       '#20B2AA', '#FF69B4', '#FFA500', '#00CED1', '#DC143C',
                       '#8B4513', '#4682B4', '#2E8B57', '#FF1493', '#00FA9A',
                       '#CD853F', '#40E0D0', '#F08080', '#98FB98', '#DDA0DD']
    fired_color = '#8B0000'  # Dark red for fired coaches
    
    # Track data for sorting legend
    coach_data = []
    
    # Create alphabetical index mapping for ALL coaches
    sorted_coaches = sorted([coach.strip() for coach in coaches_2024])
    coach_indices = {coach: idx + 1 for idx, coach in enumerate(sorted_coaches)}
    
    # Plot each coach's trajectory
    color_idx = 0
    for coach in coaches_2024:
        coach = coach.strip()  # Clean whitespace
        
        # Get all seasons for this coach
        coach_df = impact_df[impact_df['Primary_Coach'] == coach].sort_values('Year')
        
        if len(coach_df) == 0:
            continue
            
        # Calculate cumulative WAR in games
        coach_df['Cumulative_WAR_Games'] = (coach_df['Coaching_WAR'].cumsum() * 16)
        coach_df['Season_Number'] = range(1, len(coach_df) + 1)
        
        # Determine if fired
        is_fired = coach in fired_coaches
        
        # Choose color and style
        if is_fired:
            color = fired_color
            linestyle = '--'
            linewidth = 2
            alpha = 0.9
        else:
            color = retained_colors[color_idx % len(retained_colors)]
            linestyle = '-'
            linewidth = 1.5
            alpha = 0.7
            color_idx += 1
        
        # Plot the line
        line = ax.plot(coach_df['Season_Number'], coach_df['Cumulative_WAR_Games'], 
                      color=color, linestyle=linestyle, linewidth=linewidth, 
                      alpha=alpha)
        
        # Add index number to the right of the rightmost point for ALL coaches
        ax.text(coach_df['Season_Number'].iloc[-1] + 0.3,
               coach_df['Cumulative_WAR_Games'].iloc[-1],
               str(coach_indices[coach]),
               fontsize=14, color=color, fontweight='bold',
               ha='left', va='center')
        
        # Store data for summary
        final_war = coach_df['Cumulative_WAR_Games'].iloc[-1]
        seasons = len(coach_df)
        avg_war = final_war / seasons if seasons > 0 else 0
        coach_data.append({
            'Coach': coach,
            'Seasons': seasons,
            'Final_WAR_Games': final_war,
            'Avg_WAR_Games': avg_war,
            'Status': 'Fired' if is_fired else 'Retained'
        })
    
    # Add reference line at zero
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.98, 0.5, 'Replacement Level',
            ha='right', va='bottom', fontsize=18, style='italic')

    # Customize the plot
    ax.set_xlabel('Season Number in Career', fontsize=22, fontweight='bold')
    ax.set_ylabel('Cumulative WAR (Wins Above Replacement)', fontsize=22, fontweight='bold')
    
    # Style the plot
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('white')
    ax.tick_params(axis='both', labelsize=16)
    
    # Set x-axis to show integer seasons
    max_seasons = max([d['Seasons'] for d in coach_data])
    ax.set_xticks(range(0, max_seasons + 2, 2))
    
    # Prepare data for legend (removed top left popup)
    coach_df_summary = pd.DataFrame(coach_data)
    
    # Prepare data for calculations (removed bottom right popup)
    fired_coaches_data = [c for c in coach_data if c['Status'] == 'Fired']
    retained_coaches_data = [c for c in coach_data if c['Status'] == 'Retained']
    
    fired_avg = sum(c['Avg_WAR_Games'] for c in fired_coaches_data) / len(fired_coaches_data) if fired_coaches_data else 0
    retained_avg = sum(c['Avg_WAR_Games'] for c in retained_coaches_data) / len(retained_coaches_data) if retained_coaches_data else 0
    
    # Create alphabetical index legend for ALL coaches below the plot
    all_coaches_with_indices = []
    for coach_name, idx in coach_indices.items():
        coach_row = [c for c in coach_data if c['Coach'] == coach_name]
        if coach_row:
            coach_info = coach_row[0]
            all_coaches_with_indices.append({
                'Index': idx,
                'Coach': coach_name,
                'Final_WAR_Games': coach_info['Final_WAR_Games'],
                'Seasons': coach_info['Seasons'],
                'Status': coach_info['Status']
            })
    
    all_indexed_df = pd.DataFrame(all_coaches_with_indices).sort_values('Index')
    
    # Create legend text in 4 columns like the scatter plot
    legend_text = "COACH INDEX LEGEND\n\n"
    
    # Create 4 columns
    coaches_per_col = (len(all_indexed_df) + 3) // 4  # Round up division
    
    # Prepare legend data in columns (without WAR values)
    cols = [[] for _ in range(4)]
    for i, (_, row) in enumerate(all_indexed_df.iterrows()):
        col_idx = i // coaches_per_col
        if col_idx < 4:  # Safety check
            cols[col_idx].append(f"{int(row['Index']):2d}. {row['Coach']}")
    
    # Create legend text
    max_rows = max(len(col) for col in cols)

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

    fig.text(ax_center_x, 0.33, legend_text, ha='center', va='top', fontsize=16,
             family='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
             edgecolor="black", alpha=0.9))

    # Adjust layout to prevent cutoff
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.40)
    
    # Save the plot
    output_file = 'analysis/outputs/png/coach_2024_trajectories.png'
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none', pad_inches=0.5)
    print(f"\nPlot saved as: {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 60)
    print(f"Total coaches plotted: {len(coach_data)}")
    print(f"Fired coaches: {len(fired_coaches_data)}")
    print(f"Retained coaches: {len(retained_coaches_data)}")
    print(f"\nFired coaches average WAR/season: {fired_avg:+.2f} games")
    print(f"Retained coaches average WAR/season: {retained_avg:+.2f} games")
    
    print("\nFired Coaches Final Career WAR:")
    fired_coaches_sorted = sorted(fired_coaches_data, key=lambda x: x['Final_WAR_Games'], reverse=True)
    for coach in fired_coaches_sorted:
        war_sign = "+" if coach['Final_WAR_Games'] >= 0 else ""
        coach_idx = coach_indices[coach['Coach']]
        print(f"  {coach_idx:2d}. {coach['Coach']}: {war_sign}{coach['Final_WAR_Games']:.1f} games over {coach['Seasons']} seasons")
    
    print("\nTop 5 Retained Coaches by Final Career WAR:")
    retained_coaches_sorted = sorted(retained_coaches_data, key=lambda x: x['Final_WAR_Games'], reverse=True)
    for coach in retained_coaches_sorted[:5]:
        war_sign = "+" if coach['Final_WAR_Games'] >= 0 else ""
        coach_idx = coach_indices[coach['Coach']]
        print(f"  {coach_idx:2d}. {coach['Coach']}: {war_sign}{coach['Final_WAR_Games']:.1f} games over {coach['Seasons']} seasons")
    
    print("\nBottom 5 Retained Coaches by Final Career WAR:")
    for coach in retained_coaches_sorted[-5:]:
        war_sign = "+" if coach['Final_WAR_Games'] >= 0 else ""
        coach_idx = coach_indices[coach['Coach']]
        print(f"  {coach_idx:2d}. {coach['Coach']}: {war_sign}{coach['Final_WAR_Games']:.1f} games over {coach['Seasons']} seasons")
    
    plt.close()  # Close the figure to free memory
    
    return coach_df_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create 2024 NFL coaches career WAR trajectories')
    parser.add_argument('--font', type=str, default='serif',
                       help='Font family to use (default: serif)')
    args = parser.parse_args()

    print("Creating 2024 NFL coaches career WAR trajectories as PNG...")
    data = create_2024_trajectories_plot(font_family=args.font)
    print(f"\nAnalysis complete! Check the PNG file for the visualization.")
    print("Fired coaches are shown with dashed lines and X markers at career end.")