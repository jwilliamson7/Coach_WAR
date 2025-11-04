"""
Create publication-quality PNG figure showing coach WAR trajectories
for the LaTeX paper. All coaches in grey background, with O'Connell,
Shula, and Eberflus highlighted in color.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from figure_utils import configure_matplotlib_fonts, get_color_palette

def create_three_coach_trajectory_figure(font_family='Helvetica'):
    """Create Figure 1 for the LaTeX paper."""
    # Configure matplotlib fonts
    configure_matplotlib_fonts(font_family)
    colors = get_color_palette()

    print("Loading coach WAR trajectory data...")
    df = pd.read_csv('data/final/coach_war_trajectories.csv')

    # Define the three coaches to highlight
    highlight_coaches = ['Kevin O\'Connell', 'Don Shula', 'Matt Eberflus']

    # Create figure with appropriate size for publication
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all coaches in light grey background
    print("Plotting background trajectories...")
    all_coaches = df['Coach'].unique()
    for coach in all_coaches:
        if coach not in highlight_coaches:
            coach_data = df[df['Coach'] == coach]
            ax.plot(coach_data['Season_Number'],
                   coach_data['Cumulative_Games'],
                   color='lightgrey',
                   linewidth=0.8,
                   alpha=0.3,
                   zorder=1)

    # Plot highlighted coaches in distinct colors
    print("Plotting highlighted coaches...")
    coach_colors = {
        'Kevin O\'Connell': colors['primary'],    # Blue
        'Don Shula': colors['secondary'],          # Purple/Magenta
        'Matt Eberflus': colors['tertiary']        # Orange
    }

    for coach in highlight_coaches:
        coach_data = df[df['Coach'] == coach]
        if len(coach_data) > 0:
            ax.plot(coach_data['Season_Number'],
                   coach_data['Cumulative_Games'],
                   color=coach_colors[coach],
                   linewidth=2.5,
                   label=coach,
                   marker='o',
                   markersize=4,
                   zorder=3)

            # Print statistics
            final_war = coach_data['Cumulative_WAR'].iloc[-1]
            final_games = coach_data['Cumulative_Games'].iloc[-1]
            seasons = coach_data['Season_Number'].max()
            avg_war_per_season = final_war / seasons

            print(f"\n{coach}:")
            print(f"  Seasons: {seasons}")
            print(f"  Final Cumulative WAR: {final_war:.3f} ({final_games:.1f} games)")
            print(f"  Avg WAR/season: {avg_war_per_season:.3f}")

    # Add horizontal line at 0 (replacement level)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1,
               alpha=0.5, zorder=2, label='Replacement Level')

    # Formatting
    ax.set_xlabel('Season Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative WAR (Games Above Replacement)', fontsize=12, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper left', frameon=True, framealpha=0.95,
             edgecolor='black', fontsize=10)

    # Set x-axis to start at 1
    ax.set_xlim(left=0.5)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = 'latex/figures/coach_trajectories_oconnell_shula_eberflus.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")

    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create three-coach trajectory figure')
    parser.add_argument('--font', type=str, default='Helvetica',
                       help='Font family to use (default: Helvetica)')
    args = parser.parse_args()

    print("="*80)
    print("Creating Three-Coach Trajectory Figure for LaTeX Paper")
    print(f"Font: {args.font}")
    print("="*80)
    create_three_coach_trajectory_figure(font_family=args.font)
    print("\nComplete!")
