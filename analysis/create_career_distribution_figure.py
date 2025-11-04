"""
Create publication-quality PNG figure showing the relationship between
coaching quality (avg WAR per season) and career length (seasons coached).
Includes median lines to show quadrants.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from figure_utils import configure_matplotlib_fonts, get_color_palette

def create_career_distribution_figure(font_family='Helvetica'):
    """Create Figure 2 for the LaTeX paper."""
    # Configure matplotlib fonts
    configure_matplotlib_fonts(font_family)
    colors_palette = get_color_palette()

    print("Loading coach career statistics...")
    df = pd.read_csv('data/final/coach_career_impact_stats.csv')

    print(f"Loaded {len(df)} coaches")

    # Calculate medians for quadrant lines
    median_seasons = df['Seasons'].median()
    median_war = df['Avg_WAR'].median()

    print(f"\nMedian career length: {median_seasons} seasons")
    print(f"Median avg WAR: {median_war:.4f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define quadrants for color coding
    df['Quadrant'] = 'Unknown'
    df.loc[(df['Seasons'] >= median_seasons) & (df['Avg_WAR'] >= median_war), 'Quadrant'] = 'High Quality, Long Career'
    df.loc[(df['Seasons'] < median_seasons) & (df['Avg_WAR'] >= median_war), 'Quadrant'] = 'High Quality, Short Career'
    df.loc[(df['Seasons'] >= median_seasons) & (df['Avg_WAR'] < median_war), 'Quadrant'] = 'Low Quality, Long Career'
    df.loc[(df['Seasons'] < median_seasons) & (df['Avg_WAR'] < median_war), 'Quadrant'] = 'Low Quality, Short Career'

    # Color mapping for quadrants
    colors = {
        'High Quality, Long Career': colors_palette['primary'],      # Blue - efficient market
        'High Quality, Short Career': colors_palette['secondary'],   # Purple - cut too soon
        'Low Quality, Long Career': colors_palette['tertiary'],      # Orange - inefficient market
        'Low Quality, Short Career': colors_palette['quaternary']    # Red - efficient market
    }

    # Plot points by quadrant
    for quadrant, color in colors.items():
        quad_data = df[df['Quadrant'] == quadrant]
        ax.scatter(quad_data['Seasons'], quad_data['Avg_WAR'],
                  c=color, s=60, alpha=0.6, edgecolors='black',
                  linewidths=0.5, label=quadrant, zorder=3)

    # Add median lines
    ax.axhline(y=median_war, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, zorder=2, label=f'Median WAR ({median_war:.3f})')
    ax.axvline(x=median_seasons, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, zorder=2, label=f'Median Seasons ({median_seasons:.0f})')

    # Add horizontal line at 0 (replacement level)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1,
               alpha=0.5, zorder=2, label='Replacement Level')

    # Annotate a few notable coaches
    notable_coaches = [
        'Don Shula', 'Tom Landry', 'Bill Belichick',  # Legends (top right)
        'Kevin O\'Connell', 'Mike McDaniel',           # Rising stars (top left)
        'Pat Shurmur', 'Matt Eberflus'                 # Poor performers
    ]

    for coach in notable_coaches:
        coach_data = df[df['Primary_Coach'] == coach]
        if len(coach_data) > 0:
            x = coach_data['Seasons'].values[0]
            y = coach_data['Avg_WAR'].values[0]

            # Offset annotation to avoid overlapping points
            if y > median_war:
                xytext = (0, 10)  # Above
            else:
                xytext = (0, -15)  # Below

            ax.annotate(coach,
                       xy=(x, y),
                       xytext=xytext,
                       textcoords='offset points',
                       fontsize=8,
                       ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='gray', alpha=0.8),
                       zorder=4)

    # Count coaches in each quadrant
    print("\nQuadrant Breakdown:")
    for quadrant in colors.keys():
        count = len(df[df['Quadrant'] == quadrant])
        print(f"  {quadrant}: {count} coaches")

    # Formatting
    ax.set_xlabel('Career Length (Seasons Coached)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average WAR per Season', fontsize=16, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend - place in bottom right corner of plot
    ax.legend(loc='lower right',
             frameon=True, framealpha=0.95, edgecolor='black',
             fontsize=14)

    # Set reasonable axis limits
    ax.set_xlim(left=0)

    # Tight layout to accommodate legend
    plt.tight_layout()

    # Save figure
    output_path = 'latex/figures/coach_career_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")

    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create career distribution figure')
    parser.add_argument('--font', type=str, default='Helvetica',
                       help='Font family to use (default: Helvetica)')
    args = parser.parse_args()

    print("="*80)
    print("Creating Career Distribution Figure for LaTeX Paper")
    print(f"Font: {args.font}")
    print("="*80)
    create_career_distribution_figure(font_family=args.font)
    print("\nComplete!")
