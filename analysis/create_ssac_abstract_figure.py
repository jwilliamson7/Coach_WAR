"""
Create the SSAC abstract figure: offensive-minus-defensive head coach WAR
by decade, with Welch 95% confidence intervals.

Filled markers denote decades whose Mann-Whitney comparison survives the
Benjamini-Hochberg adjustment (q < 0.05) in the 13-test registry; hollow
markers do not. Points above zero favor offensive-background coaches.

Inputs:
    analysis/outputs/csv/coach_background_decade_trend_analysis.csv
    analysis/outputs/csv/statistical_test_registry.csv
Output:
    latex/ssac/figures/ssac_decade_background_gap.png (Figure 1 of the SSAC abstract .docx)
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from figure_utils import configure_matplotlib_fonts

TREND_CSV = 'analysis/outputs/csv/coach_background_decade_trend_analysis.csv'
REGISTRY_CSV = 'analysis/outputs/csv/statistical_test_registry.csv'
OUTPUT_DIR = 'latex/ssac/figures'
OUTPUT_STEM = 'ssac_decade_background_gap'

OFFENSE_COLOR = '#2a78d6'   # blue: offensive-background edge
DEFENSE_COLOR = '#e34948'   # red: defensive-background edge
ZERO_COLOR = '#b5b4af'
TEXT_SECONDARY = '#52514e'
GRID_COLOR = '#e6e5e1'
LABELED_DECADES = {'1970s', '1990s', '2020s'}


def load_decade_gaps():
    trend = pd.read_csv(TREND_CSV)
    registry = pd.read_csv(REGISTRY_CSV)
    q_by_test = registry.set_index('test_id')['q_bh']
    trend['q_bh'] = trend['Year'].map(lambda y: q_by_test.get(f'decade_{y}_OvD_MW'))
    trend['bh_significant'] = trend['q_bh'] < 0.05
    return trend


def create_ssac_decade_figure(font_family='Cambria'):
    configure_matplotlib_fonts(font_family)
    # Small print figure: override the large-figure defaults from figure_utils
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'legend.fontsize': 8})

    df = load_decade_gaps()
    x = range(len(df))

    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    ax.axhline(0, color=ZERO_COLOR, linewidth=1.0, zorder=1)

    for i, row in zip(x, df.itertuples()):
        color = OFFENSE_COLOR if row.Difference > 0 else DEFENSE_COLOR
        ax.plot([i, i], [row.Diff_CI_Lower, row.Diff_CI_Upper],
                color=color, linewidth=1.5, solid_capstyle='round', zorder=2)
        ax.plot(i, row.Difference, marker='o', markersize=6,
                markerfacecolor=color if row.bh_significant else 'white',
                markeredgecolor=color, markeredgewidth=1.5, zorder=3)
        if row.Decade in LABELED_DECADES:
            # Label sits beside the point; the last decade labels leftward to stay inside the axes
            last = i == len(df) - 1
            ax.text(i - 0.16 if last else i + 0.16, row.Difference, f'{row.Difference:+.2f}',
                    fontsize=8, color=TEXT_SECONDARY, va='center',
                    ha='right' if last else 'left')

    ax.text(-0.45, 1.08, 'Offensive coaches better', fontsize=8,
            color=TEXT_SECONDARY, va='center', ha='left')
    ax.text(0.6, -1.32, 'Defensive coaches better', fontsize=8,
            color=TEXT_SECONDARY, va='center', ha='left')

    ax.set_xticks(list(x))
    ax.set_xticklabels(df['Decade'])
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.set_ylim(-1.5, 1.25)
    ax.set_ylabel('Offensive $-$ defensive WAR\n(wins per season)')
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(ZERO_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)

    handles = [
        plt.Line2D([], [], marker='o', linestyle='', markersize=5,
                   markerfacecolor=TEXT_SECONDARY, markeredgecolor=TEXT_SECONDARY,
                   label='BH-adjusted $q < 0.05$'),
        plt.Line2D([], [], marker='o', linestyle='', markersize=5,
                   markerfacecolor='white', markeredgecolor=TEXT_SECONDARY,
                   label='Not significant'),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False, handletextpad=0.2, columnspacing=1.2)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f'{OUTPUT_STEM}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    plt.close(fig)

    print(df[['Decade', 'Difference', 'Diff_CI_Lower', 'Diff_CI_Upper',
              'p_value_mw', 'q_bh', 'bh_significant']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the SSAC abstract decade figure')
    parser.add_argument('--font', type=str, default='Cambria',
                        help="Font family (default: 'Cambria', matching the SSAC template)")
    args = parser.parse_args()
    create_ssac_decade_figure(args.font)
