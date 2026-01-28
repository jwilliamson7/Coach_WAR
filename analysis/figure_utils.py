"""
Utility functions for figure generation with consistent styling.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

def configure_matplotlib_fonts(font_family='Helvetica'):
    """
    Configure matplotlib to use specified font family.

    Args:
        font_family: Font family to use (default: 'Helvetica')
                    Common options: 'Helvetica', 'Arial', 'Times New Roman', 'Cambria', 'serif'
    """
    # Set font family
    if font_family.lower() == 'serif':
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern', 'DejaVu Serif', 'Times New Roman']
    else:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_family, 'DejaVu Sans', 'Arial']

    # Set font sizes for consistency (increased for better readability)
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14

    # Other styling
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.facecolor'] = 'white'

    print(f"Matplotlib configured with font family: {font_family}")

def get_color_palette():
    """Get consistent color palette for figures."""
    return {
        'primary': '#2E86AB',      # Blue
        'secondary': '#A23B72',    # Purple/Magenta
        'tertiary': '#F18F01',     # Orange
        'quaternary': '#C73E1D',   # Red
        'highlight1': '#06A77D',   # Teal
        'highlight2': '#D4A574',   # Gold
        'neutral': '#6C757D',      # Grey
        'background': 'lightgrey'
    }
