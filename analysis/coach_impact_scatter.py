"""
2x2 Matrix: Coach WAR Career Distribution
Shows the distribution of coaches across four quadrants based on WAR impact and career length.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_coach_war_matrix():
    """Create 2x2 matrix showing distribution of coach WAR careers."""
    
    # Load the coaching impact analysis data (includes all team-years with WAR calculations)
    print("Loading coaching impact analysis data...")
    impact_df = pd.read_csv('data/final/coaching_impact_analysis.csv')
    
    # Calculate career statistics for all coaches (including those with 1-2 seasons)
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
    
    # Filter to coaches with at least 2 seasons for meaningful analysis
    df = coach_stats[coach_stats['Seasons'] >= 2].copy()
    
    # Clean up coach names (remove any extra whitespace)
    df['Primary_Coach'] = df['Primary_Coach'].str.strip()
    
    # Convert WAR from percentage points to games (multiply by 16)
    df['Avg_WAR_Games'] = df['Avg_WAR'] * 16
    df['Total_WAR_Games'] = df['Total_WAR'] * 16
    
    # Define thresholds for the 2x2 matrix
    war_threshold = df['Avg_WAR_Games'].median()  # Use median for balanced distribution
    seasons_threshold = df['Seasons'].median()  # Use median for balanced distribution
    
    print(f"\nThresholds for 2x2 matrix:")
    print(f"Average WAR threshold: {war_threshold:+.2f} games")
    print(f"Seasons threshold: {seasons_threshold:.1f}")
    
    # Create quadrant classifications
    df['Quadrant'] = 'Unknown'
    df.loc[(df['Avg_WAR_Games'] >= war_threshold) & (df['Seasons'] >= seasons_threshold), 'Quadrant'] = 'Elite (High WAR, Long Career)'
    df.loc[(df['Avg_WAR_Games'] >= war_threshold) & (df['Seasons'] < seasons_threshold), 'Quadrant'] = 'Promising (High WAR, Short Career)'
    df.loc[(df['Avg_WAR_Games'] < war_threshold) & (df['Seasons'] >= seasons_threshold), 'Quadrant'] = 'Persistent (Low WAR, Long Career)'
    df.loc[(df['Avg_WAR_Games'] < war_threshold) & (df['Seasons'] < seasons_threshold), 'Quadrant'] = 'Failed (Low WAR, Short Career)'
    
    # Define colors for each quadrant
    color_map = {
        'Elite (High WAR, Long Career)': '#2E8B57',        # Sea green
        'Promising (High WAR, Short Career)': '#FF8C00',   # Dark orange  
        'Persistent (Low WAR, Long Career)': '#DC143C',    # Crimson
        'Failed (Low WAR, Short Career)': '#8B0000'        # Dark red
    }
    
    # Create the main scatter plot with quadrants
    fig = px.scatter(
        df,
        x='Avg_WAR_Games',
        y='Seasons',
        color='Quadrant',
        color_discrete_map=color_map,
        hover_name='Primary_Coach',
        hover_data={
            'Avg_WAR_Games': ':.2f',
            'Seasons': True,
            'Total_WAR_Games': ':.2f',
            'Avg_Actual_Win': ':.3f',
            'Quadrant': False
        },
        title='Coach WAR Career Distribution (2x2 Matrix)',
        labels={
            'Avg_WAR_Games': 'Average WAR (Games Above Replacement per Season)',
            'Seasons': 'Number of Seasons Coached'
        },
        width=1200,
        height=800
    )
    
    # Customize the plot
    fig.update_traces(
        marker=dict(
            size=10,
            opacity=0.8,
            line=dict(width=1, color='white')
        )
    )
    
    # Add reference lines to create the 2x2 matrix
    fig.add_hline(y=seasons_threshold, line_dash="solid", line_color="black", line_width=2, opacity=0.7)
    fig.add_vline(x=war_threshold, line_dash="solid", line_color="black", line_width=2, opacity=0.7)
    
    # Add quadrant background shading
    fig.add_shape(type="rect", x0=war_threshold, y0=seasons_threshold, x1=df['Avg_WAR_Games'].max()*1.1, y1=df['Seasons'].max()*1.1,
                  fillcolor="green", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=war_threshold, y0=df['Seasons'].min()*0.9, x1=df['Avg_WAR_Games'].max()*1.1, y1=seasons_threshold,
                  fillcolor="orange", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=df['Avg_WAR_Games'].min()*1.1, y0=seasons_threshold, x1=war_threshold, y1=df['Seasons'].max()*1.1,
                  fillcolor="red", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=df['Avg_WAR_Games'].min()*1.1, y0=df['Seasons'].min()*0.9, x1=war_threshold, y1=seasons_threshold,
                  fillcolor="darkred", opacity=0.1, layer="below", line_width=0)
    
    # Update layout
    fig.update_layout(
        title_font_size=18,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        legend_title_font_size=16,
        plot_bgcolor='white',
        font_family="Cambria",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font_size=14
        )
    )
    
    # Update grid styling
    fig.update_xaxes(gridcolor='lightgray', gridwidth=1)
    fig.update_yaxes(gridcolor='lightgray', gridwidth=1)
    
    # Print quadrant statistics
    print(f"\nQuadrant Distribution:")
    quadrant_stats = df.groupby('Quadrant').agg({
        'Primary_Coach': 'count',
        'Avg_WAR': ['mean', 'std'],
        'Seasons': ['mean', 'std'],
        'Total_WAR': 'mean'
    }).round(4)
    
    for quadrant in df['Quadrant'].unique():
        if quadrant != 'Unknown':
            quad_data = df[df['Quadrant'] == quadrant]
            count = len(quad_data)
            pct = count / len(df) * 100
            avg_war_games = quad_data['Avg_WAR_Games'].mean()
            avg_seasons = quad_data['Seasons'].mean()
            avg_total_war_games = quad_data['Total_WAR_Games'].mean()
            
            print(f"\n{quadrant}:")
            print(f"  Count: {count} coaches ({pct:.1f}%)")
            print(f"  Avg WAR: {avg_war_games:+.2f} games/season")
            print(f"  Avg Seasons: {avg_seasons:.1f}")
            print(f"  Avg Total WAR: {avg_total_war_games:+.2f} games")
            
            # Show top coaches in each quadrant
            top_in_quad = quad_data.nlargest(3, 'Avg_WAR_Games')[['Primary_Coach', 'Avg_WAR_Games', 'Seasons']]
            print(f"  Top coaches:")
            for _, row in top_in_quad.iterrows():
                print(f"    {row['Primary_Coach']}: {row['Avg_WAR_Games']:+.2f} games/season ({row['Seasons']} seasons)")
    
    # Add overall summary statistics
    print(f"\nOverall Dataset Summary:")
    print(f"Total coaches (2+ seasons): {len(df)}")
    print(f"Average WAR: {df['Avg_WAR_Games'].mean():+.2f} games/season")
    print(f"Average seasons: {df['Seasons'].mean():.1f}")
    print(f"WAR range: {df['Avg_WAR_Games'].min():+.2f} to {df['Avg_WAR_Games'].max():+.2f} games/season")
    print(f"Season range: {df['Seasons'].min()} to {df['Seasons'].max()}")
    
    # Show most extreme coaches
    print(f"\nMost Extreme Coaches:")
    print(f"Highest WAR: {df.loc[df['Avg_WAR_Games'].idxmax(), 'Primary_Coach']} ({df['Avg_WAR_Games'].max():+.2f} games/season)")
    print(f"Lowest WAR: {df.loc[df['Avg_WAR_Games'].idxmin(), 'Primary_Coach']} ({df['Avg_WAR_Games'].min():+.2f} games/season)")
    print(f"Longest career: {df.loc[df['Seasons'].idxmax(), 'Primary_Coach']} ({df['Seasons'].max()} seasons)")
    
    # Save the plot as HTML file
    output_file = 'analysis/outputs/html/coach_war_matrix.html'
    fig.write_html(output_file)
    print(f"\nInteractive 2x2 matrix saved to: {output_file}")
    print("Open this file in your web browser to interact with the plot.")
    
    return fig, df

if __name__ == "__main__":
    print("Creating 2x2 coach WAR matrix visualization...")
    fig, data = create_coach_war_matrix()
    print("\nHover over points to see coach details!")
    print("The plot should open in your default web browser.")