"""
2024 NFL Coaches WAR Analysis
Shows the distribution of coaches who coached in 2024 based on their career WAR and length.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def create_2024_coaches_plot():
    """Create scatter plot for coaches who coached in 2024."""
    
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
    
    print(f"\n2024 Coaches Career Statistics:")
    print(f"Total coaches: {len(df_2024)}")
    print(f"Average career length: {df_2024['Seasons'].mean():.1f} seasons")
    print(f"Average WAR per season: {df_2024['Avg_WAR_Games'].mean():+.2f} games")
    print(f"WAR range: {df_2024['Avg_WAR_Games'].min():+.2f} to {df_2024['Avg_WAR_Games'].max():+.2f} games")
    print(f"Season range: {df_2024['Seasons'].min()} to {df_2024['Seasons'].max()}")
    
    # Create the scatter plot with index numbers
    fig = px.scatter(
        df_2024,
        x='Avg_WAR_Games',
        y='Seasons',
        text='Coach_Index',
        color='Avg_WAR_Games',
        color_continuous_scale='RdYlGn',
        hover_data={
            'Primary_Coach': True,
            'Coach_Index': True,
            'Avg_WAR_Games': ':.2f',
            'Seasons': True,
            'Total_WAR_Games': ':.1f',
            'Avg_Actual_Win': ':.3f'
        },
        labels={
            'Avg_WAR_Games': 'Average WAR (Games per Season)',
            'Seasons': 'Career Length (Seasons)',
            'Primary_Coach': 'Coach',
            'Coach_Index': 'Index'
        }
    )
    
    # Update traces styling with larger markers and centered text
    fig.update_traces(
        textposition='middle center',
        textfont=dict(size=12, family="Cambria", color='black'),
        marker=dict(
            size=30,
            line=dict(width=1, color='black'),
            opacity=0.9
        )
    )
    
    # Add reference lines at median values
    median_war = df_2024['Avg_WAR_Games'].median()
    median_seasons = df_2024['Seasons'].median()
    
    fig.add_vline(x=median_war, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text=f"Median WAR: {median_war:.2f} games")
    fig.add_hline(y=median_seasons, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text=f"Median Career: {median_seasons:.0f} seasons")
    
    # Update layout and colorbar
    fig.update_layout(
        title='2024 NFL Coaches: Career WAR vs Experience',
        title_font_size=18,
        xaxis_title='Average WAR (Games Above Replacement per Season)',
        yaxis_title='Career Length (Number of Seasons)',
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        legend_title_font_size=16,
        width=1400,
        height=1200,  # Increased height to accommodate legend below
        plot_bgcolor='white',
        font_family="Cambria",
        showlegend=False,
        margin=dict(b=300)  # Add bottom margin for legend
    )
    
    # Update colorbar
    fig.update_coloraxes(
        colorbar=dict(
            title=dict(
                text="Average WAR<br>(Games per Season)",
                font=dict(size=14, family="Cambria")
            ),
            tickfont=dict(size=12, family="Cambria")
        )
    )
    
    # Update grid styling
    fig.update_xaxes(gridcolor='lightgray', gridwidth=1)
    fig.update_yaxes(gridcolor='lightgray', gridwidth=1)
    
    # Create legend table mapping index to coach names
    # Sort by index for clean presentation
    legend_df = df_2024[['Coach_Index', 'Primary_Coach', 'Avg_WAR_Games', 'Seasons']].sort_values('Coach_Index')
    
    # Create legend below the plot in 4 columns for better space usage
    legend_text = "<b>COACH INDEX LEGEND</b><br><br>"
    
    # Split into 4 columns for better horizontal space usage
    coaches_per_col = (len(legend_df) + 3) // 4  # Round up division
    
    legend_text += "<table style='font-family:Cambria; font-size:10px; width:100%;'><tr>"
    
    for col in range(4):
        start_idx = col * coaches_per_col
        end_idx = min(start_idx + coaches_per_col, len(legend_df))
        col_data = legend_df.iloc[start_idx:end_idx]
        
        legend_text += "<td valign='top' style='text-align:left; padding-right:20px;'>"
        
        for _, row in col_data.iterrows():
            war_sign = "+" if row['Avg_WAR_Games'] >= 0 else ""
            legend_text += f"{int(row['Coach_Index']):2d}. {row['Primary_Coach']:<20} {war_sign}{row['Avg_WAR_Games']:.1f}<br>"
        
        legend_text += "</td>"
    
    legend_text += "</tr></table>"
    
    # Add the legend below the plot
    fig.add_annotation(
        x=0.5, y=-0.2,
        xref="paper", yref="paper",
        text=legend_text,
        showarrow=False,
        xanchor="center",
        yanchor="top",
        bgcolor="rgba(255,255,255,1.0)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10, family="Cambria"),
        align="left"
    )
    
    # Print summary statistics by quadrants
    print(f"\nQuadrant Analysis (using median splits):")
    print(f"Median WAR threshold: {median_war:+.2f} games/season")
    print(f"Median seasons threshold: {median_seasons:.0f} seasons")
    
    high_war_long = df_2024[(df_2024['Avg_WAR_Games'] >= median_war) & (df_2024['Seasons'] >= median_seasons)]
    high_war_short = df_2024[(df_2024['Avg_WAR_Games'] >= median_war) & (df_2024['Seasons'] < median_seasons)]
    low_war_long = df_2024[(df_2024['Avg_WAR_Games'] < median_war) & (df_2024['Seasons'] >= median_seasons)]
    low_war_short = df_2024[(df_2024['Avg_WAR_Games'] < median_war) & (df_2024['Seasons'] < median_seasons)]
    
    print(f"\nElite (High WAR, Long Career): {len(high_war_long)} coaches")
    if len(high_war_long) > 0:
        for _, coach in high_war_long.iterrows():
            print(f"  {coach['Primary_Coach']}: {coach['Avg_WAR_Games']:+.2f} games/season ({coach['Seasons']} seasons)")
    
    print(f"\nPromising (High WAR, Short Career): {len(high_war_short)} coaches")
    if len(high_war_short) > 0:
        for _, coach in high_war_short.iterrows():
            print(f"  {coach['Primary_Coach']}: {coach['Avg_WAR_Games']:+.2f} games/season ({coach['Seasons']} seasons)")
    
    print(f"\nPersistent (Low WAR, Long Career): {len(low_war_long)} coaches")
    if len(low_war_long) > 0:
        for _, coach in low_war_long.iterrows():
            print(f"  {coach['Primary_Coach']}: {coach['Avg_WAR_Games']:+.2f} games/season ({coach['Seasons']} seasons)")
    
    print(f"\nStruggling (Low WAR, Short Career): {len(low_war_short)} coaches")
    if len(low_war_short) > 0:
        for _, coach in low_war_short.iterrows():
            print(f"  {coach['Primary_Coach']}: {coach['Avg_WAR_Games']:+.2f} games/season ({coach['Seasons']} seasons)")
    
    # Show extreme coaches
    print(f"\nExtreme Performers among 2024 coaches:")
    best_war = df_2024.loc[df_2024['Avg_WAR_Games'].idxmax()]
    worst_war = df_2024.loc[df_2024['Avg_WAR_Games'].idxmin()]
    longest_career = df_2024.loc[df_2024['Seasons'].idxmax()]
    
    print(f"Highest WAR: {best_war['Primary_Coach']} ({best_war['Avg_WAR_Games']:+.2f} games/season)")
    print(f"Lowest WAR: {worst_war['Primary_Coach']} ({worst_war['Avg_WAR_Games']:+.2f} games/season)")
    print(f"Longest career: {longest_career['Primary_Coach']} ({longest_career['Seasons']} seasons)")
    
    # Print index to coach mapping for reference
    print(f"\nCoach Index Reference:")
    print("=" * 60)
    for _, row in legend_df.iterrows():
        war_indicator = "+" if row['Avg_WAR_Games'] >= 0 else ""
        print(f"{int(row['Coach_Index']):2d}. {row['Primary_Coach']:<25} ({war_indicator}{row['Avg_WAR_Games']:.1f} games/season)")
    
    # Save the plot
    output_file = 'analysis/outputs/html/coach_2024_matrix.html'
    fig.write_html(output_file)
    print(f"\nInteractive plot saved to: {output_file}")
    print("Open this file in your web browser to interact with the plot.")
    print("Each coach is represented by their index number in a colored circle.")
    print("The legend table on the right maps index numbers to coach names.")
    
    return fig, df_2024

if __name__ == "__main__":
    print("Creating 2024 NFL coaches WAR analysis...")
    fig, data = create_2024_coaches_plot()
    print("\nHover over points to see coach details!")
    print("Coach names are labeled directly on the plot.")