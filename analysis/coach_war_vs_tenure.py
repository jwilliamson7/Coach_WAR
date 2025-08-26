"""
Coach WAR vs Tenure Visualization
Creates a scatter plot showing average coach career WAR over their number of seasons in the league.
WAR = Coaching Impact (delta in win percentage) × 17 games per season
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import argparse
import numpy as np

def load_coaching_data():
    """Load and process coaching impact data."""
    print("Loading coaching impact data...")
    df = pd.read_csv('data/final/coaching_impact_analysis.csv')
    
    print(f"Loaded {len(df)} team-year observations")
    return df

def calculate_coach_war_stats(df):
    """Calculate WAR statistics for each coach."""
    print("Calculating coach WAR statistics...")
    
    # Group by coach and calculate statistics
    coach_stats = df.groupby('Primary_Coach').agg({
        'Coaching_Impact': ['mean', 'std', 'count', 'sum'],
        'Actual_Win_Pct': 'mean',
        'Predicted_With_Coach': 'mean',
        'Predicted_Replacement': 'mean'
    }).round(6)
    
    # Flatten column names
    coach_stats.columns = ['_'.join(col).strip() for col in coach_stats.columns.values]
    coach_stats = coach_stats.rename(columns={
        'Coaching_Impact_mean': 'Avg_Impact',
        'Coaching_Impact_std': 'Impact_StdDev',
        'Coaching_Impact_count': 'Seasons',
        'Coaching_Impact_sum': 'Total_Impact',
        'Actual_Win_Pct_mean': 'Avg_Actual_Win',
        'Predicted_With_Coach_mean': 'Avg_Pred_Coach',
        'Predicted_Replacement_mean': 'Avg_Pred_Replace'
    })
    
    # Calculate WAR (Win percentage delta × 17 games per season)
    coach_stats['Avg_WAR'] = coach_stats['Avg_Impact'] * 17
    coach_stats['Total_WAR'] = coach_stats['Total_Impact'] * 17
    
    # Reset index to make Primary_Coach a column
    coach_stats = coach_stats.reset_index()
    
    # Filter out coaches with missing names
    coach_stats = coach_stats[coach_stats['Primary_Coach'].notna()]
    coach_stats = coach_stats[coach_stats['Primary_Coach'] != 'N/A']
    
    print(f"Calculated WAR for {len(coach_stats)} coaches")
    
    return coach_stats

def create_war_tenure_plot(coach_stats, highlight_coach=None):
    """Create interactive scatter plot of average WAR vs tenure."""
    print("Creating WAR vs Tenure visualization...")
    
    # Create base scatter plot
    fig = px.scatter(
        coach_stats,
        x='Seasons',
        y='Avg_WAR',
        hover_name='Primary_Coach',
        hover_data={
            'Seasons': True,
            'Avg_WAR': ':.3f',
            'Total_WAR': ':.3f',
            'Avg_Actual_Win': ':.3f',
            'Avg_Impact': ':.4f'
        },
        title='Coach Average WAR vs Number of Seasons',
        labels={
            'Seasons': 'Number of Seasons as Head Coach',
            'Avg_WAR': 'Average WAR per Season (Wins Above Replacement)'
        },
        width=1200,
        height=800
    )
    
    # Customize the plot styling
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.7,
            line=dict(width=1, color='white')
        )
    )
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, 
                  annotation_text="Replacement Level (0 WAR)")
    
    # Add trend line
    z = np.polyfit(coach_stats['Seasons'], coach_stats['Avg_WAR'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(coach_stats['Seasons'].min(), coach_stats['Seasons'].max(), 100)
    y_trend = p(x_trend)
    
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=y_trend,
        mode='lines',
        name=f'Trend Line (slope: {z[0]:.3f})',
        line=dict(color='red', width=2, dash='dot'),
        opacity=0.7
    ))
    
    # Highlight specific coach if requested
    if highlight_coach:
        highlight_data = coach_stats[coach_stats['Primary_Coach'] == highlight_coach]
        if not highlight_data.empty:
            fig.add_trace(go.Scatter(
                x=highlight_data['Seasons'],
                y=highlight_data['Avg_WAR'],
                mode='markers',
                name=f'{highlight_coach} (Highlighted)',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                hovertemplate=(
                    f'<b>{highlight_coach}</b><br>' +
                    'Seasons: %{x}<br>' +
                    'Avg WAR: %{y:.3f}<br>' +
                    '<extra></extra>'
                )
            ))
            
            coach_info = highlight_data.iloc[0]
            print(f"\n{highlight_coach} Statistics:")
            print(f"  Seasons: {coach_info['Seasons']}")
            print(f"  Average WAR: {coach_info['Avg_WAR']:.3f}")
            print(f"  Total WAR: {coach_info['Total_WAR']:.3f}")
            print(f"  Average Win%: {coach_info['Avg_Actual_Win']:.3f}")
            print(f"  Average Impact: {coach_info['Avg_Impact']:.4f}")
        else:
            print(f"Warning: Coach '{highlight_coach}' not found in data")
    
    # Update layout
    fig.update_layout(
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        plot_bgcolor='white',
        showlegend=True if highlight_coach else False
    )
    
    # Update grid styling
    fig.update_xaxes(gridcolor='lightgray', range=[0, coach_stats['Seasons'].max() + 2])
    fig.update_yaxes(gridcolor='lightgray')
    
    # Add quadrant annotations
    max_seasons = coach_stats['Seasons'].max()
    max_war = coach_stats['Avg_WAR'].max()
    min_war = coach_stats['Avg_WAR'].min()
    
    fig.add_annotation(
        x=max_seasons * 0.8, y=max_war * 0.8,
        text="High WAR<br>Long Career",
        showarrow=False,
        font=dict(size=10, color="green"),
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    fig.add_annotation(
        x=max_seasons * 0.8, y=min_war * 0.8,
        text="Low WAR<br>Long Career",
        showarrow=False,
        font=dict(size=10, color="red"),
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    fig.add_annotation(
        x=3, y=max_war * 0.8,
        text="High WAR<br>Short Career",
        showarrow=False,
        font=dict(size=10, color="orange"),
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    return fig, coach_stats

def show_summary_stats(coach_stats):
    """Display summary statistics."""
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    print(f"\nTotal coaches analyzed: {len(coach_stats)}")
    print(f"Average WAR across all coaches: {coach_stats['Avg_WAR'].mean():.3f}")
    print(f"Standard deviation of average WAR: {coach_stats['Avg_WAR'].std():.3f}")
    print(f"WAR range: {coach_stats['Avg_WAR'].min():.3f} to {coach_stats['Avg_WAR'].max():.3f}")
    
    print(f"\nAverage career length: {coach_stats['Seasons'].mean():.1f} seasons")
    print(f"Career length range: {coach_stats['Seasons'].min()} to {coach_stats['Seasons'].max()} seasons")
    
    # Show top and bottom performers
    print(f"\nTop 10 by Average WAR:")
    print(f"{'Coach':<25} {'Seasons':<8} {'Avg WAR':<10} {'Total WAR':<11} {'Win%'}")
    print("-" * 70)
    
    top_10 = coach_stats.nlargest(10, 'Avg_WAR')
    for _, row in top_10.iterrows():
        coach_name = row['Primary_Coach'][:23] if len(row['Primary_Coach']) > 23 else row['Primary_Coach']
        print(f"{coach_name:<25} {int(row['Seasons']):<8} {row['Avg_WAR']:+.3f}    {row['Total_WAR']:+.3f}     {row['Avg_Actual_Win']:.3f}")
    
    print(f"\nBottom 10 by Average WAR:")
    print(f"{'Coach':<25} {'Seasons':<8} {'Avg WAR':<10} {'Total WAR':<11} {'Win%'}")
    print("-" * 70)
    
    bottom_10 = coach_stats.nsmallest(10, 'Avg_WAR')
    for _, row in bottom_10.iterrows():
        coach_name = row['Primary_Coach'][:23] if len(row['Primary_Coach']) > 23 else row['Primary_Coach']
        print(f"{coach_name:<25} {int(row['Seasons']):<8} {row['Avg_WAR']:+.3f}    {row['Total_WAR']:+.3f}     {row['Avg_Actual_Win']:.3f}")

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Coach WAR vs Tenure Visualization')
    parser.add_argument('--highlight', type=str, 
                       help='Name of specific coach to highlight on the plot')
    args = parser.parse_args()
    
    print("="*80)
    print("COACH WAR VS TENURE ANALYSIS")
    print("WAR = Coaching Impact (win % delta) × 17 games per season")
    print("="*80)
    
    # Load and process data
    df = load_coaching_data()
    coach_stats = calculate_coach_war_stats(df)
    
    # Create visualization
    fig, processed_stats = create_war_tenure_plot(coach_stats, args.highlight)
    
    # Show summary statistics
    show_summary_stats(processed_stats)
    
    # Save the plot
    output_file = 'analysis/coach_war_vs_tenure.html'
    fig.write_html(output_file)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Interactive plot saved to: {output_file}")
    print("Open this file in your web browser to explore the visualization.")
    
    if args.highlight:
        print(f"Highlighted coach: {args.highlight}")
    
    # Save processed statistics
    stats_file = 'data/final/coach_war_tenure_stats.csv'
    processed_stats.to_csv(stats_file, index=False)
    print(f"Coach WAR statistics saved to: {stats_file}")

if __name__ == "__main__":
    main()