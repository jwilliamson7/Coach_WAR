"""
Coach WAR Career Trajectory Visualization
Traces the cumulative WAR over time for coaches, showing how their career WAR builds season by season.
WAR = Actual Win% - Replacement-Level Prediction (wins above replacement)
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
    
    # Filter out rows with missing coach information
    df = df[df['Primary_Coach'].notna()]
    df = df[df['Primary_Coach'] != 'N/A']
    
    print(f"Loaded {len(df)} team-year observations with valid coach data")
    return df

def calculate_coach_trajectories(df):
    """Calculate cumulative WAR trajectories for each coach."""
    print("Calculating coach WAR trajectories...")
    
    # Use the pre-calculated Coaching_WAR (Actual Win% - Replacement Prediction)
    trajectories = []
    
    for coach in df['Primary_Coach'].unique():
        coach_data = df[df['Primary_Coach'] == coach].copy()
        
        # Sort by year to get chronological order
        coach_data = coach_data.sort_values('Year')
        
        # Calculate cumulative WAR
        coach_data['Cumulative_WAR'] = coach_data['Coaching_WAR'].cumsum()
        coach_data['Season_Number'] = range(1, len(coach_data) + 1)
        
        # Add coach trajectory to list
        for _, row in coach_data.iterrows():
            # Calculate games equivalent (WAR as percentage * 16 games)
            annual_games = row['Coaching_WAR'] * 16
            cumulative_games = row['Cumulative_WAR'] * 16
            
            trajectories.append({
                'Coach': coach,
                'Year': row['Year'],
                'Season_Number': row['Season_Number'],
                'Annual_WAR': row['Coaching_WAR'],
                'Annual_Games': annual_games,
                'Cumulative_WAR': row['Cumulative_WAR'],
                'Cumulative_Games': cumulative_games,
                'Win_Pct': row['Actual_Win_Pct'],
                'Predicted_Impact': row['Predicted_Impact'],
                'Predicted_Replacement': row['Predicted_Replacement'],
                'Total_Seasons': len(coach_data)
            })
    
    trajectory_df = pd.DataFrame(trajectories)
    
    print(f"Calculated trajectories for {trajectory_df['Coach'].nunique()} coaches")
    
    return trajectory_df

def create_trajectory_plot(trajectory_df, highlight_coaches=None, min_seasons=1):
    """Create interactive line plot of cumulative WAR trajectories."""
    print(f"Creating WAR trajectory visualization (min {min_seasons} seasons)...")
    
    # Filter for coaches with minimum seasons
    coach_seasons = trajectory_df.groupby('Coach')['Season_Number'].max()
    eligible_coaches = coach_seasons[coach_seasons >= min_seasons].index
    plot_df = trajectory_df[trajectory_df['Coach'].isin(eligible_coaches)].copy()
    
    print(f"Plotting {len(eligible_coaches)} coaches with {min_seasons}+ seasons")
    
    # Create the plot
    if highlight_coaches:
        # Plot all coaches in background (light gray)
        background_coaches = plot_df[~plot_df['Coach'].isin(highlight_coaches)]
        
        fig = go.Figure()
        
        # Add background trajectories
        for coach in background_coaches['Coach'].unique():
            coach_data = background_coaches[background_coaches['Coach'] == coach]
            
            fig.add_trace(go.Scatter(
                x=coach_data['Season_Number'],
                y=coach_data['Cumulative_Games'],
                mode='lines',
                name=coach,
                line=dict(color='lightgray', width=1),
                opacity=0.3,
                showlegend=False,
                hovertemplate=(
                    f'<b>{coach}</b><br>' +
                    'Season: %{x}<br>' +
                    'Cumulative WAR: %{y:.1f} games<br>' +
                    '<extra></extra>'
                )
            ))
        
        # Add highlighted trajectories
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, coach in enumerate(highlight_coaches):
            if coach in plot_df['Coach'].values:
                coach_data = plot_df[plot_df['Coach'] == coach]
                color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter(
                    x=coach_data['Season_Number'],
                    y=coach_data['Cumulative_Games'],
                    mode='lines+markers',
                    name=f'{coach} ({coach_data["Total_Seasons"].iloc[0]} seasons)',
                    line=dict(color=color, width=3),
                    marker=dict(size=6),
                    hovertemplate=(
                        f'<b>{coach}</b><br>' +
                        'Season: %{x}<br>' +
                        'Cumulative WAR: %{y:.1f} games<br>' +
                        'Year: %{customdata[0]}<br>' +
                        'Annual WAR: %{customdata[4]:.1f} games<br>' +
                        'Win%: %{customdata[2]:.3f}<br>' +
                        '<extra></extra>'
                    ),
                    customdata=coach_data[['Year', 'Annual_WAR', 'Win_Pct', 'Cumulative_Games', 'Annual_Games']].values
                ))
                
                # Print coach statistics
                final_war = coach_data['Cumulative_WAR'].iloc[-1]
                final_games = coach_data['Cumulative_Games'].iloc[-1]
                seasons = coach_data['Season_Number'].iloc[-1]
                avg_war = final_war / seasons
                avg_games = final_games / seasons
                
                print(f"\n{coach} Career Trajectory:")
                print(f"  Total seasons: {seasons}")
                print(f"  Final cumulative WAR: {final_war:.3f} ({final_games:.1f} games)")
                print(f"  Average WAR per season: {avg_war:.3f} ({avg_games:.1f} games)")
                print(f"  Career win percentage: {coach_data['Win_Pct'].mean():.3f}")
            else:
                print(f"Warning: Coach '{coach}' not found in data")
        
        title = f"Coach WAR Career Trajectories (Highlighted: {', '.join(highlight_coaches)})"
        
    else:
        # Plot all coaches with different opacity based on career length
        fig = px.line(
            plot_df,
            x='Season_Number',
            y='Cumulative_Games',
            color='Coach',
            title='Coach WAR Career Trajectories',
            hover_data={
                'Year': True,
                'Annual_WAR': ':.3f',
                'Annual_Games': ':.1f',
                'Cumulative_WAR': ':.3f',
                'Win_Pct': ':.3f',
                'Coach': False
            }
        )
        
        # Update traces to reduce visual clutter
        fig.update_traces(
            line=dict(width=2),
            opacity=0.7,
            showlegend=False  # Too many coaches to show legend
        )
        
        title = f"Coach WAR Career Trajectories ({len(eligible_coaches)} coaches, {min_seasons}+ seasons)"
    
    # Add reference line at 0 WAR
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5,
                  annotation_text="Replacement Level (0 Games)")
    
    # Update layout
    fig.update_layout(
        title=title,
        title_font_size=18,
        xaxis_title='Season Number',
        yaxis_title='Cumulative WAR (Games Above Replacement)',
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        legend_title_font_size=16,
        width=1200,
        height=800,
        plot_bgcolor='white',
        font_family="Cambria",
        legend=dict(
            font_size=14
        )
    )
    
    # Update grid styling
    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')
    
    return fig

def show_trajectory_summary(trajectory_df):
    """Show summary statistics about career trajectories."""
    print(f"\n{'='*80}")
    print("TRAJECTORY SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    # Calculate final career stats for each coach
    final_stats = trajectory_df.groupby('Coach').agg({
        'Cumulative_WAR': 'last',
        'Season_Number': 'max',
        'Win_Pct': 'mean'
    }).round(3)
    
    final_stats['Avg_WAR_Per_Season'] = (final_stats['Cumulative_WAR'] / 
                                        final_stats['Season_Number']).round(3)
    final_stats['Final_Games'] = (final_stats['Cumulative_WAR'] * 16).round(1)
    final_stats['Avg_Games_Per_Season'] = (final_stats['Avg_WAR_Per_Season'] * 16).round(1)
    
    print(f"\nTotal coaches analyzed: {len(final_stats)}")
    print(f"Average career length: {final_stats['Season_Number'].mean():.1f} seasons")
    print(f"Average final cumulative WAR: {final_stats['Cumulative_WAR'].mean():.3f} ({final_stats['Final_Games'].mean():.1f} games)")
    
    print(f"\nTop 10 Career Trajectories (by final cumulative WAR):")
    print(f"{'Coach':<25} {'Seasons':<8} {'Final WAR':<15} {'Avg WAR':<15} {'Win%'}")
    print("-" * 85)
    
    top_careers = final_stats.nlargest(10, 'Cumulative_WAR')
    for coach, row in top_careers.iterrows():
        coach_name = coach[:23] if len(coach) > 23 else coach
        print(f"{coach_name:<25} {int(row['Season_Number']):<8} {row['Cumulative_WAR']:+.3f} ({row['Final_Games']:+.1f}g)  "
              f"{row['Avg_WAR_Per_Season']:+.3f} ({row['Avg_Games_Per_Season']:+.1f}g)  {row['Win_Pct']:.3f}")
    
    print(f"\nWorst 10 Career Trajectories (by final cumulative WAR):")
    print(f"{'Coach':<25} {'Seasons':<8} {'Final WAR':<15} {'Avg WAR':<15} {'Win%'}")
    print("-" * 85)
    
    worst_careers = final_stats.nsmallest(10, 'Cumulative_WAR')
    for coach, row in worst_careers.iterrows():
        coach_name = coach[:23] if len(coach) > 23 else coach
        print(f"{coach_name:<25} {int(row['Season_Number']):<8} {row['Cumulative_WAR']:+.3f} ({row['Final_Games']:+.1f}g)  "
              f"{row['Avg_WAR_Per_Season']:+.3f} ({row['Avg_Games_Per_Season']:+.1f}g)  {row['Win_Pct']:.3f}")

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Coach WAR Career Trajectory Visualization')
    parser.add_argument('--coaches', nargs='+', 
                       help='Names of specific coaches to highlight (space-separated)')
    parser.add_argument('--min-seasons', type=int, default=1,
                       help='Minimum number of seasons to include coach (default: 1)')
    args = parser.parse_args()
    
    print("="*80)
    print("COACH WAR CAREER TRAJECTORY ANALYSIS")
    print("Shows cumulative WAR building over each coach's career")
    print("WAR = Actual Win% - Replacement-Level Prediction")
    print("="*80)
    
    # Load and process data
    df = load_coaching_data()
    trajectory_df = calculate_coach_trajectories(df)
    
    # Create visualization
    fig = create_trajectory_plot(trajectory_df, args.coaches, args.min_seasons)
    
    # Show summary statistics
    show_trajectory_summary(trajectory_df)
    
    # Save the plot
    if args.coaches:
        coach_suffix = "_".join([c.replace(" ", "_") for c in args.coaches])
        output_file = f'analysis/coach_war_trajectory_{coach_suffix}.html'
    else:
        output_file = 'analysis/coach_war_trajectory.html'
    
    fig.write_html(output_file)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Interactive plot saved to: {output_file}")
    print("Open this file in your web browser to explore coach career trajectories.")
    
    if args.coaches:
        print(f"Highlighted coaches: {', '.join(args.coaches)}")
    
    # Save trajectory data
    trajectory_file = 'data/final/coach_war_trajectories.csv'
    trajectory_df.to_csv(trajectory_file, index=False)
    print(f"Coach trajectory data saved to: {trajectory_file}")

if __name__ == "__main__":
    main()