"""
Interactive Scatter Plot: Coach Impact vs Seasons
Shows the relationship between average coaching impact and number of seasons coached.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_coach_impact_scatter():
    """Create interactive scatter plot of coach impact vs seasons."""
    
    # Load the data
    print("Loading coach career impact data...")
    df = pd.read_csv('data/final/coach_career_impact_stats.csv')
    
    # Clean up coach names (remove any extra whitespace)
    df['Primary_Coach'] = df['Primary_Coach'].str.strip()
    
    # Create hover text with detailed information
    df['hover_text'] = (
        df['Primary_Coach'] + '<br>' +
        'Avg Impact: ' + df['Avg_Impact'].apply(lambda x: f"{x:+.4f}") + '<br>' +
        'Seasons: ' + df['Seasons'].astype(str) + '<br>' +
        'Total Impact: ' + df['Total_Impact'].apply(lambda x: f"{x:+.4f}") + '<br>' +
        'Avg Win%: ' + df['Avg_Actual_Win'].apply(lambda x: f"{x:.3f}")
    )
    
    # Create the scatter plot
    fig = px.scatter(
        df,
        x='Avg_Impact',
        y='Seasons',
        hover_name='Primary_Coach',
        hover_data={
            'Avg_Impact': ':.4f',
            'Seasons': True,
            'Total_Impact': ':.4f',
            'Avg_Actual_Win': ':.3f'
        },
        title='Coach Career Impact vs Number of Seasons',
        labels={
            'Avg_Impact': 'Average Coaching Impact (vs Replacement)',
            'Seasons': 'Number of Seasons Coached'
        },
        width=1000,
        height=700
    )
    
    # Customize the plot
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.7,
            line=dict(width=1, color='white')
        )
    )
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        showlegend=False,
        plot_bgcolor='white'
    )
    
    # Update grid styling
    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')
    
    # Add annotations for quadrants
    fig.add_annotation(
        x=0.02, y=25,
        text="High Impact<br>Long Career",
        showarrow=False,
        font=dict(size=10, color="green"),
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    fig.add_annotation(
        x=-0.02, y=25,
        text="Low Impact<br>Long Career",
        showarrow=False,
        font=dict(size=10, color="red"),
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    fig.add_annotation(
        x=0.02, y=5,
        text="High Impact<br>Short Career",
        showarrow=False,
        font=dict(size=10, color="orange"),
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    fig.add_annotation(
        x=-0.02, y=5,
        text="Low Impact<br>Short Career",
        showarrow=False,
        font=dict(size=10, color="gray"),
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    # Show some key statistics
    print(f"\nDataset Summary:")
    print(f"Total coaches: {len(df)}")
    print(f"Average impact: {df['Avg_Impact'].mean():+.4f}")
    print(f"Average seasons: {df['Seasons'].mean():.1f}")
    print(f"\nTop 5 by Impact:")
    top_5 = df.nlargest(5, 'Avg_Impact')[['Primary_Coach', 'Avg_Impact', 'Seasons']]
    for _, row in top_5.iterrows():
        print(f"  {row['Primary_Coach']}: {row['Avg_Impact']:+.4f} ({row['Seasons']} seasons)")
    
    print(f"\nLongest careers (15+ seasons):")
    long_career = df[df['Seasons'] >= 15].sort_values('Seasons', ascending=False)
    for _, row in long_career.iterrows():
        print(f"  {row['Primary_Coach']}: {row['Seasons']} seasons ({row['Avg_Impact']:+.4f} impact)")
    
    # Save the plot as HTML file
    output_file = 'analysis/coach_impact_scatter.html'
    fig.write_html(output_file)
    print(f"\nInteractive plot saved to: {output_file}")
    print("Open this file in your web browser to interact with the plot.")
    
    return fig, df

if __name__ == "__main__":
    print("Creating interactive coach impact scatter plot...")
    fig, data = create_coach_impact_scatter()
    print("\nHover over points to see coach details!")
    print("The plot should open in your default web browser.")