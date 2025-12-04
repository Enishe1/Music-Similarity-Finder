import plotly.graph_objects as go
import plotly.express as px
import numpy as np

class RadarVisualizer:
    def __init__(self, features_df):
        """Initialize with ALL songs' features for global normalization"""
        self.features_df = features_df
        self.feature_columns = ['tempo', 'energy', 'danceability', 'brightness', 'complexity']
        
        # Store global min/max for each feature
        self.global_min = {}
        self.global_max = {}
        
        for col in self.feature_columns:
            if col in features_df.columns:
                self.global_min[col] = features_df[col].min()
                self.global_max[col] = features_df[col].max()
    
    def create_radar_chart(self, song_features, song_name="", genre=""):
        """Create radar chart with GLOBAL normalization"""
        if not song_features:
            return go.Figure()
        
        # Extract features in consistent order
        categories = ['tempo', 'energy', 'danceability', 'brightness', 'complexity']
        values = []
        
        for cat in categories:
            if cat in song_features:
                # Normalize against GLOBAL min/max
                if cat in self.global_min and cat in self.global_max:
                    global_range = self.global_max[cat] - self.global_min[cat]
                    if global_range > 0:
                        normalized = (song_features[cat] - self.global_min[cat]) / global_range
                    else:
                        normalized = 0.5
                else:
                    # Fallback: normalize within this song's features
                    normalized = 0.5
                values.append(normalized)
            else:
                values.append(0.5)
        
        # Genre-based coloring
        genre_colors = {
            "Pop": "rgb(255, 75, 75)",
            "Hip Hop": "rgb(75, 75, 255)", 
            "Rock": "rgb(255, 150, 50)",
            "Country": "rgb(50, 200, 100)",
            "Jazz": "rgb(180, 75, 255)"
        }
        
        color = genre_colors.get(genre, "rgb(31, 119, 180)")
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color=color, width=2),
            marker=dict(size=4),
            name=f"{song_name} ({genre})" if genre else song_name
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.5, 1],
                    ticktext=['Low', 'Medium', 'High']
                ),
                angularaxis=dict(
                    direction="clockwise"
                )
            ),
            showlegend=True,
            title=dict(
                text=f"Musical DNA: {song_name}",
                font=dict(size=16)
            ),
            margin=dict(l=80, r=80, t=80, b=80)
        )
        
        return fig

def plot_similarity_matrix(similarity_matrix, song_names, genres=None):
    """Create heatmap with genre annotations"""
    fig = px.imshow(
        similarity_matrix,
        x=song_names,
        y=song_names,
        color_continuous_scale='Viridis',
        title="Song Similarity Matrix",
        labels=dict(color="Similarity")
    )
    
    # Add genre annotations if available
    if genres and len(genres) == len(song_names):
        # Create color mapping for genres
        genre_colors = {
            "Pop": "#FF4B4B",
            "Hip Hop": "#4B4BFF", 
            "Rock": "#FF9632",
            "Country": "#32C864",
            "Jazz": "#B44BFF"
        }
        
        # Add genre indicators
        annotations = []
        for i, genre in enumerate(genres):
            if genre in genre_colors:
                annotations.append(dict(
                    x=i,
                    y=-0.5,
                    text=genre,
                    showarrow=False,
                    font=dict(color=genre_colors[genre], size=10),
                    xref="x",
                    yref="y"
                ))
        
        fig.update_layout(annotations=annotations)
    
    fig.update_layout(
        width=1000, 
        height=1000,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )
    
    return fig