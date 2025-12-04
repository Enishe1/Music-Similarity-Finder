import streamlit as st
import pandas as pd
import numpy as np
import random
from music_engine import MusicSimilarityFinder
from visualize import RadarVisualizer

# Initialize
@st.cache_resource
def load_finder():
    finder = MusicSimilarityFinder("data/audio")
    st.session_state.features_df = finder.features_df
    return finder

finder = load_finder()
visualizer = RadarVisualizer(finder.features_df)

# Debug test
st.sidebar.markdown("##Debug Info")
st.sidebar.write(f"Songs loaded: {len(finder.song_names)}")
if hasattr(finder, 'features_df'):
    st.sidebar.write(f"Features shape: {finder.features_df.shape}")

# ---      UI Setup
st.set_page_config(layout="wide", page_title="Music Discovery Radar")
st.title("Music Discovery Radar")

# Sidebar
st.sidebar.header("Controls")
if st.sidebar.button("Random Song", use_container_width=True):
    random_song = random.choice(finder.song_names)
    st.session_state.selected_song = random_song
    st.rerun()

selected = st.sidebar.selectbox(
    "Pick a song:",
    finder.song_names,
    key="selected_song",
    index=0
)

# Show features with numpy array handling
if selected:
    features = finder.get_features(selected)
    st.sidebar.markdown(f"### Features for '{selected}'")
    if features:
        for key, val in features.items():
            # Handle numpy arrays
            if isinstance(val, (np.ndarray, list)):
                if len(val) > 0:
                    display_val = float(val[0]) if hasattr(val[0], '__float__') else str(val[0])
                else:
                    display_val = "N/A"
            else:
                display_val = val
            
            # Format numbers
            try:
                if isinstance(display_val, (int, float, np.number)):
                    st.sidebar.write(f"{key}: {float(display_val):.4f}")
                else:
                    st.sidebar.write(f"{key}: {display_val}")
            except:
                st.sidebar.write(f"{key}: {str(display_val)}")

# Sliders
st.sidebar.header("Adjust Preferences")
energy = st.sidebar.slider("Energy", -1.0, 1.0, 0.0, 0.1)
dance = st.sidebar.slider("Danceability", -1.0, 1.0, 0.0, 0.1)

# Main columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header(f"**{selected}**")
    
    # Play audio
    audio_path = f"data/audio/{selected}.mp3"
    try:
        st.audio(audio_path)
    except:
        st.warning(f"Audio file not found: {audio_path}")
    
    # Radar chart
    features = finder.get_features(selected)
    if features:
        # Clean features for display
        clean_features = {}
        for key, val in features.items():
            if isinstance(val, (np.ndarray, list)):
                clean_features[key] = float(val[0]) if len(val) > 0 else 0.0
            else:
                clean_features[key] = float(val) if isinstance(val, (int, float, np.number)) else 0.0
        
        st.write("**Extracted Features:**", clean_features)
        fig = visualizer.create_radar_chart(clean_features, selected)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No features available for this song")

with col2:
    st.header("Similar Songs")
    
    # Get recommendations
    similar = finder.recommend(selected, adjustments=[energy, dance])
    
    if not similar:
        st.info("No similar songs found")
    else:
        st.success(f"Found {len(similar)} similar songs")
        for i, (song, score) in enumerate(similar.items(), 1):
            with st.container():
                cols = st.columns([4, 1, 1])
                cols[0].write(f"**{i}. {song}**")
                cols[1].write(f"`{float(score):.3f}`")
                
                # Play button
                song_path = f"data/audio/{song}.mp3"
                if cols[2].button("▶️", key=f"play_{song}"):
                    st.audio(song_path)
                
                # Mashup compatibility
                # (kinda obsolete function)
                if finder.check_mashup_compatibility(selected, song)['overall_compatible']:
                    st.caption("DJ Compatible")
                
                st.divider()

# Footer
st.sidebar.markdown("---")
st.sidebar.info(f"Total songs: {len(finder.song_names)}")
st.sidebar.info(f"App version: 1.0")

if st.sidebar.button("Reload Features", type="secondary"):
    st.cache_resource.clear()
    st.rerun()