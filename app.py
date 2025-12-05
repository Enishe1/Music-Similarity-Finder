import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
from music_engine import MusicSimilarityFinder

# Initialize
@st.cache_resource
def load_finder():
    finder = MusicSimilarityFinder("data/audio")
    st.session_state.features_df = finder.features_df
    return finder

finder = load_finder()

# Debug test
st.sidebar.markdown("##Debug Info")
st.sidebar.write(f"Songs loaded: {len(finder.song_names)}")
if hasattr(finder, 'features_df'):
    st.sidebar.write(f"Features shape: {finder.features_df.shape}")

# --- UI Setup
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
            if isinstance(val, (np.ndarray, list)):
                if len(val) > 0:
                    display_val = float(val[0]) if hasattr(val[0], '__float__') else str(val[0])
                else:
                    display_val = "N/A"
            else:
                display_val = val
            
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
    
    audio_path = f"data/audio/{selected}.mp3"
    try:
        st.audio(audio_path)
    except:
        st.warning(f"Audio file not found: {audio_path}")
    
    features = finder.get_features(selected)
    if features:
        clean_features = {}
        for key, val in features.items():
            if isinstance(val, (np.ndarray, list)):
                clean_features[key] = float(val[0]) if len(val) > 0 else 0.0
            else:
                clean_features[key] = float(val) if isinstance(val, (int, float, np.number)) else 0.0
        
        # Get mapped features
        mapped_features = {}
        feature_mapping = {
            'tempo': ['tempo', 'bpm', 'beat', 'tempo_bpm'],
            'energy': ['energy', 'rms', 'loudness', 'rms_energy'],
            'danceability': ['danceability', 'dance', 'groove'],
            'brightness': ['brightness', 'spectral_centroid', 'high_freq', 'spectral_rolloff'],
            'complexity': ['complexity', 'zcr', 'zero_crossing_rate', 'spectral_complexity']
        }
        
        for target_feature, possible_names in feature_mapping.items():
            found = False
            for name in possible_names:
                if name in clean_features:
                    mapped_features[target_feature] = clean_features[name]
                    found = True
                    break
            if not found:
                mapped_features[target_feature] = 0.5
        
        # Create Musical DNA display
        st.markdown("### Musical DNA")
        
        def map_to_display(value, value_type):
            if value_type == "tempo":
                if value > 140: return "on"
                elif value > 120: return "fast"
                elif value > 100: return "medium"
                elif value > 80: return "slow"
                else: return "off"
            else:
                if value > 0.7: return "high"
                elif value > 0.4: return "medium"
                else: return "low"
        
        # Create normalized values for display
        normalized = {}
        for feat, val in mapped_features.items():
            if feat == 'tempo':
                normalized[feat] = max(0, min(1, (val - 50) / 150))  # 50-200 BPM
            elif feat == 'energy':
                normalized[feat] = max(0, min(1, val / 5000))  # 0-5000
            elif feat == 'danceability':
                normalized[feat] = max(0, min(1, (val + 1) / 2))  # -1 to 1
            elif feat == 'brightness':
                normalized[feat] = max(0, min(1, val / 100))  # 0-100
            elif feat == 'complexity':
                normalized[feat] = max(0, min(1, val / 20))  # 0-20
            else:
                normalized[feat] = 0.5
        
        dna_html = f"""
        <div style="font-family: monospace; background: #f5f5f5; padding: 15px; border-radius: 8px;">
            <div style="display: flex; justify-content: space-between;">
                <span>tempo</span>
                <span style="font-weight: bold;">{map_to_display(mapped_features.get('tempo', 100), "tempo")}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>complexity</span>
                <span style="font-weight: bold;">{map_to_display(normalized.get('complexity', 0.5), "complexity")}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>brightness</span>
                <span style="font-weight: bold;">{map_to_display(normalized.get('brightness', 0.5), "brightness")}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>danceability</span>
                <span style="font-weight: bold;">{map_to_display(normalized.get('danceability', 0.5), "danceability")}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>energy</span>
                <span style="font-weight: bold;">{map_to_display(normalized.get('energy', 0.5), "energy")}</span>
            </div>
            <hr>
            <div style="text-align: center; font-size: 1.2em;">
                {selected.replace('.mp3', '').replace('_', ' ')}
            </div>
        </div>
        """
        st.markdown(dna_html, unsafe_allow_html=True)
        
        # Create radar chart
        categories = ['tempo', 'energy', 'danceability', 'brightness', 'complexity']
        radar_values = []
        
        for cat in categories:
            val = mapped_features.get(cat, 0.5)
            # Normalize to 0-1 range
            if cat == 'tempo':
                norm_val = max(0, min(1, (val - 50) / 150))  # 50-200 BPM
            elif cat == 'energy':
                norm_val = max(0, min(1, val / 5000))  # 0-5000
            elif cat == 'danceability':
                norm_val = max(0, min(1, (val + 1) / 2))  # -1 to 1
            elif cat == 'brightness':
                norm_val = max(0, min(1, val / 100))  # 0-100
            elif cat == 'complexity':
                norm_val = max(0, min(1, val / 20))  # 0-20
            else:
                norm_val = 0.5
            radar_values.append(norm_val)
        
        # Create the radar chart
        fig = go.Figure(data=go.Scatterpolar(
            r=radar_values,
            theta=categories,
            fill='toself',
            line=dict(color='rgb(31, 119, 180)', width=2),
            marker=dict(size=4),
            name=selected
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.5, 1],
                    ticktext=['Low', 'Medium', 'High']
                ),
                angularaxis=dict(direction="clockwise")
            ),
            showlegend=False,
            title=dict(text=f"Musical DNA: {selected}", font=dict(size=16)),
            margin=dict(l=80, r=80, t=80, b=80)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No features available for this song")

with col2:
    st.header("Similar Songs")
    
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
                
                song_path = f"data/audio/{song}.mp3"
                if cols[2].button("Play", key=f"play_{song}"):
                    st.audio(song_path)
                
                if finder.check_mashup_compatibility(selected, song)['overall_compatible']:
                    st.caption("DJ Compatible")
                
                st.divider()

# Footer
st.sidebar.markdown("---")
st.sidebar.info(f"Total songs: {len(finder.song_names)}")

if st.sidebar.button("Reload Features"):
    st.cache_resource.clear()
    st.rerun()