# Music Discovery Radar

Music Discovery Radar is a Streamlit-powered music similarity app that analyzes audio features and visualizes a song’s Musical DNA using radar charts.

Built on a dataset of 50 AI-generated songs, this app extracts audio features from MP3 files, computes similarity between tracks, and lets users interactively explore music based on energy, danceability and tempo preferences.

---

## Features

- Audio Feature Extraction using librosa
  - Tempo (BPM)
  - Spectral centroid (energy/brightness)
  - Zero-crossing rate (danceability proxy)
  - RMS energy
  - MFCC mean & standard deviation features

- Song Similarity Engine
  - Standardized feature vectors
  - Euclidean distance-based similarity
  - Adjustable preference weighting (energy, danceability, tempo)

- Musical DNA Visualization
  - Radar chart (tempo, energy, danceability, brightness, complexity)
  - Human-readable descriptors (Low / Medium / High, Fast / Slow)

- Interactive Streamlit UI
  - Song picker + random song button
  - Live audio playback
  - Adjustable sliders for personalized recommendations

---

## How It Works

### 1. Feature Extraction
Each song is loaded (first 90 seconds) and analyzed with librosa to extract rhythmic, spectral, and timbral features.

### 2. Feature Normalization
All numeric features are standardized using StandardScaler to ensure fair distance comparisons.

### 3. Similarity Computation
Similarity between songs is calculated as:
```bash
similarity = 1 / (1 + euclidean_distance)
```

This returns values between 0 and 1, where higher means more similar.

### 4. Preference Adjustments
Users can bias recommendations using sliders:

- Energy – spectral centroid
- Danceability – zero-crossing rate
- Tempo – BPM

These adjustments gently nudge similarity scores in real time.

---

## Running the App

### Clone the repository:
```bash
git clone https://github.com/Enishe1/Music-Similarity-Finder.git
cd Music-Similarity-Finder
```

### Install dependencies

```bash
pip install -r requirements.txt
```


### Start Streamlit

```bash
streamlit run app.py
```

Then open the provided local URL in your browser.
