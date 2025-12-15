import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

class MusicSimilarityFinder:
    def __init__(self, audio_dir="data/audio"):
        self.audio_dir = Path(audio_dir)
        self.song_files = list(self.audio_dir.glob("*.mp3"))
        self.song_names = [f.stem for f in self.song_files]
        self.metadata = pd.read_csv("data/metadata.csv")
        
        print(f"Loading {len(self.song_files)} songs...")
        self.features_df = self._extract_all_features()
        self.similarity_matrix = self._build_similarity_matrix()
        print("Music engine ready!")
    
    def _extract_features(self, audio_path):
        """Extract audio features from MP3 file"""
        try:
            y, sr = librosa.load(audio_path, duration=30)
            
            features = {
                'tempo': float(librosa.beat.beat_track(y=y, sr=sr)[0]),  # Ensure float
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'rmse': float(np.mean(librosa.feature.rms(y=y))),
            }
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
            return features
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def _extract_all_features(self):
        """Extract features for all songs"""
        features_list = []
        
        for file in self.song_files:
            print(f"Processing: {file.stem}")
            features = self._extract_features(file)
            if features:
                features['filename'] = file.name
                features['song_name'] = file.stem
                features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        # Ensure all numeric columns are actually numeric
        numeric_cols = [c for c in df.columns if c not in ['filename', 'song_name']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill any NaN values with column mean
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def _build_similarity_matrix(self):
        """Build similarity matrix from features"""
        numeric_cols = [c for c in self.features_df.columns 
                       if c not in ['filename', 'song_name']]
        X = self.features_df[numeric_cols].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        distances = euclidean_distances(X_scaled)
        similarities = 1 / (1 + distances)  # Convert to similarity (0-1)
        
        return similarities
    
    def recommend(self, song_name, n=5, adjustments=None):
        """Get top N similar songs"""
        if song_name not in self.song_names:
            return {}
        
        idx = self.song_names.index(song_name)
        similarities = self.similarity_matrix[idx].copy()
        
        # Apply adjustments if provided
        if adjustments:
            if len(adjustments) == 2:
                energy_adj, dance_adj = adjustments
                tempo_adj = 0
            elif len(adjustments) == 3:
                energy_adj, dance_adj, tempo_adj = adjustments
            else:
                energy_adj = dance_adj = tempo_adj = 0
            
            # Convert features to numpy arrays of floats
            energy_feat = self.features_df['spectral_centroid'].to_numpy(dtype=float)
            dance_feat = self.features_df['zero_crossing_rate'].to_numpy(dtype=float)
            tempo_feat = self.features_df['tempo'].to_numpy(dtype=float)
            
            # Energy adjustment
            if energy_adj != 0:
                energy_min = np.min(energy_feat)
                energy_max = np.max(energy_feat)
                if energy_max > energy_min:
                    energy_norm = (energy_feat - energy_min) / (energy_max - energy_min)
                    similarities += energy_adj * energy_norm * 0.1
            
            # Danceability adjustment 
            if dance_adj != 0:
                dance_min = np.min(dance_feat)
                dance_max = np.max(dance_feat)
                if dance_max > dance_min:
                    dance_norm = (dance_feat - dance_min) / (dance_max - dance_min)
                    similarities += dance_adj * (1 - dance_norm) * 0.1
            
            # Tempo adjustment 
            if tempo_adj != 0:
                tempo_min = np.min(tempo_feat)
                tempo_max = np.max(tempo_feat)
                if tempo_max > tempo_min:
                    tempo_norm = (tempo_feat - tempo_min) / (tempo_max - tempo_min)
                    similarities += tempo_adj * tempo_norm * 0.1
        
        # Sort by similarity (exclude current song)
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_indices = [i for i in sorted_indices if i != idx][:n]
        
        return {self.song_names[i]: float(similarities[i]) for i in sorted_indices}
    
    def get_features(self, song_name):
        """Get feature vector for radar chart"""
        if song_name in self.song_names:
            idx = self.song_names.index(song_name)
            return {
                'tempo': float(self.features_df.iloc[idx]['tempo']),
                'energy': float(self.features_df.iloc[idx]['spectral_centroid']),
                'danceability': float(1 - self.features_df.iloc[idx]['zero_crossing_rate'] / 0.1),
                'brightness': float(self.features_df.iloc[idx]['mfcc_1_mean']),
                'complexity': float(self.features_df.iloc[idx]['mfcc_12_std'])
            }
        return {}