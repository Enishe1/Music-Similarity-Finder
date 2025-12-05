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
                'tempo': librosa.beat.beat_track(y=y, sr=sr)[0],
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
                'rmse': np.mean(librosa.feature.rms(y=y)),
            }
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
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
        
        return pd.DataFrame(features_list)
    
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
        if adjustments and len(adjustments) == 2:
            energy_adj, dance_adj = adjustments
            # Simple adjustment: modify similarity scores
         
            if energy_adj != 0:
                energy_feat = self.features_df['spectral_centroid'].values
                energy_norm = (energy_feat - np.min(energy_feat)) / (np.max(energy_feat) - np.min(energy_feat))
                similarities += energy_adj * energy_norm * 0.2
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_indices = [i for i in sorted_indices if i != idx][:n]
        
        return {self.song_names[i]: similarities[i] for i in sorted_indices}
    
    def get_features(self, song_name):
        """Get feature vector for radar chart"""
        if song_name in self.song_names:
            idx = self.song_names.index(song_name)
            return {
                'tempo': self.features_df.iloc[idx]['tempo'],
                'energy': self.features_df.iloc[idx]['spectral_centroid'],
                'danceability': 1 - self.features_df.iloc[idx]['zero_crossing_rate'] / 0.1,
                'brightness': self.features_df.iloc[idx]['mfcc_1_mean'],
                'complexity': self.features_df.iloc[idx]['mfcc_12_std']
            }
        return {}
    