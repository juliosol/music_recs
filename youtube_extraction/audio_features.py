"""
Audio feature extraction from YouTube videos using librosa
Extracts Spotify-like features from audio files
"""
import os
import pickle
import logging
import tempfile
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import yt_dlp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Extract audio features similar to Spotify's audio features"""

    def __init__(self, cache_dir='cache/audio_features'):
        """
        Initialize feature extractor
        Args:
            cache_dir: Directory to cache extracted features
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def extract_audio_features(self, audio_path):
        """
        Extract Spotify-like features from audio file
        Args:
            audio_path: Path to audio file
        Returns:
            Dict with audio features
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, duration=120)  # Load up to 2 minutes

            # Extract features
            features = {
                'danceability': self._compute_danceability(y, sr),
                'energy': self._compute_energy(y),
                'key': self._compute_key(y, sr),
                'loudness': self._compute_loudness(y),
                'mode': self._compute_mode(y, sr),
                'speechiness': self._compute_speechiness(y, sr),
                'acousticness': self._compute_acousticness(y, sr),
                'instrumentalness': self._compute_instrumentalness(y, sr),
                'liveness': self._compute_liveness(y, sr),
                'valence': self._compute_valence(y, sr),
                'tempo': self._compute_tempo(y, sr),
                'duration_ms': int(len(y) / sr * 1000)
            }

            return features

        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            return None

    def download_and_analyze(self, video_id, use_cache=True):
        """
        Download audio from YouTube video and analyze
        Args:
            video_id: YouTube video ID
            use_cache: Use cached features if available
        Returns:
            Feature dict or None on error
        """
        # Check cache first
        if use_cache:
            cached = self.get_cached_features(video_id)
            if cached is not None:
                logger.info(f"Using cached features for {video_id}")
                return cached

        # Download audio
        audio_path = self._download_audio(video_id)
        if not audio_path:
            return None

        try:
            # Extract features
            features = self.extract_audio_features(audio_path)

            # Cache features
            if features and use_cache:
                self.save_cached_features(video_id, features)

            return features

        finally:
            # Clean up downloaded file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass

    def _download_audio(self, video_id):
        """
        Download audio from YouTube video
        Args:
            video_id: YouTube video ID
        Returns:
            Path to downloaded audio file
        """
        try:
            # Create temp file
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"{video_id}.mp3")

            # yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(temp_dir, f"{video_id}.%(ext)s"),
                'quiet': True,
                'no_warnings': True,
            }

            # Download
            url = f"https://www.youtube.com/watch?v={video_id}"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            if os.path.exists(output_path):
                return output_path

            logger.error(f"Failed to download audio for {video_id}")
            return None

        except Exception as e:
            logger.error(f"Error downloading {video_id}: {e}")
            return None

    def get_cached_features(self, video_id):
        """
        Load cached features from disk
        Args:
            video_id: YouTube video ID
        Returns:
            Feature dict or None if not cached
        """
        cache_file = os.path.join(self.cache_dir, f"{video_id}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading cache for {video_id}: {e}")
        return None

    def save_cached_features(self, video_id, features):
        """
        Save features to disk cache
        Args:
            video_id: YouTube video ID
            features: Feature dict
        """
        cache_file = os.path.join(self.cache_dir, f"{video_id}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
        except Exception as e:
            logger.error(f"Error saving cache for {video_id}: {e}")

    # Feature computation methods
    def _compute_danceability(self, y, sr):
        """
        Compute danceability (0-1)
        Based on tempo stability and rhythm strength
        """
        try:
            # Get tempo and beat strength
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

            # Compute onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # Rhythm regularity (lower std = more regular = more danceable)
            beat_frames = librosa.frames_to_samples(beats)
            if len(beat_frames) > 1:
                beat_intervals = np.diff(beat_frames)
                rhythm_regularity = 1 - min(np.std(beat_intervals) / np.mean(beat_intervals), 1)
            else:
                rhythm_regularity = 0.5

            # Combine tempo factor and rhythm regularity
            # Ideal dance tempo: 90-130 BPM
            tempo_factor = 1 - abs(tempo - 110) / 110
            tempo_factor = max(0, min(1, tempo_factor))

            danceability = 0.6 * rhythm_regularity + 0.4 * tempo_factor
            return float(np.clip(danceability, 0, 1))

        except:
            return 0.5

    def _compute_energy(self, y):
        """
        Compute energy (0-1)
        Based on RMS and spectral energy
        """
        try:
            rms = librosa.feature.rms(y=y)[0]
            energy = np.mean(rms)
            # Normalize to 0-1 range
            return float(np.clip(energy * 10, 0, 1))
        except:
            return 0.5

    def _compute_key(self, y, sr):
        """
        Compute musical key (0-11)
        0=C, 1=C#, 2=D, etc.
        """
        try:
            chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
            key = np.argmax(np.sum(chromagram, axis=1))
            return int(key)
        except:
            return 0

    def _compute_loudness(self, y):
        """
        Compute loudness in dB
        """
        try:
            rms = librosa.feature.rms(y=y)[0]
            loudness = librosa.amplitude_to_db(rms)
            return float(np.mean(loudness))
        except:
            return -10.0

    def _compute_mode(self, y, sr):
        """
        Compute mode (0=minor, 1=major)
        """
        try:
            chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)

            # Major and minor templates
            major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

            # Average chromagram
            chroma_mean = np.mean(chromagram, axis=1)

            # Correlation with templates
            major_corr = np.corrcoef(chroma_mean, major_template)[0, 1]
            minor_corr = np.corrcoef(chroma_mean, minor_template)[0, 1]

            return 1 if major_corr > minor_corr else 0
        except:
            return 1

    def _compute_speechiness(self, y, sr):
        """
        Compute speechiness (0-1)
        Based on spectral characteristics of speech
        """
        try:
            # Zero crossing rate (high for speech)
            zcr = librosa.feature.zero_crossing_rate(y)[0]

            # Spectral centroid
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

            # Speech has characteristic ZCR and centroid patterns
            speechiness = (np.mean(zcr) * 2 + np.std(cent) / 1000) / 2
            return float(np.clip(speechiness, 0, 1))
        except:
            return 0.1

    def _compute_acousticness(self, y, sr):
        """
        Compute acousticness (0-1)
        Higher for acoustic instruments
        """
        try:
            # Spectral rolloff (lower for acoustic)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

            # Spectral centroid (lower for acoustic)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

            # Lower values = more acoustic
            acousticness = 1 - (np.mean(centroid) / sr * 2)
            return float(np.clip(acousticness, 0, 1))
        except:
            return 0.5

    def _compute_instrumentalness(self, y, sr):
        """
        Compute instrumentalness (0-1)
        Predicts if track contains no vocals
        """
        try:
            # Detect harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)

            # Spectral contrast (higher = more instrumental)
            contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)

            # MFCC variance (lower = more instrumental)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = np.var(mfcc, axis=1)

            instrumentalness = (np.mean(contrast) / 40 + (1 - np.mean(mfcc_var) / 100)) / 2
            return float(np.clip(instrumentalness, 0, 1))
        except:
            return 0.5

    def _compute_liveness(self, y, sr):
        """
        Compute liveness (0-1)
        Detects presence of audience
        """
        try:
            # Spectral flatness (higher = more live/noise)
            flatness = librosa.feature.spectral_flatness(y=y)[0]

            # RMS variance (more variance = more live)
            rms = librosa.feature.rms(y=y)[0]
            rms_var = np.var(rms)

            liveness = (np.mean(flatness) * 2 + rms_var * 10) / 2
            return float(np.clip(liveness, 0, 1))
        except:
            return 0.2

    def _compute_valence(self, y, sr):
        """
        Compute valence (0-1)
        Musical positiveness
        """
        try:
            # Tempo (faster = happier)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_factor = min(tempo / 150, 1)

            # Mode (major = happier)
            mode = self._compute_mode(y, sr)

            # Spectral centroid (brighter = happier)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            brightness = min(np.mean(centroid) / sr, 1)

            valence = (tempo_factor * 0.3 + mode * 0.4 + brightness * 0.3)
            return float(np.clip(valence, 0, 1))
        except:
            return 0.5

    def _compute_tempo(self, y, sr):
        """
        Compute tempo in BPM
        """
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return float(tempo)
        except:
            return 120.0
