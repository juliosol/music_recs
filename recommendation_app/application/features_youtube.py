"""
YouTube Feature Extraction for Flask Application
"""
import sys
import os
import pandas as pd
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from youtube_extraction.youtube_pipeline import YouTubeMusicPipeline
from youtube_extraction.feature_eng_youtube import playlist_preprocessing_youtube

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features_from_youtube_playlist(playlist_url, max_videos=20):
    """
    Extract features from YouTube playlist URL
    Args:
        playlist_url: Full YouTube playlist URL
        max_videos: Maximum number of videos to process
    Returns:
        DataFrame with normalized features matching database schema
    """
    try:
        logger.info(f"Extracting features from playlist: {playlist_url}")

        # Initialize pipeline
        pipeline = YouTubeMusicPipeline()

        # Extract tracks from playlist
        tracks = pipeline.extract_playlist_features(playlist_url, max_videos=max_videos, use_cache=True)

        if not tracks:
            logger.error("No tracks extracted from playlist")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(tracks)

        # Preprocess (without saving to file)
        logger.info("Preprocessing features...")

        # Just do the essential preprocessing without saving
        from youtube_extraction.feature_eng_youtube import (
            drop_duplicates_df_youtube,
            sentiment_analysis,
            datetime_converter,
            normalize_engagement_metrics,
            feature_normalizer
        )

        df = drop_duplicates_df_youtube(df)
        df = sentiment_analysis(df, 'song_info')
        df = datetime_converter(df)
        df = normalize_engagement_metrics(df)
        df = df.fillna(0)

        # Get normalized features
        normalized_df = feature_normalizer(df.copy())
        normalized_df = normalized_df.fillna(0)

        logger.info(f"Extracted {len(normalized_df)} tracks successfully")

        return normalized_df

    except Exception as e:
        logger.error(f"Error extracting playlist features: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_features_from_video_ids(video_ids):
    """
    Extract features from list of YouTube video IDs
    Args:
        video_ids: List of YouTube video IDs
    Returns:
        DataFrame with features
    """
    try:
        pipeline = YouTubeMusicPipeline()

        tracks = []
        for video_id in video_ids:
            video_details = pipeline.youtube_api.get_video_details(video_id)
            if not video_details:
                continue

            artist, track_name = pipeline._parse_title(video_details['title'])

            track = {
                'id': video_id,
                'track_name': track_name,
                'track_artist_name': artist,
                'youtube_url': f"https://youtube.com/watch?v={video_id}",
                'track_duration_ms': video_details['duration'],
                'views': video_details['views'],
                'likes': video_details['likes'],
                'track_album_release_date': video_details['published_at'],
                'track_album_type': 'single',
                'track_album_name': track_name,
                'artist_genres': [],
                'artist_popularity': pipeline._estimate_popularity(video_details['views']),
                'track_popularity': pipeline._estimate_popularity(video_details['views']),
                'track_explicit': False,
            }

            # Get audio features
            audio_features = pipeline.audio_extractor.download_and_analyze(video_id, use_cache=True)

            if audio_features:
                for key, value in audio_features.items():
                    if key != 'duration_ms':
                        track[f'track_{key}'] = value
            else:
                track.update(pipeline._get_default_audio_features())

            tracks.append(track)

        if not tracks:
            return None

        df = pd.DataFrame(tracks)

        # Preprocess
        from youtube_extraction.feature_eng_youtube import (
            drop_duplicates_df_youtube,
            sentiment_analysis,
            datetime_converter,
            normalize_engagement_metrics,
            feature_normalizer
        )

        df = drop_duplicates_df_youtube(df)
        df = sentiment_analysis(df, 'song_info')
        df = datetime_converter(df)
        df = normalize_engagement_metrics(df)
        df = df.fillna(0)

        normalized_df = feature_normalizer(df.copy())
        normalized_df = normalized_df.fillna(0)

        return normalized_df

    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None
