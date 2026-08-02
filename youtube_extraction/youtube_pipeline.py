"""
YouTube Data Pipeline
Combines YouTube API and audio analysis into complete feature extraction pipeline
"""
import logging
import time
import pandas as pd
from tqdm import tqdm
from .youtube_api import YouTubeAPI, load_api_key
from .audio_features import AudioFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeMusicPipeline:
    """Complete pipeline for extracting music features from YouTube"""

    def __init__(self, api_key=None, cache_dir='cache/audio_features'):
        """
        Initialize pipeline
        Args:
            api_key: YouTube API key (loads from credentials.json if None)
            cache_dir: Directory for caching audio features
        """
        if api_key is None:
            # Try multiple locations for credentials.json
            import os
            possible_paths = [
                'credentials.json',  # Current directory
                os.path.join(os.path.dirname(__file__), '..', 'credentials.json'),  # Project root
                os.path.join(os.path.dirname(__file__), '..', 'recommendation_app', 'application', 'credentials.json'),  # App directory
            ]

            api_key = None
            for path in possible_paths:
                test_key = load_api_key(path)
                if test_key:
                    logger.info(f"Loaded API key from: {path}")
                    api_key = test_key
                    break

        if not api_key:
            raise ValueError(
                "YouTube API key is required. Set YOUTUBE_API_KEY or run: python create_youtube_credentials.py --key YOUR_KEY"
            )

        self.youtube_api = YouTubeAPI(api_key)
        self.audio_extractor = AudioFeatureExtractor(cache_dir)

    def extract_track_features_from_youtube(self, query, use_cache=True, download_audio=True):
        """
        Complete pipeline: Search → Get video → Analyze audio → Return features
        Args:
            query: "Artist - Song Name" or just song name
            use_cache: Use cached features if available
            download_audio: If False, skip audio analysis (faster but no audio features)
        Returns:
            Dict with all track information + audio features
        """
        try:
            # Search for video
            search_results = self.youtube_api.search_video(query, max_results=1, use_api=False)

            if not search_results:
                logger.warning(f"No results found for: {query}")
                return None

            video_id = search_results[0]['id']

            # Get detailed video info
            video_details = self.youtube_api.get_video_details(video_id)

            if not video_details:
                logger.warning(f"Could not get details for: {video_id}")
                return None

            # Parse artist and track name from title
            artist, track_name = self._parse_title(video_details['title'])

            # Initialize track dict
            track_features = {
                'id': video_id,
                'track_name': track_name,
                'track_artist_name': artist,
                'youtube_url': f"https://youtube.com/watch?v={video_id}",
                'track_duration_ms': video_details['duration'],
                'views': video_details['views'],
                'likes': video_details['likes'],
                'track_album_release_date': video_details['published_at'],
                'track_album_type': 'single',  # Default for YouTube
                'track_album_name': track_name,  # Use track name as album
                'artist_genres': [],  # Will be filled if available
                'artist_popularity': self._estimate_popularity(video_details['views']),
                'track_popularity': self._estimate_popularity(video_details['views']),
                'track_explicit': False,  # Cannot determine from YouTube
            }

            # Extract audio features if requested
            if download_audio:
                logger.info(f"Analyzing audio for: {track_name}")
                audio_features = self.audio_extractor.download_and_analyze(video_id, use_cache)

                if audio_features:
                    # Add audio features with 'track_' prefix
                    for key, value in audio_features.items():
                        if key != 'duration_ms':
                            track_features[f'track_{key}'] = value
                else:
                    # Use default values if audio analysis fails
                    logger.warning(f"Using default audio features for {video_id}")
                    track_features.update(self._get_default_audio_features())
            else:
                track_features.update(self._get_default_audio_features())

            return track_features

        except Exception as e:
            logger.error(f"Error processing '{query}': {e}")
            return None

    def extract_playlist_features(self, playlist_url, max_videos=50, use_cache=True, download_audio=True):
        """
        Extract features for all videos in a YouTube playlist
        Args:
            playlist_url: Full YouTube playlist URL or playlist ID
            max_videos: Maximum number of videos to process
            use_cache: Use cached features if available
            download_audio: If False, skip audio download and use default audio features
        Returns:
            List of track feature dicts
        """
        # Extract playlist ID
        playlist_id = self.youtube_api.extract_playlist_id(playlist_url)
        if not playlist_id:
            logger.error(f"Invalid playlist URL: {playlist_url}")
            return []

        # Get video IDs from playlist
        logger.info(f"Fetching videos from playlist: {playlist_id}")
        video_ids = self.youtube_api.get_playlist_videos(playlist_id, max_videos)

        if not video_ids:
            logger.error("No videos found in playlist")
            return []

        logger.info(f"Processing {len(video_ids)} videos...")

        # Extract features for each video
        tracks = []
        for video_id in tqdm(video_ids, desc="Extracting features"):
            try:
                # Get video details
                video_details = self.youtube_api.get_video_details(video_id)
                if not video_details:
                    continue

                # Parse title
                artist, track_name = self._parse_title(video_details['title'])

                # Build track dict
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
                    'artist_popularity': self._estimate_popularity(video_details['views']),
                    'track_popularity': self._estimate_popularity(video_details['views']),
                    'track_explicit': False,
                }

                if download_audio:
                    # Extract audio features
                    audio_features = self.audio_extractor.download_and_analyze(video_id, use_cache)

                    if audio_features:
                        for key, value in audio_features.items():
                            if key != 'duration_ms':
                                track[f'track_{key}'] = value
                    else:
                        track.update(self._get_default_audio_features())
                else:
                    # Fast path for online requests: avoid request-time audio download.
                    track.update(self._get_default_audio_features())

                tracks.append(track)

            except Exception as e:
                logger.error(f"Error processing video {video_id}: {e}")
                continue

        logger.info(f"Successfully processed {len(tracks)} tracks")
        return tracks

    def extract_features_from_video_ids(self, video_ids, use_cache=True, download_audio=True, source='seed'):
        """
        Extract track features from explicit video IDs.
        Args:
            video_ids: List of YouTube video IDs
            use_cache: Use cached audio features
            download_audio: Whether to download audio for full feature extraction
            source: Source label for provenance
        Returns:
            List of track feature dicts
        """
        tracks = []
        for video_id in tqdm(video_ids, desc=f"Extracting features ({source})"):
            try:
                video_details = self.youtube_api.get_video_details(video_id)
                if not video_details:
                    continue

                artist, track_name = self._parse_title(video_details['title'])

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
                    'artist_popularity': self._estimate_popularity(video_details['views']),
                    'track_popularity': self._estimate_popularity(video_details['views']),
                    'track_explicit': False,
                    'source_type': source,
                    'source_ref': source,
                    'ingested_at': int(time.time()),
                }

                if download_audio:
                    audio_features = self.audio_extractor.download_and_analyze(video_id, use_cache)
                    if audio_features:
                        for key, value in audio_features.items():
                            if key != 'duration_ms':
                                track[f'track_{key}'] = value
                    else:
                        track.update(self._get_default_audio_features())
                else:
                    track.update(self._get_default_audio_features())

                tracks.append(track)
            except Exception as e:
                logger.error(f"Error processing video {video_id} in {source}: {e}")
                continue

        return tracks

    def expand_from_related_videos(self, seed_video_ids, per_seed=10):
        """
        Expansion source: collect related video IDs from seed videos.
        Args:
            seed_video_ids: Seed video IDs
            per_seed: Max related IDs per seed
        Returns:
            Unique related video IDs
        """
        related = []
        for seed_id in tqdm(seed_video_ids, desc='Fetching related IDs'):
            rel = self.youtube_api.get_related_videos(seed_id, max_results=per_seed)
            related.extend(rel)

        # Deduplicate while preserving order
        return list(dict.fromkeys(related))

    def expand_from_channels(self, channel_ids, per_channel=10):
        """
        Expansion source: collect video IDs from seed channels.
        Args:
            channel_ids: Channel IDs to expand from
            per_channel: Max IDs per channel
        Returns:
            Unique channel video IDs
        """
        collected = []
        for channel_id in tqdm(channel_ids, desc='Fetching channel IDs'):
            vids = self.youtube_api.get_channel_videos(channel_id, max_results=per_channel)
            collected.extend(vids)

        return list(dict.fromkeys(collected))

    def extract_popular_music_videos(self, region='US', max_results=50, use_cache=True):
        """
        Get features for trending/popular music videos
        Args:
            region: Region code (e.g., 'US', 'GB')
            max_results: Number of videos
            use_cache: Use cached features
        Returns:
            List of track feature dicts
        """
        logger.info(f"Fetching popular music videos in {region}...")

        # Get popular video IDs
        video_ids = self.youtube_api.get_popular_music_videos(region, max_results)

        if not video_ids:
            logger.error("No popular videos found")
            return []

        # Process each video
        tracks = []
        for video_id in tqdm(video_ids, desc="Processing popular videos"):
            try:
                video_details = self.youtube_api.get_video_details(video_id)
                if not video_details:
                    continue

                artist, track_name = self._parse_title(video_details['title'])

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
                    'artist_popularity': self._estimate_popularity(video_details['views']),
                    'track_popularity': self._estimate_popularity(video_details['views']),
                    'track_explicit': False,
                }

                # Extract audio features
                audio_features = self.audio_extractor.download_and_analyze(video_id, use_cache)

                if audio_features:
                    for key, value in audio_features.items():
                        if key != 'duration_ms':
                            track[f'track_{key}'] = value
                else:
                    track.update(self._get_default_audio_features())

                tracks.append(track)

            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
                continue

        return tracks

    def search_and_extract_multiple(self, queries, use_cache=True):
        """
        Search and extract features for multiple queries
        Args:
            queries: List of search queries
            use_cache: Use cached features
        Returns:
            List of track feature dicts
        """
        tracks = []
        for query in tqdm(queries, desc="Processing queries"):
            track = self.extract_track_features_from_youtube(query, use_cache)
            if track:
                tracks.append(track)

        return tracks

    @staticmethod
    def _parse_title(title):
        """
        Parse artist and track name from YouTube video title
        Args:
            title: Video title
        Returns:
            Tuple of (artist, track_name)
        """
        # Common patterns: "Artist - Song", "Artist: Song", "Song by Artist"
        separators = [' - ', ' – ', ' — ', ': ', ' by ']

        for sep in separators:
            if sep in title:
                parts = title.split(sep, 1)
                artist = parts[0].strip()
                track_name = parts[1].strip()

                # Clean up common suffixes
                suffixes = ['(Official Video)', '(Official Audio)', '(Lyrics)', '(Official Music Video)',
                           '[Official Video]', '[Official Audio]', 'Official Video', 'Official Audio']

                for suffix in suffixes:
                    track_name = track_name.replace(suffix, '').strip()
                    artist = artist.replace(suffix, '').strip()

                return artist, track_name

        # If no separator found, use channel as artist and title as track
        return 'Unknown Artist', title

    @staticmethod
    def _estimate_popularity(views):
        """
        Estimate popularity score (0-100) from view count
        Args:
            views: View count
        Returns:
            Popularity score
        """
        # Logarithmic scale
        if views <= 0:
            return 0

        import math
        # 1M views = ~70, 10M = ~80, 100M = ~90, 1B = ~100
        popularity = min(100, max(0, (math.log10(views) - 3) * 20))
        return int(popularity)

    @staticmethod
    def _get_default_audio_features():
        """
        Get default audio feature values when analysis fails
        Returns:
            Dict with default audio features
        """
        return {
            'track_danceability': 0.5,
            'track_energy': 0.5,
            'track_key': 0,
            'track_loudness': -10.0,
            'track_mode': 1,
            'track_speechiness': 0.1,
            'track_acousticness': 0.5,
            'track_instrumentalness': 0.5,
            'track_liveness': 0.2,
            'track_valence': 0.5,
            'track_tempo': 120.0,
        }
