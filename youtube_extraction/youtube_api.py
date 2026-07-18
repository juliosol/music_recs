"""
YouTube API Wrapper for music video search and metadata extraction
"""
import json
import os
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtubesearchpython import VideosSearch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeAPI:
    """Wrapper for YouTube Data API v3"""

    def __init__(self, api_key):
        """
        Initialize YouTube API client
        Args:
            api_key: YouTube Data API v3 key
        """
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)

    def search_video(self, query, max_results=10, use_api=False):
        """
        Search YouTube for music videos
        Args:
            query: Artist + Song name (e.g., "The Beatles Yesterday")
            max_results: Number of results to return
            use_api: If True, use API (costs quota), else use youtube-search-python
        Returns:
            List of dicts with video metadata
        """
        if not use_api:
            # Use youtube-search-python (no API quota cost)
            return self._search_without_api(query, max_results)

        try:
            request = self.youtube.search().list(
                part='snippet',
                q=query + ' official audio',
                type='video',
                videoCategoryId='10',  # Music category
                maxResults=max_results,
                order='relevance'
            )
            response = request.execute()

            results = []
            for item in response.get('items', []):
                results.append({
                    'id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'channel': item['snippet']['channelTitle'],
                    'description': item['snippet']['description'],
                    'thumbnail': item['snippet']['thumbnails']['high']['url']
                })

            return results

        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            return []

    def _search_without_api(self, query, max_results=10):
        """
        Search YouTube without using API quota
        Args:
            query: Search query
            max_results: Number of results
        Returns:
            List of video metadata dicts
        """
        try:
            search = VideosSearch(query + ' official audio', limit=max_results)
            results_data = search.result()

            results = []
            for item in results_data.get('result', []):
                results.append({
                    'id': item['id'],
                    'title': item['title'],
                    'channel': item['channel']['name'],
                    'description': item.get('descriptionSnippet', [{}])[0].get('text', ''),
                    'thumbnail': item['thumbnails'][0]['url'] if item.get('thumbnails') else '',
                    'duration': item.get('duration', ''),
                    'views': self._parse_view_count(item.get('viewCount', {}).get('short', '0'))
                })

            return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_video_details(self, video_id):
        """
        Get detailed information about a video
        Args:
            video_id: YouTube video ID
        Returns:
            Dict with video metadata
        """
        try:
            request = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            )
            response = request.execute()

            if not response.get('items'):
                return None

            item = response['items'][0]
            snippet = item['snippet']
            stats = item.get('statistics', {})
            content = item.get('contentDetails', {})

            return {
                'id': video_id,
                'title': snippet['title'],
                'channel': snippet['channelTitle'],
                'channel_id': snippet['channelId'],
                'description': snippet.get('description', ''),
                'published_at': snippet['publishedAt'],
                'duration': self._parse_duration(content.get('duration', '')),
                'views': int(stats.get('viewCount', 0)),
                'likes': int(stats.get('likeCount', 0)),
                'comments': int(stats.get('commentCount', 0)),
                'thumbnail': snippet['thumbnails']['high']['url']
            }

        except HttpError as e:
            logger.error(f"Error fetching video {video_id}: {e}")
            return None

    def get_playlist_videos(self, playlist_id, max_results=50):
        """
        Get all videos from a YouTube playlist
        Args:
            playlist_id: YouTube playlist ID
            max_results: Maximum number of videos to retrieve
        Returns:
            List of video IDs
        """
        try:
            video_ids = []
            next_page_token = None

            while len(video_ids) < max_results:
                request = self.youtube.playlistItems().list(
                    part='contentDetails',
                    playlistId=playlist_id,
                    maxResults=min(50, max_results - len(video_ids)),
                    pageToken=next_page_token
                )
                response = request.execute()

                for item in response.get('items', []):
                    video_ids.append(item['contentDetails']['videoId'])

                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break

            return video_ids

        except HttpError as e:
            logger.error(f"Error fetching playlist {playlist_id}: {e}")
            return []

    def get_popular_music_videos(self, region_code='US', max_results=50):
        """
        Get popular/trending music videos
        Args:
            region_code: Country code (e.g., 'US', 'GB')
            max_results: Number of videos
        Returns:
            List of video IDs
        """
        try:
            request = self.youtube.videos().list(
                part='id',
                chart='mostPopular',
                regionCode=region_code,
                videoCategoryId='10',  # Music category
                maxResults=max_results
            )
            response = request.execute()

            return [item['id'] for item in response.get('items', [])]

        except HttpError as e:
            logger.error(f"Error fetching popular videos: {e}")
            return []

    @staticmethod
    def _parse_duration(duration_str):
        """
        Parse ISO 8601 duration to milliseconds
        Args:
            duration_str: ISO 8601 duration (e.g., 'PT3M45S')
        Returns:
            Duration in milliseconds
        """
        if not duration_str:
            return 0

        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration_str)

        if not match:
            return 0

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return (hours * 3600 + minutes * 60 + seconds) * 1000

    @staticmethod
    def _parse_view_count(view_str):
        """
        Parse view count string like '1.2M views' to integer
        Args:
            view_str: View count string
        Returns:
            Integer view count
        """
        if not view_str:
            return 0

        # Remove non-numeric characters except K, M, B
        view_str = view_str.replace('views', '').replace(',', '').strip()

        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}

        for suffix, multiplier in multipliers.items():
            if suffix in view_str:
                try:
                    number = float(view_str.replace(suffix, ''))
                    return int(number * multiplier)
                except ValueError:
                    return 0

        try:
            return int(float(view_str))
        except ValueError:
            return 0

    @staticmethod
    def extract_playlist_id(url):
        """
        Extract playlist ID from YouTube URL
        Args:
            url: YouTube playlist URL
        Returns:
            Playlist ID or None
        """
        patterns = [
            r'list=([a-zA-Z0-9_-]+)',
            r'playlist\?list=([a-zA-Z0-9_-]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    @staticmethod
    def extract_video_id(url):
        """
        Extract video ID from YouTube URL
        Args:
            url: YouTube video URL
        Returns:
            Video ID or None
        """
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)',
            r'youtube\.com\/embed\/([a-zA-Z0-9_-]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None


def load_api_key(credentials_path='credentials.json'):
    """
    Load YouTube API key from credentials file or environment
    Args:
        credentials_path: Path to credentials JSON file
    Returns:
        API key string
    """
    env_key = os.getenv('YOUTUBE_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if env_key:
        return env_key

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    candidate_paths = []

    if credentials_path:
        candidate_paths.append(credentials_path)
        if not os.path.isabs(credentials_path):
            candidate_paths.append(os.path.join(base_dir, credentials_path))

    candidate_paths.extend([
        os.path.join(base_dir, 'credentials.json'),
        os.path.join(base_dir, 'recommendation_app', 'application', 'credentials.json'),
    ])

    seen_paths = set()
    for path in candidate_paths:
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)

        if not os.path.exists(path):
            continue

        try:
            with open(path, 'r') as f:
                credentials = json.load(f)
            api_key = credentials.get('youtube_api_key')
            if api_key:
                logger.info(f"Loaded YouTube API key from: {path}")
                return api_key
            logger.warning(f"'youtube_api_key' missing in: {path}")
        except Exception as e:
            logger.error(f"Error loading credentials from {path}: {e}")

    logger.error("YouTube API key not found. Set YOUTUBE_API_KEY or add youtube_api_key to credentials.json.")
    return None
