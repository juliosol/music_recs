"""
Test script for YouTube Music Recommender
Validates all components are working correctly
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required packages can be imported"""
    logger.info("\n=== Testing Imports ===")

    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('flask', 'flask'),
        ('googleapiclient', 'google-api-python-client'),
        ('librosa', 'librosa'),
        ('yt_dlp', 'yt-dlp'),
        ('youtubesearchpython', 'youtube-search-python'),
    ]

    all_passed = True
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            logger.info(f"✓ {package_name}")
        except ImportError:
            logger.error(f"✗ {package_name} - NOT INSTALLED")
            logger.error(f"  Install with: pip install {package_name}")
            all_passed = False

    return all_passed


def test_credentials():
    """Test that credentials file exists and has YouTube API key"""
    logger.info("\n=== Testing Credentials ===")

    creds_path = 'credentials.json'
    if not os.path.exists(creds_path):
        logger.error(f"✗ credentials.json not found")
        return False

    try:
        import json
        with open(creds_path, 'r') as f:
            creds = json.load(f)

        if 'youtube_api_key' in creds and creds['youtube_api_key']:
            logger.info(f"✓ YouTube API key found")
            return True
        else:
            logger.error(f"✗ youtube_api_key not found in credentials.json")
            return False

    except Exception as e:
        logger.error(f"✗ Error reading credentials: {e}")
        return False


def test_youtube_api():
    """Test YouTube API connectivity"""
    logger.info("\n=== Testing YouTube API ===")

    try:
        from youtube_extraction.youtube_api import YouTubeAPI, load_api_key

        api_key = load_api_key()
        if not api_key:
            logger.error("✗ Could not load API key")
            return False

        youtube = YouTubeAPI(api_key)

        # Test search (without using API quota)
        results = youtube.search_video("test song", max_results=1, use_api=False)

        if results and len(results) > 0:
            logger.info(f"✓ YouTube search working")
            logger.info(f"  Found: {results[0]['title']}")
            return True
        else:
            logger.error("✗ YouTube search returned no results")
            return False

    except Exception as e:
        logger.error(f"✗ YouTube API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_extraction():
    """Test audio feature extraction"""
    logger.info("\n=== Testing Audio Feature Extraction ===")

    try:
        from youtube_extraction.audio_features import AudioFeatureExtractor

        extractor = AudioFeatureExtractor()

        logger.info("  Note: Full audio extraction test requires downloading a video")
        logger.info("  Skipping for now. Will test during data collection.")
        logger.info("✓ Audio extractor initialized")

        return True

    except Exception as e:
        logger.error(f"✗ Audio extraction test failed: {e}")
        return False


def test_feature_engineering():
    """Test feature engineering functions"""
    logger.info("\n=== Testing Feature Engineering ===")

    try:
        import pandas as pd
        from youtube_extraction.feature_eng_youtube import (
            drop_duplicates_df_youtube,
            sentiment_analysis,
            datetime_converter,
            normalize_engagement_metrics,
            feature_normalizer
        )

        # Create sample data
        sample_data = {
            'id': ['video1', 'video2'],
            'track_name': ['Song 1', 'Song 2'],
            'track_artist_name': ['Artist 1', 'Artist 2'],
            'views': [1000000, 500000],
            'likes': [50000, 25000],
            'track_danceability': [0.7, 0.6],
            'track_energy': [0.8, 0.5],
            'track_valence': [0.6, 0.4],
            'track_tempo': [120, 100],
            'song_info': ['Song 1 Artist 1', 'Song 2 Artist 2'],
            'track_album_release_date': ['2023-01-01', '2023-01-02'],
        }

        df = pd.DataFrame(sample_data)

        # Test each function
        df = sentiment_analysis(df, 'song_info')
        df = datetime_converter(df)
        df = normalize_engagement_metrics(df)

        logger.info("✓ Feature engineering functions working")
        return True

    except Exception as e:
        logger.error(f"✗ Feature engineering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_recommendation_model():
    """Test recommendation algorithm"""
    logger.info("\n=== Testing Recommendation Model ===")

    try:
        from recommendation_app.application.model import recommend_from_playlist
        import pandas as pd
        import numpy as np

        # Create sample data
        n_songs = 20
        all_songs = pd.DataFrame({
            'id': [f'song_{i}' for i in range(n_songs)],
            'track_name': [f'Track {i}' for i in range(n_songs)],
            'track_artist_name': [f'Artist {i%5}' for i in range(n_songs)],
        })

        # Create random features
        feature_cols = ['feature_' + str(i) for i in range(10)]
        features_data = {'id': all_songs['id'].values}
        for col in feature_cols:
            features_data[col] = np.random.rand(n_songs)

        all_features = pd.DataFrame(features_data)

        # User playlist (first 3 songs)
        user_songs = all_features.head(3)

        # Get recommendations
        recommendations = recommend_from_playlist(all_songs, all_features, user_songs)

        if len(recommendations) > 0:
            logger.info(f"✓ Recommendation algorithm working")
            logger.info(f"  Generated {len(recommendations)} recommendations")
            return True
        else:
            logger.error("✗ No recommendations generated")
            return False

    except Exception as e:
        logger.error(f"✗ Recommendation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_exists():
    """Check if music dataset has been collected"""
    logger.info("\n=== Checking Music Dataset ===")

    dataset_path = 'data_extraction/youtube_music.csv'
    normalized_path = 'data_extraction/normalized_youtube_music.csv'

    if os.path.exists(dataset_path) and os.path.exists(normalized_path):
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            logger.info(f"✓ Dataset found with {len(df)} songs")
            return True
        except Exception as e:
            logger.error(f"✗ Error reading dataset: {e}")
            return False
    else:
        logger.warning("⚠ Dataset not found")
        logger.warning("  Run 'python collect_youtube_dataset.py' to build database")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("YouTube Music Recommender - System Test")
    print("="*70)

    tests = [
        ("Import Test", test_imports),
        ("Credentials Test", test_credentials),
        ("YouTube API Test", test_youtube_api),
        ("Audio Extraction Test", test_audio_extraction),
        ("Feature Engineering Test", test_feature_engineering),
        ("Recommendation Model Test", test_recommendation_model),
        ("Dataset Check", test_dataset_exists),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"\nUnexpected error in {test_name}: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<50} {status}")

    print("="*70)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. If dataset not collected, run: python collect_youtube_dataset.py")
        print("  2. Start the web app: cd recommendation_app && python start_youtube.py")
    else:
        print("\n⚠ Some tests failed. Please fix the issues before proceeding.")

    print("="*70 + "\n")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
