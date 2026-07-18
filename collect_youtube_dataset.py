"""
Collect YouTube Music Dataset
Script to build initial music database from YouTube
"""
import os
import argparse
import pandas as pd
import logging
from youtube_extraction.youtube_pipeline import YouTubeMusicPipeline
from youtube_extraction.feature_eng_youtube import playlist_preprocessing_youtube

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_from_search_queries(pipeline, queries, use_cache=True):
    """
    Collect music from search queries
    Args:
        pipeline: YouTubeMusicPipeline instance
        queries: List of search queries
        use_cache: Use cached features
    Returns:
        List of track dicts
    """
    logger.info(f"Collecting {len(queries)} songs from search queries...")
    tracks = pipeline.search_and_extract_multiple(queries, use_cache)
    return tracks


def collect_from_playlists(pipeline, playlist_urls, max_videos=50, use_cache=True):
    """
    Collect music from YouTube playlists
    Args:
        pipeline: YouTubeMusicPipeline instance
        playlist_urls: List of playlist URLs
        use_cache: Use cached features
    Returns:
        List of track dicts
    """
    all_tracks = []

    for url in playlist_urls:
        logger.info(f"Processing playlist: {url}")
        tracks = pipeline.extract_playlist_features(url, max_videos=max_videos, use_cache=use_cache)
        all_tracks.extend(tracks)

    return all_tracks


def collect_popular_music(pipeline, regions=['US'], max_per_region=50, use_cache=True):
    """
    Collect popular/trending music videos
    Args:
        pipeline: YouTubeMusicPipeline instance
        regions: List of region codes
        max_per_region: Max videos per region
        use_cache: Use cached features
    Returns:
        List of track dicts
    """
    all_tracks = []

    for region in regions:
        logger.info(f"Collecting popular music from {region}...")
        tracks = pipeline.extract_popular_music_videos(region, max_per_region, use_cache)
        all_tracks.extend(tracks)

    return all_tracks


def get_diverse_music_queries():
    """
    Get diverse set of music queries across genres
    Returns:
        List of search queries
    """
    # Popular songs across different genres
    queries = [
        # Pop
        "Taylor Swift Anti-Hero",
        "The Weeknd Blinding Lights",
        "Dua Lipa Levitating",
        "Harry Styles As It Was",
        "Olivia Rodrigo drivers license",

        # Rock
        "Queen Bohemian Rhapsody",
        "The Beatles Hey Jude",
        "Led Zeppelin Stairway to Heaven",
        "Nirvana Smells Like Teen Spirit",
        "Foo Fighters Everlong",

        # Hip Hop / R&B
        "Drake God's Plan",
        "Kendrick Lamar HUMBLE",
        "Travis Scott SICKO MODE",
        "Post Malone Circles",
        "SZA Kill Bill",

        # Electronic / Dance
        "Daft Punk Get Lucky",
        "Calvin Harris Summer",
        "The Chainsmokers Closer",
        "Avicii Wake Me Up",
        "Swedish House Mafia Don't You Worry Child",

        # Latin
        "Bad Bunny Tití Me Preguntó",
        "Shakira Hips Don't Lie",
        "J Balvin Mi Gente",
        "Daddy Yankee Gasolina",
        "Karol G TQG",

        # Country
        "Luke Combs Fast Car",
        "Morgan Wallen Last Night",
        "Zach Bryan Something in the Orange",
        "Chris Stapleton Tennessee Whiskey",
        "Carrie Underwood Before He Cheats",

        # Alternative / Indie
        "Radiohead Creep",
        "Arctic Monkeys Do I Wanna Know",
        "Tame Impala The Less I Know The Better",
        "Coldplay Fix You",
        "Imagine Dragons Believer",

        # R&B / Soul
        "Frank Ocean Thinkin Bout You",
        "The Weeknd Starboy",
        "Bruno Mars 24K Magic",
        "Beyoncé Halo",
        "Alicia Keys Fallin",

        # Classic / Oldies
        "Elvis Presley Can't Help Falling in Love",
        "Frank Sinatra My Way",
        "Louis Armstrong What a Wonderful World",
        "The Beach Boys Good Vibrations",
        "David Bowie Heroes",
    ]

    return queries


def _load_seed_lines(path):
    """
    Load seed lines from a text file (ignores blanks and comments)
    """
    if not path or not os.path.exists(path):
        return []

    lines = []
    with open(path, 'r') as f:
        for raw in f.readlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            lines.append(line)
    return lines


def _merge_existing_raw(new_df, data_dir):
    """
    Merge with existing raw dataset if present.
    """
    existing_raw_path = os.path.join(data_dir, 'youtube_music_raw.csv')
    if not os.path.exists(existing_raw_path):
        return new_df

    try:
        existing_df = pd.read_csv(existing_raw_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        if 'id' in combined.columns:
            combined = combined.drop_duplicates(subset=['id'])
        return combined
    except Exception as e:
        logger.warning(f"Could not merge existing raw dataset: {e}")
        return new_df


def main():
    """Main function to collect YouTube music dataset"""

    parser = argparse.ArgumentParser(description="Collect and expand the YouTube music dataset")
    parser.add_argument('--query-file', default=os.path.join('data_extraction', 'seed_queries.txt'))
    parser.add_argument('--playlist-file', default=os.path.join('data_extraction', 'seed_playlists.txt'))
    parser.add_argument('--max-per-playlist', type=int, default=50)
    parser.add_argument('--include-popular', action='store_true')
    parser.add_argument('--regions', default='US')
    parser.add_argument('--max-per-region', type=int, default=50)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--skip-curated', action='store_true')
    args = parser.parse_args()

    use_cache = not args.no_cache

    # Initialize pipeline
    logger.info("Initializing YouTube Music Pipeline...")
    pipeline = YouTubeMusicPipeline()

    all_tracks = []

    # Method 1: Collect from curated search queries (fast, diverse)
    if not args.skip_curated:
        logger.info("\n=== Collecting from curated queries ===")
        queries = get_diverse_music_queries()
        query_tracks = collect_from_search_queries(pipeline, queries, use_cache=use_cache)
        all_tracks.extend(query_tracks)
        logger.info(f"Collected {len(query_tracks)} tracks from curated queries")

    # Method 2: Collect from seed queries file
    seed_queries = _load_seed_lines(args.query_file)
    if seed_queries:
        logger.info("\n=== Collecting from seed queries file ===")
        seed_query_tracks = collect_from_search_queries(pipeline, seed_queries, use_cache=use_cache)
        all_tracks.extend(seed_query_tracks)
        logger.info(f"Collected {len(seed_query_tracks)} tracks from seed queries")

    # Method 3: Collect from popular videos (optional - uses API quota)
    if args.include_popular:
        regions = [r.strip() for r in args.regions.split(',') if r.strip()]
        logger.info("\n=== Collecting popular music ===")
        popular_tracks = collect_popular_music(
            pipeline,
            regions=regions,
            max_per_region=args.max_per_region,
            use_cache=use_cache
        )
        all_tracks.extend(popular_tracks)
        logger.info(f"Collected {len(popular_tracks)} popular tracks")

    # Method 4: Collect from playlists file
    seed_playlists = _load_seed_lines(args.playlist_file)
    if seed_playlists:
        logger.info("\n=== Collecting from playlists ===")
        playlist_tracks = collect_from_playlists(
            pipeline,
            seed_playlists,
            max_videos=args.max_per_playlist,
            use_cache=use_cache
        )
        all_tracks.extend(playlist_tracks)
        logger.info(f"Collected {len(playlist_tracks)} tracks from playlists")

    # Check if we have any tracks
    if not all_tracks:
        logger.error("No tracks collected! Check your API key and internet connection.")
        return

    # Convert to DataFrame
    logger.info(f"\n=== Processing {len(all_tracks)} total tracks ===")
    df = pd.DataFrame(all_tracks)

    # Save raw data
    os.makedirs('data_extraction', exist_ok=True)
    if args.append:
        df = _merge_existing_raw(df, 'data_extraction')
    df.to_csv('data_extraction/youtube_music_raw.csv', index=False)
    logger.info("Saved raw data to data_extraction/youtube_music_raw.csv")

    # Preprocess data
    logger.info("\n=== Preprocessing data ===")
    processed_df, normalized_df = playlist_preprocessing_youtube(df, 'youtube_music')

    logger.info(f"\n=== Collection Complete! ===")
    logger.info(f"Total tracks: {len(processed_df)}")
    logger.info(f"Files created:")
    logger.info(f"  - data_extraction/youtube_music_raw.csv")
    logger.info(f"  - data_extraction/youtube_music.csv")
    logger.info(f"  - data_extraction/normalized_youtube_music.csv")

    # Print sample
    logger.info(f"\nSample tracks:")
    for i, row in processed_df.head(5).iterrows():
        logger.info(f"  {row['track_artist_name']} - {row['track_name']}")


if __name__ == '__main__':
    main()
