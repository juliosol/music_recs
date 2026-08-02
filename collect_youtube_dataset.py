"""
Collect YouTube Music Dataset
Script to build initial music database from YouTube
"""
import os
import argparse
import json
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


def collect_from_search_queries_with_source(pipeline, queries, use_cache=True, source_type='seed_query', source_ref='manual', download_audio=True):
    """
    Collect from query list with source provenance labels.
    """
    logger.info(f"Collecting {len(queries)} query tracks for source={source_ref}...")
    tracks = []
    for query in queries:
        track = pipeline.extract_track_features_from_youtube(query, use_cache=use_cache, download_audio=download_audio)
        if track:
            track['source_type'] = source_type
            track['source_ref'] = source_ref
            tracks.append(track)
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


def _load_seed_source_config(path):
    """
    Load seed source config JSON.
    Format:
    {
      "queries": [{"name": "global_pop", "items": ["...", "..."]}],
      "playlists": [{"name": "editorial_us", "items": ["url1", "url2"]}],
      "channels": [{"name": "label_channels", "items": ["UC..."]}]
    }
    """
    if not path or not os.path.exists(path):
        return {}

    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning(f"Could not load seed source config {path}: {e}")
    return {}


def _normalize_regions(regions):
    if isinstance(regions, str):
        return [r.strip() for r in regions.split(',') if r.strip()]
    if isinstance(regions, list):
        return [str(r).strip() for r in regions if str(r).strip()]
    return ['US']


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
    parser.add_argument('--seed-source-config', default=os.path.join('data_extraction', 'seed_sources.json'))
    parser.add_argument('--expand-related', action='store_true')
    parser.add_argument('--expand-from-channels', action='store_true')
    parser.add_argument('--related-per-seed', type=int, default=5)
    parser.add_argument('--videos-per-channel', type=int, default=10)
    parser.add_argument('--max-seed-expansion-base', type=int, default=100)
    parser.add_argument('--request-download-audio', action='store_true')
    args = parser.parse_args()

    use_cache = not args.no_cache
    download_audio = args.request_download_audio

    # Initialize pipeline
    logger.info("Initializing YouTube Music Pipeline...")
    pipeline = YouTubeMusicPipeline()

    all_tracks = []
    source_stats = {}

    def _add_tracks(source_name, tracks):
        if not tracks:
            return
        all_tracks.extend(tracks)
        source_stats[source_name] = source_stats.get(source_name, 0) + len(tracks)

    # Method 1: Collect from curated search queries (fast, diverse)
    if not args.skip_curated:
        logger.info("\n=== Collecting from curated queries ===")
        queries = get_diverse_music_queries()
        query_tracks = collect_from_search_queries_with_source(
            pipeline,
            queries,
            use_cache=use_cache,
            source_type='seed_query',
            source_ref='curated_builtin',
            download_audio=download_audio,
        )
        _add_tracks('seed_query:curated_builtin', query_tracks)
        logger.info(f"Collected {len(query_tracks)} tracks from curated queries")

    # Method 2: Collect from seed queries file
    seed_queries = _load_seed_lines(args.query_file)
    if seed_queries:
        logger.info("\n=== Collecting from seed queries file ===")
        seed_query_tracks = collect_from_search_queries_with_source(
            pipeline,
            seed_queries,
            use_cache=use_cache,
            source_type='seed_query',
            source_ref=f'file:{args.query_file}',
            download_audio=download_audio,
        )
        _add_tracks(f'seed_query:file:{args.query_file}', seed_query_tracks)
        logger.info(f"Collected {len(seed_query_tracks)} tracks from seed queries")

    # Method 3: Collect from popular videos (optional - uses API quota)
    if args.include_popular:
        regions = _normalize_regions(args.regions)
        logger.info("\n=== Collecting popular music ===")
        popular_tracks = collect_popular_music(
            pipeline,
            regions=regions,
            max_per_region=args.max_per_region,
            use_cache=use_cache
        )
        for t in popular_tracks:
            t['source_type'] = 'seed_chart'
            t['source_ref'] = 'youtube_most_popular'
        _add_tracks('seed_chart:youtube_most_popular', popular_tracks)
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
        for t in playlist_tracks:
            t['source_type'] = 'seed_playlist'
            t['source_ref'] = f'file:{args.playlist_file}'
        _add_tracks(f'seed_playlist:file:{args.playlist_file}', playlist_tracks)
        logger.info(f"Collected {len(playlist_tracks)} tracks from playlists")

    # Method 5: Config-driven seed sources (queries/playlists/channels)
    seed_cfg = _load_seed_source_config(args.seed_source_config)
    if seed_cfg:
        logger.info("\n=== Collecting from seed source config ===")

        for q_group in seed_cfg.get('queries', []):
            name = q_group.get('name', 'unnamed_query_group')
            items = q_group.get('items', [])
            if not items:
                continue
            tracks = collect_from_search_queries_with_source(
                pipeline,
                items,
                use_cache=use_cache,
                source_type='seed_query',
                source_ref=f'config_query:{name}',
                download_audio=download_audio,
            )
            _add_tracks(f'seed_query:config:{name}', tracks)

        for p_group in seed_cfg.get('playlists', []):
            name = p_group.get('name', 'unnamed_playlist_group')
            items = p_group.get('items', [])
            if not items:
                continue
            tracks = collect_from_playlists(
                pipeline,
                items,
                max_videos=args.max_per_playlist,
                use_cache=use_cache,
            )
            for t in tracks:
                t['source_type'] = 'seed_playlist'
                t['source_ref'] = f'config_playlist:{name}'
            _add_tracks(f'seed_playlist:config:{name}', tracks)

        for c_group in seed_cfg.get('channels', []):
            name = c_group.get('name', 'unnamed_channel_group')
            items = c_group.get('items', [])
            if not items:
                continue
            channel_video_ids = pipeline.expand_from_channels(items, per_channel=args.videos_per_channel)
            channel_tracks = pipeline.extract_features_from_video_ids(
                channel_video_ids,
                use_cache=use_cache,
                download_audio=download_audio,
                source=f'config_channel:{name}',
            )
            for t in channel_tracks:
                t['source_type'] = 'seed_channel'
                t['source_ref'] = f'config_channel:{name}'
            _add_tracks(f'seed_channel:config:{name}', channel_tracks)

    # Method 6: Expansion sources from already collected seeds
    if args.expand_related or args.expand_from_channels:
        logger.info("\n=== Running expansion sources ===")
        seed_ids = []
        for t in all_tracks:
            track_id = t.get('id')
            if track_id:
                seed_ids.append(track_id)
        seed_ids = list(dict.fromkeys(seed_ids))[:args.max_seed_expansion_base]

        if args.expand_related and seed_ids:
            related_ids = pipeline.expand_from_related_videos(seed_ids, per_seed=args.related_per_seed)
            related_tracks = pipeline.extract_features_from_video_ids(
                related_ids,
                use_cache=use_cache,
                download_audio=download_audio,
                source='exp_related',
            )
            for t in related_tracks:
                t['source_type'] = 'exp_related'
                t['source_ref'] = 'related_to_seed_videos'
            _add_tracks('exp_related:seed_videos', related_tracks)

        if args.expand_from_channels:
            channel_ids = []
            for t in all_tracks:
                cid = t.get('channel_id')
                if cid:
                    channel_ids.append(cid)
            channel_ids = list(dict.fromkeys(channel_ids))[:args.max_seed_expansion_base]
            if channel_ids:
                exp_channel_ids = pipeline.expand_from_channels(channel_ids, per_channel=args.videos_per_channel)
                exp_channel_tracks = pipeline.extract_features_from_video_ids(
                    exp_channel_ids,
                    use_cache=use_cache,
                    download_audio=download_audio,
                    source='exp_channel',
                )
                for t in exp_channel_tracks:
                    t['source_type'] = 'exp_channel'
                    t['source_ref'] = 'seed_channel_uploads'
                _add_tracks('exp_channel:seed_channels', exp_channel_tracks)

    # Check if we have any tracks
    if not all_tracks:
        logger.error("No tracks collected! Check your API key and internet connection.")
        return

    # Convert to DataFrame
    logger.info(f"\n=== Processing {len(all_tracks)} total tracks ===")
    df = pd.DataFrame(all_tracks)

    if 'source_type' not in df.columns:
        df['source_type'] = 'unknown'
    if 'source_ref' not in df.columns:
        df['source_ref'] = 'unknown'

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
    if source_stats:
        logger.info("Source contribution summary:")
        for src, count in sorted(source_stats.items(), key=lambda kv: kv[1], reverse=True):
            logger.info(f"  - {src}: {count}")
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
