"""
Flask Routes for YouTube Music Recommender
"""
import os
import sys
import time
from collections import deque
from threading import Lock
import pandas as pd
from flask import Flask, render_template, request, jsonify
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from application import app
from application.features_youtube import extract_features_from_youtube_playlist
from application.model import recommend_from_playlist
from application.diversity import create_varied_playlist, create_mood_based_playlist, diversify_by_artist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OBS_LOCK = Lock()
OBS_METRICS = {
    'recommend_requests_total': 0,
    'recommend_success_total': 0,
    'recommend_error_total': 0,
    'fallback_counts': {
        'none': 0,
        'feature_space': 0,
        'popular_no_extraction': 0,
        'popular_after_feature': 0,
    },
    'event_counts': {
        'play_click': 0,
        'copy_links': 0,
        'download_playlist': 0,
    },
    'recommend_latency_ms': deque(maxlen=300),
    'extraction_ms': deque(maxlen=300),
    'ranking_ms': deque(maxlen=300),
    'formatting_ms': deque(maxlen=300),
    'playlist_match_rate': deque(maxlen=300),
}


def _p95(values):
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(0.95 * (len(sorted_vals) - 1))
    return float(sorted_vals[idx])


def _observe_recommend(total_ms, extraction_ms, ranking_ms, formatting_ms,
                       extracted_tracks, matched_tracks, fallback_mode, success):
    with OBS_LOCK:
        OBS_METRICS['recommend_requests_total'] += 1
        if success:
            OBS_METRICS['recommend_success_total'] += 1
        else:
            OBS_METRICS['recommend_error_total'] += 1

        if fallback_mode not in OBS_METRICS['fallback_counts']:
            OBS_METRICS['fallback_counts'][fallback_mode] = 0
        OBS_METRICS['fallback_counts'][fallback_mode] += 1

        OBS_METRICS['recommend_latency_ms'].append(float(total_ms))
        OBS_METRICS['extraction_ms'].append(float(extraction_ms))
        OBS_METRICS['ranking_ms'].append(float(ranking_ms))
        OBS_METRICS['formatting_ms'].append(float(formatting_ms))

        if extracted_tracks > 0:
            match_rate = min(max(matched_tracks / extracted_tracks, 0.0), 1.0)
            OBS_METRICS['playlist_match_rate'].append(float(match_rate))


def _observability_snapshot():
    with OBS_LOCK:
        req_total = OBS_METRICS['recommend_requests_total']
        success_total = OBS_METRICS['recommend_success_total']
        error_total = OBS_METRICS['recommend_error_total']

        latency_vals = list(OBS_METRICS['recommend_latency_ms'])
        extraction_vals = list(OBS_METRICS['extraction_ms'])
        ranking_vals = list(OBS_METRICS['ranking_ms'])
        formatting_vals = list(OBS_METRICS['formatting_ms'])
        match_vals = list(OBS_METRICS['playlist_match_rate'])

        return {
            'recommend': {
                'requests_total': req_total,
                'success_total': success_total,
                'error_total': error_total,
                'success_rate': round((success_total / req_total), 4) if req_total else 0.0,
                'fallback_counts': dict(OBS_METRICS['fallback_counts']),
            },
            'latency_ms': {
                'recommend_p95': round(_p95(latency_vals), 2),
                'recommend_avg': round(sum(latency_vals) / len(latency_vals), 2) if latency_vals else 0.0,
                'extraction_p95': round(_p95(extraction_vals), 2),
                'ranking_p95': round(_p95(ranking_vals), 2),
                'formatting_p95': round(_p95(formatting_vals), 2),
            },
            'playlist_match_rate': {
                'avg': round(sum(match_vals) / len(match_vals), 4) if match_vals else 0.0,
                'recent_samples': len(match_vals),
            },
            'events': dict(OBS_METRICS['event_counts']),
        }


def _queue_missing_playlist_tracks(playlist_url, user_playlist_df):
    """Persist missing playlist tracks for offline enrichment jobs."""
    try:
        os.makedirs('data_extraction', exist_ok=True)
        queue_path = os.path.join('data_extraction', 'missing_playlist_tracks.csv')

        if user_playlist_df is None or user_playlist_df.empty:
            return

        ids = user_playlist_df.get('id', pd.Series(dtype='object')).astype(str)
        known_ids = set(allSongFeatureSetDF['id'].astype(str).values)
        missing_ids = sorted([track_id for track_id in ids if track_id and track_id not in known_ids])

        if not missing_ids:
            return

        rows = pd.DataFrame({
            'playlist_url': [playlist_url] * len(missing_ids),
            'track_id': missing_ids,
            'queued_at': [int(time.time())] * len(missing_ids),
        })
        rows.to_csv(queue_path, mode='a', header=not os.path.exists(queue_path), index=False)
        logger.info(f"Queued {len(missing_ids)} missing tracks to {queue_path}")
    except Exception as e:
        logger.warning(f"Failed to queue missing tracks: {e}")


def _feature_space_fallback_recommendations(user_playlist_df, candidate_size=50):
    """Fallback recommender using normalized feature space when user IDs are mostly unseen."""
    if user_playlist_df is None or user_playlist_df.empty:
        return pd.DataFrame()

    if allSongFeatureSetDF.empty or allSongDF.empty:
        return pd.DataFrame()

    candidate_features = allSongFeatureSetDF.copy()
    if 'id' not in candidate_features.columns or 'id' not in user_playlist_df.columns:
        return pd.DataFrame()

    feature_cols = [c for c in candidate_features.columns if c != 'id' and c in user_playlist_df.columns]
    if not feature_cols:
        return pd.DataFrame()

    user_matrix = user_playlist_df[feature_cols].fillna(0)
    if user_matrix.empty:
        return pd.DataFrame()

    user_profile = user_matrix.mean(axis=0)

    from sklearn.metrics.pairwise import cosine_similarity
    feature_matrix = candidate_features[feature_cols].fillna(0)
    sims = cosine_similarity(feature_matrix.values, user_profile.values.reshape(1, -1))[:, 0]

    scored = candidate_features[['id']].copy()
    scored['sim'] = sims

    seen_ids = set(user_playlist_df['id'].astype(str).values)
    scored = scored[~scored['id'].astype(str).isin(seen_ids)]

    top_ids = scored.sort_values('sim', ascending=False).head(candidate_size)['id'].astype(str).tolist()
    if not top_ids:
        return pd.DataFrame()

    results = allSongDF[allSongDF['id'].astype(str).isin(top_ids)].copy()
    if results.empty:
        return pd.DataFrame()

    order_map = {song_id: idx for idx, song_id in enumerate(top_ids)}
    results['__order'] = results['id'].astype(str).map(order_map)
    results = results.sort_values('__order').drop(columns='__order')
    return results


def _popular_fallback_recommendations(limit=50):
    """Final fallback: serve popular tracks if personalized candidates are empty."""
    if allSongDF.empty:
        return pd.DataFrame()

    if 'track_popularity' in allSongDF.columns:
        return allSongDF.sort_values('track_popularity', ascending=False).head(limit).copy()
    if 'views' in allSongDF.columns:
        return allSongDF.sort_values('views', ascending=False).head(limit).copy()
    return allSongDF.head(limit).copy()

# Load YouTube music dataset
try:
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'data_extraction')

    all_songs_path = os.path.join(data_path, 'youtube_music.csv')
    normalized_path = os.path.join(data_path, 'normalized_youtube_music.csv')

    if os.path.exists(all_songs_path) and os.path.exists(normalized_path):
        allSongDF = pd.read_csv(all_songs_path)
        allSongFeatureSetDF = pd.read_csv(normalized_path)
        logger.info(f"Loaded {len(allSongDF)} songs from database")
    else:
        logger.warning("Dataset not found. Please run collect_youtube_dataset.py first!")
        allSongDF = pd.DataFrame()
        allSongFeatureSetDF = pd.DataFrame()
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    allSongDF = pd.DataFrame()
    allSongFeatureSetDF = pd.DataFrame()


@app.route("/")
def home():
    """Home page"""
    return render_template("home_youtube.html",
                         total_songs=len(allSongDF) if not allSongDF.empty else 0)


@app.route("/about")
def about():
    """About page"""
    return render_template("about_youtube.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    """Generate recommendations from YouTube playlist"""
    route_start = time.perf_counter()
    extraction_ms = 0.0
    ranking_ms = 0.0
    formatting_ms = 0.0
    extracted_tracks = 0
    matched_tracks = 0
    fallback_mode = 'none'
    success = False

    try:
        # Check if dataset is loaded
        if allSongDF.empty or allSongFeatureSetDF.empty:
            return render_template('error.html',
                                 message="Music database not loaded. Please run collect_youtube_dataset.py first!")

        # Get parameters from form
        youtube_url = request.form.get('youtube_playlist_url', '').strip()
        number_of_recs = int(request.form.get('number-of-recs', 10))
        diversity = float(request.form.get('diversity', 0.3))
        max_per_artist = int(request.form.get('max_per_artist', 3))

        if not youtube_url:
            return render_template('error.html',
                                 message="Please provide a YouTube playlist URL")

        logger.info(f"Processing playlist: {youtube_url}")

        extraction_start = time.perf_counter()

        # Fast request-time path: avoid heavy audio download during live user requests.
        user_playlist_df = extract_features_from_youtube_playlist(
            youtube_url,
            max_videos=20,
            download_audio=False,
        )
        extraction_ms = (time.perf_counter() - extraction_start) * 1000

        if user_playlist_df is None or user_playlist_df.empty:
            logger.warning("Playlist extraction returned no tracks; using popular fallback")
            fallback_mode = 'popular_no_extraction'
            recommendations = _popular_fallback_recommendations(limit=max(number_of_recs * 2, 20))
            if recommendations.empty:
                return render_template('error.html',
                                     message="Could not extract features from playlist. Please check the URL and try again.")
        else:
            extracted_tracks = len(user_playlist_df)
            if 'id' in user_playlist_df.columns and 'id' in allSongFeatureSetDF.columns:
                known_ids = set(allSongFeatureSetDF['id'].astype(str).values)
                matched_tracks = sum(user_playlist_df['id'].astype(str).isin(known_ids))

            _queue_missing_playlist_tracks(youtube_url, user_playlist_df)

            logger.info(f"Generating recommendations with diversity={diversity}")
            ranking_start = time.perf_counter()

            # Generate recommendations
            if diversity > 0.1:
                # Use diversity algorithm
                recommendations = create_varied_playlist(
                    allSongDF,
                    allSongFeatureSetDF,
                    user_playlist_df,
                    playlist_length=number_of_recs * 2,  # Get more, then filter
                    diversity=diversity
                )
            else:
                # Standard similarity-based recommendations
                recommendations = recommend_from_playlist(
                    allSongDF,
                    allSongFeatureSetDF,
                    user_playlist_df
                )

            # If sparse ID overlap prevents model output, use feature-space fallback.
            if recommendations is None or recommendations.empty:
                logger.warning("Primary recommendation path returned empty; using feature-space fallback")
                fallback_mode = 'feature_space'
                recommendations = _feature_space_fallback_recommendations(
                    user_playlist_df,
                    candidate_size=max(number_of_recs * 3, 30),
                )

            # Final fallback to popular songs to always return results.
            if recommendations is None or recommendations.empty:
                logger.warning("Feature-space fallback returned empty; using popular fallback")
                fallback_mode = 'popular_after_feature'
                recommendations = _popular_fallback_recommendations(limit=max(number_of_recs * 2, 20))

            if recommendations is None or recommendations.empty:
                return render_template('error.html',
                                     message="No recommendations found. Try adjusting the diversity slider.")

            ranking_ms = (time.perf_counter() - ranking_start) * 1000

        # Diversify by artist
        recommendations = diversify_by_artist(recommendations, max_per_artist=max_per_artist)

        # Limit to requested number
        recommendations = recommendations.head(number_of_recs)

        # Format results
        formatting_start = time.perf_counter()
        my_songs = []
        for i, row in recommendations.iterrows():
            title = f"{row['track_artist_name']} - {row['track_name']}"
            url = row.get('youtube_url', f"https://youtube.com/watch?v={row['id']}")
            video_id = row['id']

            my_songs.append({
                'title': title,
                'url': url,
                'video_id': video_id,
                'artist': row['track_artist_name'],
                'track': row['track_name']
            })
        formatting_ms = (time.perf_counter() - formatting_start) * 1000

        logger.info(f"Returning {len(my_songs)} recommendations")
        logger.info(f"Recommend route completed in {(time.perf_counter() - route_start):.2f}s")
        success = True

        return render_template(
            'results_youtube.html',
            songs=my_songs,
            total_songs=len(allSongDF) if not allSongDF.empty else 0
        )

    except Exception as e:
        logger.error(f"Error in recommend route: {e}")
        import traceback
        traceback.print_exc()
        return render_template('error.html',
                             message=f"An error occurred: {str(e)}")
    finally:
        total_ms = (time.perf_counter() - route_start) * 1000
        _observe_recommend(
            total_ms=total_ms,
            extraction_ms=extraction_ms,
            ranking_ms=ranking_ms,
            formatting_ms=formatting_ms,
            extracted_tracks=extracted_tracks,
            matched_tracks=matched_tracks,
            fallback_mode=fallback_mode,
            success=success,
        )


@app.route("/discover", methods=["POST"])
def discover():
    """Discover new music based on mood"""
    try:
        if allSongDF.empty or allSongFeatureSetDF.empty:
            return render_template('error.html',
                                 message="Music database not loaded. Please run collect_youtube_dataset.py first!")

        # Get parameters
        mood = request.form.get('mood', 'energetic')
        number_of_recs = int(request.form.get('number-of-recs', 20))

        logger.info(f"Discovering music for mood: {mood}")

        # Create mood-based playlist
        recommendations = create_mood_based_playlist(
            allSongDF,
            allSongFeatureSetDF,
            target_mood=mood,
            playlist_length=number_of_recs
        )

        if recommendations.empty:
            return render_template('error.html',
                                 message=f"No songs found for mood: {mood}")

        # Format results
        my_songs = []
        for i, row in recommendations.iterrows():
            title = f"{row['track_artist_name']} - {row['track_name']}"
            url = row.get('youtube_url', f"https://youtube.com/watch?v={row['id']}")
            video_id = row['id']

            my_songs.append({
                'title': title,
                'url': url,
                'video_id': video_id,
                'artist': row['track_artist_name'],
                'track': row['track_name']
            })

        return render_template(
            'results_youtube.html',
            songs=my_songs,
            total_songs=len(allSongDF) if not allSongDF.empty else 0
        )

    except Exception as e:
        logger.error(f"Error in discover route: {e}")
        return render_template('error.html',
                             message=f"An error occurred: {str(e)}")


@app.route("/health")
def health():
    """Health check endpoint"""
    obs = _observability_snapshot()
    return jsonify({
        'status': 'ok',
        'songs_loaded': len(allSongDF),
        'features_loaded': len(allSongFeatureSetDF),
        'observability': obs,
    })


@app.route("/events", methods=["POST"])
def track_event():
    """Lightweight event collector for recommendation UI interactions."""
    payload = request.get_json(silent=True) or {}
    event_name = payload.get('event')

    if event_name not in OBS_METRICS['event_counts']:
        return jsonify({'status': 'ignored'}), 200

    with OBS_LOCK:
        OBS_METRICS['event_counts'][event_name] += 1

    return jsonify({'status': 'ok'}), 200


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message="Internal server error. Please try again."), 500
