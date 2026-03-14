"""
Flask Routes for YouTube Music Recommender
"""
import os
import sys
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

        # Extract features from user's playlist
        user_playlist_df = extract_features_from_youtube_playlist(youtube_url, max_videos=20)

        if user_playlist_df is None or user_playlist_df.empty:
            return render_template('error.html',
                                 message="Could not extract features from playlist. Please check the URL and try again.")

        logger.info(f"Generating recommendations with diversity={diversity}")

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

        if recommendations.empty:
            return render_template('error.html',
                                 message="No recommendations found. Try adjusting the diversity slider.")

        # Diversify by artist
        recommendations = diversify_by_artist(recommendations, max_per_artist=max_per_artist)

        # Limit to requested number
        recommendations = recommendations.head(number_of_recs)

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

        logger.info(f"Returning {len(my_songs)} recommendations")

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
    return jsonify({
        'status': 'ok',
        'songs_loaded': len(allSongDF),
        'features_loaded': len(allSongFeatureSetDF)
    })


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message="Internal server error. Please try again."), 500
