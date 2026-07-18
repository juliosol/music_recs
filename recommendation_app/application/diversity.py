"""
Diversity and Variety Logic for Music Recommendations
Ensures playlists are varied and not repetitive
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_diversity_score(candidate, selected_songs, diversity_weight=0.5):
    """
    Calculate diversity score for a candidate song
    Args:
        candidate: Feature vector of candidate song
        selected_songs: List of feature vectors of already selected songs
        diversity_weight: Weight for diversity (0-1), higher = more diverse
    Returns:
        Diversity score (higher is more diverse)
    """
    if len(selected_songs) == 0:
        return 1.0

    # Calculate similarity to all selected songs
    similarities = []
    for selected in selected_songs:
        sim = cosine_similarity([candidate], [selected])[0][0]
        similarities.append(sim)

    # Average similarity to selected songs
    avg_similarity = np.mean(similarities)

    # Diversity is inverse of similarity
    diversity = 1 - avg_similarity

    return diversity


def create_varied_playlist(all_songs_df, normalized_df, seed_songs, playlist_length=30, diversity=0.5):
    """
    Create a varied playlist that explores different musical spaces
    Args:
        all_songs_df: Complete song database
        normalized_df: Normalized features for all songs
        seed_songs: Songs user likes (DataFrame)
        playlist_length: Target playlist length
        diversity: 0 = very similar, 1 = very diverse
    Returns:
        DataFrame of recommended songs with variety
    """
    from .model import recommend_from_playlist

    logger.info(f"Creating varied playlist with diversity={diversity}")

    # Get initial candidates (2x playlist length)
    candidates_df = recommend_from_playlist(all_songs_df, normalized_df, seed_songs)
    candidates_df = candidates_df.head(playlist_length * 3)  # Get more candidates

    if len(candidates_df) == 0:
        logger.warning("No candidates found")
        return pd.DataFrame()

    # Get feature vectors
    candidate_features = normalized_df[normalized_df['id'].isin(candidates_df['id'].values)]

    selected_indices = []
    selected_features = []

    # Select diverse subset
    for i in range(min(playlist_length, len(candidates_df))):
        if i == 0:
            # First song: highest similarity
            best_idx = 0
        else:
            # Balance similarity and diversity
            scores = []

            for idx in range(len(candidates_df)):
                if idx in selected_indices:
                    scores.append(-1)  # Skip already selected
                    continue

                # Get candidate features
                candidate_id = candidates_df.iloc[idx]['id']
                candidate_feat = candidate_features[candidate_features['id'] == candidate_id]

                if candidate_feat.empty:
                    scores.append(-1)
                    continue

                candidate_feat_values = candidate_feat.drop('id', axis=1).values[0]

                # Calculate diversity from selected songs
                diversity_score = calculate_diversity_score(
                    candidate_feat_values,
                    selected_features,
                    diversity
                )

                # Combine similarity rank and diversity
                # Higher similarity rank = better (lower index)
                similarity_score = 1 - (idx / len(candidates_df))

                # Weighted combination
                final_score = (1 - diversity) * similarity_score + diversity * diversity_score

                scores.append(final_score)

            # Select best scoring song
            if max(scores) < 0:
                break

            best_idx = np.argmax(scores)

        # Add to selection
        selected_indices.append(best_idx)

        # Add features
        candidate_id = candidates_df.iloc[best_idx]['id']
        candidate_feat = candidate_features[candidate_features['id'] == candidate_id]

        if not candidate_feat.empty:
            candidate_feat_values = candidate_feat.drop('id', axis=1).values[0]
            selected_features.append(candidate_feat_values)

    # Return selected songs
    result = candidates_df.iloc[selected_indices].reset_index(drop=True)
    logger.info(f"Created playlist with {len(result)} songs")

    return result


def diversify_by_artist(recommendations_df, max_per_artist=3):
    """
    Limit number of songs per artist to increase variety
    Args:
        recommendations_df: DataFrame of recommendations
        max_per_artist: Maximum songs per artist
    Returns:
        Filtered DataFrame
    """
    if 'track_artist_name' not in recommendations_df.columns:
        return recommendations_df

    result = []
    artist_counts = {}

    for idx, row in recommendations_df.iterrows():
        artist = row['track_artist_name']

        if artist not in artist_counts:
            artist_counts[artist] = 0

        if artist_counts[artist] < max_per_artist:
            result.append(row)
            artist_counts[artist] += 1

    return pd.DataFrame(result)


def filter_by_mood_and_genre(all_songs_df, mood='energetic', genre='any'):
    """
    Filter songs by mood and genre
    Args:
        all_songs_df: DataFrame of all songs
        mood: 'energetic', 'calm', 'happy', 'melancholic'
        genre: Genre filter or 'any'
    Returns:
        Filtered DataFrame
    """
    df = all_songs_df.copy()

    # Filter by mood using audio features
    if mood == 'energetic':
        df = df[(df['track_energy'] > 0.6) & (df['track_tempo'] > 110)]
    elif mood == 'calm':
        df = df[(df['track_energy'] < 0.4) & (df['track_acousticness'] > 0.3)]
    elif mood == 'happy':
        df = df[df['track_valence'] > 0.6]
    elif mood == 'melancholic':
        df = df[(df['track_valence'] < 0.4) & (df['track_energy'] < 0.5)]

    # TODO: Add genre filtering if genre data available
    # For now, genre filtering is not implemented as YouTube doesn't provide genre data

    logger.info(f"Filtered to {len(df)} songs for mood={mood}")

    return df


def get_mood_from_features(audio_features):
    """
    Determine mood from audio features
    Args:
        audio_features: Dict or Series with audio features
    Returns:
        Mood string
    """
    valence = audio_features.get('track_valence', 0.5)
    energy = audio_features.get('track_energy', 0.5)
    tempo = audio_features.get('track_tempo', 120)

    if energy > 0.6 and tempo > 110:
        return 'energetic'
    elif valence < 0.4 and energy < 0.5:
        return 'melancholic'
    elif valence > 0.6:
        return 'happy'
    else:
        return 'calm'


def create_mood_based_playlist(all_songs_df, normalized_df, target_mood, playlist_length=30):
    """
    Create playlist based on target mood
    Args:
        all_songs_df: Complete song database
        normalized_df: Normalized features
        target_mood: Target mood string
        playlist_length: Length of playlist
    Returns:
        DataFrame of songs matching mood
    """
    # Filter by mood
    mood_songs = filter_by_mood_and_genre(all_songs_df, mood=target_mood)

    if len(mood_songs) == 0:
        logger.warning(f"No songs found for mood: {target_mood}")
        return pd.DataFrame()

    # Random diverse selection
    if len(mood_songs) > playlist_length:
        result = mood_songs.sample(n=playlist_length)
    else:
        result = mood_songs

    return result.reset_index(drop=True)
