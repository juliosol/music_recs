"""
Feature engineering for YouTube music data
Adapted from Spotify feature engineering
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def drop_duplicates_df_youtube(dataframe):
    """
    Drop duplicate videos based on video ID and track info
    Keep highest view count version
    Args:
        dataframe: pandas DataFrame
    Returns:
        DataFrame without duplicates
    """
    # First, keep unique video IDs
    dataframe = dataframe.drop_duplicates(subset=['id'], keep='first')

    # Then, for same song/artist combo, keep highest views
    dataframe['song_info'] = dataframe.apply(
        lambda row: f"{row['track_name']} {row['track_artist_name']}".lower(),
        axis=1
    )

    dataframe = dataframe.sort_values('views', ascending=False)
    dataframe = dataframe.drop_duplicates('song_info', keep='first')

    return dataframe


def get_polarity(text):
    """
    Get sentiment polarity of text
    Args:
        text: String to analyze
    Returns:
        Float polarity score [-1.0, 1.0]
    """
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0


def get_subjectivity(text):
    """
    Get sentiment subjectivity of text
    Args:
        text: String to analyze
    Returns:
        Float subjectivity score [0.0, 1.0]
    """
    try:
        return TextBlob(str(text)).sentiment.subjectivity
    except:
        return 0.5


def categorize_analysis(score, case="polarity"):
    """
    Categorize sentiment scores
    Args:
        score: Numeric score
        case: 'polarity' or 'subjectivity'
    Returns:
        Category string
    """
    if case == "polarity":
        if score < 0:
            return "negative"
        elif score == 0:
            return "neutral"
        else:
            return "positive"
    else:
        if score < 1/3:
            return "low"
        elif score >= 1/3 and score < 2/3:
            return "medium"
        else:
            return "high"


def sentiment_analysis(dataframe, column_name='song_info'):
    """
    Perform sentiment analysis on text columns
    Args:
        dataframe: pandas DataFrame
        column_name: Column to analyze
    Returns:
        DataFrame with sentiment features
    """
    dataframe['track_name_polarity'] = dataframe[column_name].apply(get_polarity).apply(
        lambda x: categorize_analysis(x, "polarity")
    )
    dataframe['track_name_subjectivity'] = dataframe[column_name].apply(get_subjectivity).apply(
        lambda x: categorize_analysis(x, "subjectivity")
    )

    return dataframe


def datetime_converter(dataframe):
    """
    Convert datetime columns to numeric format
    Args:
        dataframe: pandas DataFrame
    Returns:
        DataFrame with converted dates
    """
    try:
        # Convert YouTube published_at to timestamp
        dataframe['track_album_release_date'] = pd.to_datetime(
            dataframe['track_album_release_date'],
            errors='coerce'
        )
        dataframe['track_album_release_date'] = dataframe['track_album_release_date'].astype(int) / 10**9

        # Handle NaN values
        dataframe['track_album_release_date'] = dataframe['track_album_release_date'].fillna(0)

    except Exception as e:
        logger.error(f"Error converting dates: {e}")
        dataframe['track_album_release_date'] = 0

    # Track explicit is already False for all YouTube videos
    if 'track_explicit' in dataframe.columns:
        dataframe['track_explicit'] = dataframe['track_explicit'].replace({True: 1, False: 0})

    return dataframe


def normalize_engagement_metrics(dataframe):
    """
    Normalize views and likes to 0-1 scale
    Args:
        dataframe: pandas DataFrame
    Returns:
        DataFrame with normalized metrics
    """
    scaler = MinMaxScaler()

    if 'views' in dataframe.columns and len(dataframe) > 0:
        views_reshaped = dataframe[['views']].values.reshape(-1, 1)
        dataframe['normalized_views'] = scaler.fit_transform(views_reshaped)
    else:
        dataframe['normalized_views'] = 0.5

    if 'likes' in dataframe.columns and len(dataframe) > 0:
        likes_reshaped = dataframe[['likes']].values.reshape(-1, 1)
        dataframe['normalized_likes'] = scaler.fit_transform(likes_reshaped)
    else:
        dataframe['normalized_likes'] = 0.5

    return dataframe


def feature_normalizer(dataframe):
    """
    Normalize audio features for recommendation algorithm
    Args:
        dataframe: pandas DataFrame
    Returns:
        DataFrame with normalized features
    """
    # Columns to drop for normalized feature set
    drop_cols = ['track_name', 'track_artist_name', 'song_info',
                 'track_name_polarity', 'track_name_subjectivity',
                 'youtube_url', 'artist_genres', 'track_album_type',
                 'track_album_name', 'track_explicit', 'views', 'likes']

    # Keep only columns that exist
    drop_cols = [col for col in drop_cols if col in dataframe.columns]

    cleaned_df = dataframe.drop(drop_cols, axis=1, errors='ignore').reset_index(drop=True)

    # Normalize popularity scores (already 0-100)
    if 'artist_popularity' in cleaned_df.columns and 'track_popularity' in cleaned_df.columns:
        pop = cleaned_df[["artist_popularity", "track_popularity"]].reset_index(drop=True)
        scaler = MinMaxScaler()
        pop_scaled = pd.DataFrame(scaler.fit_transform(pop), columns=pop.columns) * 0.2
        cleaned_df = cleaned_df.drop(['artist_popularity', 'track_popularity'], axis=1)
    else:
        pop_scaled = pd.DataFrame()

    # Normalize audio features (float columns)
    float_cols = cleaned_df.dtypes[cleaned_df.dtypes == 'float64'].index.values

    if len(float_cols) > 0:
        floats = cleaned_df[float_cols].reset_index(drop=True)
        scaler = MinMaxScaler()
        floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns=floats.columns) * 0.2
        cleaned_df = cleaned_df.drop(list(floats.columns), axis=1)
    else:
        floats_scaled = pd.DataFrame()

    # Combine all normalized features
    result_parts = [cleaned_df]
    if not pop_scaled.empty:
        result_parts.append(pop_scaled)
    if not floats_scaled.empty:
        result_parts.append(floats_scaled)

    return pd.concat(result_parts, axis=1)


def playlist_preprocessing_youtube(dataframe, df_name='youtube_music'):
    """
    Complete preprocessing pipeline for YouTube music data
    Args:
        dataframe: pandas DataFrame with raw YouTube track data
        df_name: Name for saving CSV files
    Returns:
        Tuple of (processed_df, normalized_df)
    """
    logger.info(f"Starting preprocessing for {len(dataframe)} tracks...")

    # Drop duplicates
    dataframe = drop_duplicates_df_youtube(dataframe)
    logger.info(f"After deduplication: {len(dataframe)} tracks")

    # Sentiment analysis
    dataframe = sentiment_analysis(dataframe, 'song_info')

    # Convert dates
    dataframe = datetime_converter(dataframe)

    # Normalize engagement metrics
    dataframe = normalize_engagement_metrics(dataframe)

    # Fill NaN values
    dataframe = dataframe.fillna(0)

    # Save processed data
    try:
        dataframe.to_csv(f'data_extraction/{df_name}.csv', index=False)
        logger.info(f"Saved processed data to data_extraction/{df_name}.csv")
    except Exception as e:
        logger.warning(f"Could not save processed data: {e}")

    # Create normalized feature set
    normalized_dataframe = feature_normalizer(dataframe.copy())
    normalized_dataframe = normalized_dataframe.fillna(0)

    # Save normalized data
    try:
        normalized_dataframe.to_csv(f'data_extraction/normalized_{df_name}.csv', index=False)
        logger.info(f"Saved normalized data to data_extraction/normalized_{df_name}.csv")
    except Exception as e:
        logger.warning(f"Could not save normalized data: {e}")

    logger.info("Preprocessing complete!")

    return dataframe, normalized_dataframe
