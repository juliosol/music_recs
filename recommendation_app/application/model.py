import csv
import pandas as pd
import sklearn.metrics.pairwise as cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

def generate_seen_playlists(completeFeaturesDataDF, userDF):
    """
    Function that will take the complete data set with features and user list dataframe.
    Returns:
    - vector of sum of features of user playlist used for finding similar songs
    - df of features not seen by user in personal playlist
    """
    user_seen_songs = completeFeaturesDataDF[completeFeaturesDataDF['id'].isin(userDF['id'].values)]
    not_user_seen_songs = completeFeaturesDataDF[~completeFeaturesDataDF['id'].isin(userDF['id'].values)]
    user_seen_songs_non_id = user_seen_songs.drop(columns = 'id')
    return user_seen_songs_non_id.sum(axis=0), not_user_seen_songs

def generate_recommendations(completeDataDF, user_sum_features, playlist_non_user_seen_songs):
    non_user_completeDataDF = completeDataDF[completeDataDF['id'].isin(playlist_non_user_seen_songs['id'].values)]
    #import pdb
    #pdb.set_trace()
    # Find cosine similarity between user feature vector and non-seen feature vectors
    non_user_completeDataDF['sim'] = cosine_similarity(playlist_non_user_seen_songs.drop('id', axis = 1).values, user_sum_features.values.reshape(1,-1))[:,0]
    non_user_completeDataDF_top50 = non_user_completeDataDF.sort_values('sim', ascending=False).head(50)
    return non_user_completeDataDF_top50

def recommend_from_playlist(completeDataDF, completeFeaturesDataDF, userDF):
    #print(completeDataDF.head())
    #print(completeDataDF.columns)
    #print(completeFeaturesDataDF.head())
    #print(completeFeaturesDataDF.columns)
    #print(userDF.head())
    #print(userDF.columns)
    user_sum_features, playlist_non_user_seen_songs = generate_seen_playlists(completeFeaturesDataDF, userDF)
    #print(playlist_non_user_seen_songs.head())
    #print(playlist_non_user_seen_songs.columns)
    #import pdb
    #pdb.set_trace()
    top_50_completeDataDF = generate_recommendations(completeDataDF, user_sum_features, playlist_non_user_seen_songs)
    return top_50_completeDataDF
