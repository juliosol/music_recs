from tqdm import tqdm
import pandas as pd

audio_features_names = ['danceability', 'energy', 'key', 'loudness', 'mode',
                        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                        'valence', 'tempo']
## TODO: Add audio analysis data
## TODO: Add lyrics analysis data
## audio_analysis = spotifyObject.audio_analysis(first_track)

def track_labeler(song_track, label_string):
    """
    Function used to label a list of tracks
    Inputs:
    - list of track_lists: list of tracks (tracks should be dictionary objects)
    - label_string: string to label the tracks (i.e. user, summer, top_50, etc.)
    Return:
    - Returns list of track objects with labels
    """
    for tr in song_track:
        tr['label'] = label_string
    return song_track

def user_track_feature_extraction(track_list, spotifyObject, type='user'):
    """
    Function used to extract track features and analysis from a list of tracks
    from spotify.
    Inputs:
    - list of track_lists: page of tracks
    - spotifyObject: spotify object initialized with user credentials to get audio features
    Return:
    - Returns dataframe with features 
    """
    user_top_tracks_data = []

    for tr in tqdm(track_list):
        if (type != 'user'):
            tr = tr['track']
        #import pdb
        #pdb.set_trace()
        curr_track_dict = {}
        track_artist = tr['artists'][0]
        track_album = tr['album']
        curr_track_dict['id'] = tr['id']
        curr_track_dict['track_name'] = tr['name']
        curr_track_dict['track_artist_name'] = track_artist['name']
        artist_info = spotifyObject.artist(tr['artists'][0]['id'])
        curr_track_dict['artist_genres'] = artist_info['genres']
        curr_track_dict['artist_popularity'] = artist_info['popularity']
        curr_track_dict['track_album_type'] = track_album['album_type']
        curr_track_dict['track_album_name'] = track_album['name']
        curr_track_dict['track_album_release_date'] = track_album['release_date']
        curr_track_dict['track_duration_ms'] = tr['duration_ms']
        curr_track_dict['track_explicit'] = tr['explicit']
        curr_track_dict['track_popularity'] = tr['popularity']
        audio_features = spotifyObject.audio_features(tr['id'])[0]
        for feat_name in audio_features_names:
            curr_track_dict['track_'+feat_name] = audio_features[feat_name]
        #curr_track_dict['label'] = tr['label']
        user_top_tracks_data.append(curr_track_dict)
    
    return pd.DataFrame.from_dict(user_top_tracks_data)