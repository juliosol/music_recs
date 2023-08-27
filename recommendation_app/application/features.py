import spotipy
import spotipy.util as util
import os
#from spotipy.oauth2 import SpotifyClientCredentials
import json
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler



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

def user_track_feature_extraction(track_list, spotifyObject):
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

    for tr in track_list:
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
        curr_track_dict['label'] = tr['label']
        user_top_tracks_data.append(curr_track_dict)
    
    return pd.DataFrame.from_dict(user_top_tracks_data)




def drop_duplicates_df(dataframe):
    '''
    Drop duplicates of columns in dataframes
    '''
    dataframe['song_info'] = dataframe.apply(lambda row: row['track_name'] + " " + row['track_artist_name'] + " " + row['track_album_type'] + " " + row['track_album_name'], axis=1)
    dataframe = dataframe.drop_duplicates('song_info')
    dataframe = dataframe.drop(['track_album_type', 'track_album_name'], axis=1)
    return dataframe

def get_polarity(text):
    """
    Function that will get the polarity of a piece of text.
    Input:
    - text: string to get polarity from
    Output:
    - float polarity: Polarity is float within the range [-1.0, 1.0]. -1.0 is very negative, 1.0 very positive. subjectivity is float with range [0.0, 1.0]
    """
    text_sentiment = TextBlob(text).sentiment
    return text_sentiment.polarity

def get_subjectivity(text):
    """
    Function that will get the polarity of a piece of text.
    Input:
    - text: string to get polarity from
    Output:
    -  float score: subjectivity is float with range [0.0, 1.0]
    """
    text_sentiment = TextBlob(text).sentiment
    return text_sentiment.subjectivity

def categorize_analysis(score, case="polarity"):
    """
    Categorize the results of polarity and subjectivity.
    Input:
    - Int score: integer of score previous transformations
    - string case: type of case the score represents
    Output:
    - String categorizing the score.
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

def sentiment_analysis(dataframe, column_name):
    """
    Perform sentiment analysis on text columns (track name)
    Input:
    - pandas dataframe dataframe: where we will be taking columns to perform analysis
    - string column_name: Column which we will take to perform analysis.
    """
    dataframe['track_name_subjectivity'] = dataframe[column_name].apply(get_polarity).apply(lambda x: categorize_analysis(x, "polarity"))
    dataframe['track_name_polarity'] = dataframe[column_name].apply(get_subjectivity).apply(lambda x: categorize_analysis(x, "subjectivity"))
    dataframe = dataframe.drop([column_name], axis=1)
    return dataframe


def one_hot_encoder(dataframe, col_name='subjectivity'):
    """
    Sklearn one hot encoder for polarity and subjectivity
    """
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    if col_name == 'subjectivity':
        col = 'track_name_subjectivity'
    elif col_name == 'polarity':
        col = 'track_name_polarity'
    elif col_name == 'mode':
        col = 'track_mode'
    elif col_name == 'key':
        col = 'track_key'
    df_subj_pol = dataframe[[col]]
    enc.fit(df_subj_pol)
    cat_lists = enc.categories_
    new_cols = [col + '|' + str(i) for i in cat_lists[0]]
    df_transformed = pd.DataFrame(enc.transform(df_subj_pol), columns=new_cols)
    return pd.concat([dataframe, df_transformed], axis=1)

def genres_tf_idf(dataframe):
    """
    Function used to transform the genres into tf-idf matrix.
    Input:
    - dataframe: pandas dataframe of features
    Output:
    - dataframe
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataframe['artist_genres'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in vectorizer.get_feature_names_out()]
    genre_df.reset_index(drop=True, inplace=True)
    genre_df.iloc[0]
    return pd.concat([dataframe, genre_df], axis=1)
    
def datetime_converter(dataframe):
    """
    Function used to convert the datetime columns into datetime format.
    Input:
    - dataframe: pandas dataframe of features
    """
    dataframe['track_album_release_date'] = pd.to_datetime(dataframe['track_album_release_date'])#, format='mixed')
    dataframe['track_album_release_date'] = dataframe['track_album_release_date'].astype(int) / 10**9
    dataframe['track_explicit'] = dataframe['track_explicit'].replace({True: 1, False: 0})
    return dataframe

def feature_normalizer(dataframe):
    """
    Function used to normalize the audio features.
    Input:
    - dataframe: pandas dataframe of features
    """
    cleaned_df = dataframe.drop(['track_name', 'track_artist_name','artist_genres', 'track_mode', 'track_key', 'track_name_subjectivity', 'track_name_polarity'], axis=1).reset_index(drop = True)
    pop = cleaned_df[["artist_popularity","track_popularity"]].reset_index(drop = True)
    scaler = MinMaxScaler()
    pop_scaled = pd.DataFrame(scaler.fit_transform(pop), columns = pop.columns) * 0.2 

    # Scale audio columns
    float_cols = cleaned_df.dtypes[cleaned_df.dtypes == 'float64'].index.values
    floats = cleaned_df[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

    cleaned_df = cleaned_df.drop(['artist_popularity', 'track_popularity'], axis=1)
    cleaned_df = cleaned_df.drop(list(floats.columns), axis=1)
    
    return pd.concat([cleaned_df, pop_scaled, floats_scaled], axis=1)

def playlist_preprocessing(dataframe, df_name=''):
    """
    Function doing all the preprocessing of the dataframe
    Input: 
    - df_name: name of the dataframe that will be saved
    """
    dataframe = drop_duplicates_df(dataframe)
    dataframe = sentiment_analysis(dataframe, 'song_info')
    dataframe = datetime_converter(dataframe)

    #dataframe = one_hot_encoder(dataframe)
    #dataframe = one_hot_encoder(dataframe, 'polarity')
    #dataframe = one_hot_encoder(dataframe, 'key')
    #dataframe = one_hot_encoder(dataframe, 'mode')

    #dataframe = genres_tf_idf(dataframe)
    #dataframe.to_csv('data_extraction/' + df_name + '.csv', index=False)

    normalized_dataframe = feature_normalizer(dataframe)
    normalized_dataframe.fillna(0)

    return normalized_dataframe

def extract(SpotUserName):
    print(os.getcwd())
    print(os.listdir())
    # Importing credentials and logging into Spotify
    #with open('./credentials.json', 'r') as f:
    #    credentials = json.load(f)
    
    credentials = {
        "client_id": "6b98b7391cf54daeaa9099707df3c8d2",
        "client_secret": "f91fa17c60424fe9af925e3f3814f9cc",
        "redirect_uri": "http://localhost:3000/callback"
    }


    username = SpotUserName
    scope = 'user-library-read user-read-playback-state user-modify-playback-state user-top-read'

    # Erase cache and prompt for user permission
    try:
        token = util.prompt_for_user_token(username, 
                                        scope, 
                                        client_id=credentials['client_id'],
                                        client_secret=credentials['client_secret'],
                                        redirect_uri=credentials['redirect_uri']) # add scope
    except (AttributeError, JSONDecodeError):
        print("Entering except")
        os.remove(f".cache-{username}")

    # Create spotify object with permisssions
    spotifyObject = spotipy.Spotify(auth=token)

    # Current saved songs information
    top_tracks = track_labeler(spotifyObject.current_user_top_tracks(limit=50)['items'], 'user')

    total_tracks = top_tracks #+ summer_playlist_tracks

    # Get track features
    track_features = user_track_feature_extraction(total_tracks, spotifyObject)
    track_features = playlist_preprocessing(track_features)
    
    return track_features



