from textblob import TextBlob
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def drop_duplicates_df(dataframe):
    '''
    Drop duplicates of columns in dataframes
    '''
    dataframe['song_info'] = dataframe.apply(lambda row: row['track_name'] + " " + row['track_artist_name'] + " " + row['track_album_type'] + " " + row['track_album_name'], axis=1)
    dataframe = dataframe.drop_duplicates('song_info')
    dataframe = dataframe.drop(['track_name', 'track_artist_name', 'track_album_type', 'track_album_name'], axis=1)
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
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataframe['artist_genres'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in vectorizer.get_feature_names_out()]
    #genre_df.drop(columns='genre|unknown')
    genre_df.reset_index(drop=True, inplace=True)
    genre_df.iloc[0]
    return pd.concat([dataframe, genre_df], axis=1)
    
def datetime_converter(dataframe):
    dataframe['track_album_release_date'] = pd.to_datetime(dataframe['track_album_release_date'], format='mixed')
    dataframe['track_album_release_date'] = dataframe['track_album_release_date'].astype(int) / 10**9
    dataframe['track_explicit'] = dataframe['track_explicit'].replace({True: 1, False: 0})
    return dataframe

def feature_normalizer(dataframe):
    cleaned_df = dataframe.drop(['artist_genres', 'track_mode', 'track_key', 'track_name_subjectivity', 'track_name_polarity', 'song_info'], axis=1)
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
    print(cleaned_df.columns)
    print(cleaned_df.dtypes)

    return pd.concat([cleaned_df, pop_scaled, floats_scaled], axis=1)

def playlist_preprocessing(dataframe):
    dataframe = drop_duplicates_df(dataframe)
    dataframe = sentiment_analysis(dataframe, 'song_info')
    dataframe = datetime_converter(dataframe)

    dataframe = one_hot_encoder(dataframe)
    dataframe = one_hot_encoder(dataframe, 'polarity')
    dataframe = one_hot_encoder(dataframe, 'key')
    dataframe = one_hot_encoder(dataframe, 'mode')

    dataframe = genres_tf_idf(dataframe)
    dataframe.to_csv('data_extraction/user_track_features.csv', index=False)

    normalized_dataframe = feature_normalizer(dataframe)
    normalized_dataframe.to_csv('data_extraction/normalized_user_track_features.csv', index=False)
