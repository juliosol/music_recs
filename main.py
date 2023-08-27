
import os
import sys
import json
import spotipy
import spotipy.util as util
from json.decoder import JSONDecodeError
import pandas as pd
from tqdm import tqdm
from data_extraction.extraction_fns import user_track_feature_extraction, track_labeler
from data_extraction.feature_eng import playlist_preprocessing



# Importing credentials
with open('credentials.json') as f:
	credentials = json.load(f)

# Get the username from terminal
username = sys.argv[1]
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
    token = util.prompt_for_user_token(username, 
                                       scope,
                                       client_id=credentials['client_id'],
                                       client_secret=credentials['client_secret'],
                                       redirect_uri=credentials['redirect_uri']) # add scope

# Create our spotify object with permissions
spotifyObject = spotipy.Spotify(auth=token)

# Get current device
devices = spotifyObject.devices()
deviceID = devices['devices'][0]['id']
print(deviceID)

# Current saved songs information
user_top_tracks = spotifyObject.current_user_top_tracks(limit=50)['items'] #track_labeler(spotifyObject.current_user_top_tracks(limit=50)['items'], 'user')
print(f"THis is the type of user_top_tracks {type(user_top_tracks)}")

# Get track features
track_features = user_track_feature_extraction(user_top_tracks, spotifyObject)
user_track_df, normalized_user_track_df = playlist_preprocessing(track_features, 'track_features')


# Get top 50 US tracks
categories = spotifyObject.categories()
top_list_names = ['Today\'s Top Hits ', 'Rock This', 'Viva Latino', 'Top 50 - USA', 'Top 50 - Global', 'Viral 50 - Global', 'Viral 50 - USA', 'New Music Friday']


total_playlist_tracks = []
playlist_info = spotifyObject.category_playlists(category_id='toplists')['playlists']['items']
for playlist in playlist_info:
    if playlist['name'] in top_list_names:
        print(f"Adding playlist of name {playlist['name']}")
        currPlaylist = spotifyObject.playlist(playlist['id'])['tracks']['items']
        print(f"This is type of currPlaylist {type(currPlaylist)}")
        total_playlist_tracks += currPlaylist

# Get playlist tracks
total_playlist_features = user_track_feature_extraction(total_playlist_tracks, spotifyObject, type='playlists')
import pdb
pdb.set_trace()
playlist_df, normalized_playlist_df = playlist_preprocessing(total_playlist_features, 'playlist_track_features')

print("Finished")
