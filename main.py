
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
top_tracks = track_labeler(spotifyObject.current_user_top_tracks(limit=50)['items'], 'user')


# Get top 50 US tracks
#categories = spotifyObject.categories()
#print([(cat['id'], cat['name']) for cat in categories['categories']['items']])
summer_playlist_info = spotifyObject.category_playlists(category_id='0JQ5DAqbMKFLVaM30PMBm4')['playlists']['items'][0]
summer_playlist_id = summer_playlist_info['id']
summer_playlist_tracks = track_labeler([it['track'] for it in spotifyObject.playlist(summer_playlist_id)['tracks']['items']], 'non_user')

total_tracks = top_tracks + summer_playlist_tracks


# Get track features
track_features = user_track_feature_extraction(total_tracks, spotifyObject)
playlist_preprocessing(track_features, 'track_features')

print("Finished")
