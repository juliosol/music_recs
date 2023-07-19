
import os
import sys
import json
import spotipy
import spotipy.util as util
from json.decoder import JSONDecodeError
import pandas as pd
from tqdm import tqdm
from data_extraction.extraction_fns import user_track_feature_extraction
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
top_tracks = spotifyObject.current_user_top_tracks(limit=50)['items']
us_top_playlist = spotifyObject

# Get user top track playlist
user_track_features = user_track_feature_extraction(top_tracks, spotifyObject)
playlist_preprocessing(user_track_features)

# Get top 50 US tracks
top_us_tracks = spotifyObject.featured_playlists(country='US', limit=50)
import pdb
pdb.set_trace()
print("Finished")

# Get top today tracks
