
import os
import sys
import json
import spotipy
import spotipy.util as util
from json.decoder import JSONDecodeError
import pandas as pd
from tqdm import tqdm
from data_extraction.extraction_fns import track_feature_extraction



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

track_df = track_feature_extraction(top_tracks, spotifyObject)
track_df.to_csv('data_extraction/track_dataframe.csv', index=False)
print(track_df.head())

#audio_features = spotifyObject.audio_features(top_tracks[0]['id'])[0]
#print(audio_features_names)
#print(audio_features)

