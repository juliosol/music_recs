from application import app
from flask import Flask, render_template, request
from application.features import *
from application.model import *

allSongDF = pd.read_csv("./data/playlist_track_features.csv")
allSongFeatureSetDF = pd.read_csv("./data/normalized_playlist_track_features.csv")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    # Rquest URL from the HTML form
    URL = request.form['SpotUserName']
    # use the extract function to get features dataframe
    df = extract(URL)
    # retrieve the results and get as many recommendations as the user requested
    edm_recs = recommend_from_playlist(allSongDF, allSongFeatureSetDF, df)
    #import pdb
    #pdb.set_trace()
    number_of_recs = int(request.form['number-of-recs'])
    my_songs = []
    for i in range(number_of_recs):
      my_songs.append([str(edm_recs.iloc[i,1]) + ' - '+ '"'+str(edm_recs.iloc[i,2])+'"', "https://open.spotify.com/track/"+ str(edm_recs.iloc[i,0])])
    return render_template('results.html',songs= my_songs)