# Music Recommendation System

A personalized music recommendation system that uses the Spotify API to analyze your listening preferences and recommend new songs based on audio features and similarity metrics.

## Features

- Extract audio features from your top Spotify tracks
- Analyze popular playlists (Top 50, Viral 50, etc.)
- Generate personalized music recommendations using similarity algorithms
- Web interface for easy interaction and playlist-based recommendations

## Project Structure

```
music_recs/
├── data_extraction/          # Data extraction and feature engineering modules
│   ├── extraction_fns.py    # Functions to extract track features from Spotify
│   └── feature_eng.py       # Feature preprocessing and normalization
├── recommendation_app/       # Flask web application
│   ├── application/         # Application modules
│   │   ├── features.py     # Feature extraction for web app
│   │   ├── model.py        # Recommendation model
│   │   └── routes.py       # Flask routes
│   └── start.py            # Application entry point
├── dataset/                 # Dataset utilities
├── main.py                 # Main script for data extraction
├── credentials.json        # Spotify API credentials (DO NOT COMMIT)
└── requirements.txt        # Project dependencies
```

## Prerequisites

- Python 3.7+
- Spotify account
- Spotify Developer credentials (Client ID, Client Secret)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd music_recs
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- spotipy (Spotify API wrapper)
- pandas, numpy (Data manipulation)
- scikit-learn (Machine learning)
- flask (Web application)
- tqdm (Progress bars)

### 3. Set Up Spotify API Credentials

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new application
3. Note your **Client ID** and **Client Secret**
4. Add `http://localhost:8888/callback` as a Redirect URI in your app settings

5. Create a `credentials.json` file in the project root:

```json
{
  "client_id": "your_client_id_here",
  "client_secret": "your_client_secret_here",
  "redirect_uri": "http://localhost:8888/callback"
}
```

**Important:** Never commit `credentials.json` to version control!

## Running the Project

### Option 1: Data Extraction Script

Extract audio features from your Spotify library and popular playlists:

```bash
python main.py
```

When prompted, enter your Spotify username. The first time you run this, you'll be redirected to authorize the application.

**Note:** The script contains debug breakpoints (`pdb.set_trace()`). You may need to remove or comment these out in:
- `main.py:51`
- `data_extraction/extraction_fns.py:55`

### Option 2: Web Application

Run the Flask web application for interactive recommendations:

```bash
cd recommendation_app
python start.py
```

The app will be available at `http://localhost:5000`

**Features:**
- Enter a Spotify playlist URL
- Specify number of recommendations
- Get personalized song recommendations based on the playlist

### Data Flow

1. **Data Extraction**: `main.py` authenticates with Spotify and extracts:
   - User's top 10 tracks with audio features
   - Tracks from popular playlists (Top Hits, Rock, Latino, etc.)

2. **Feature Engineering**: Audio features are normalized:
   - Danceability, energy, valence, tempo
   - Acousticness, instrumentalness, speechiness
   - Key, mode, loudness, liveness

3. **Recommendation**: Similarity metrics are used to find songs similar to user preferences

## Audio Features Used

The system analyzes the following Spotify audio features:
- **Danceability**: How suitable a track is for dancing
- **Energy**: Intensity and activity measure
- **Key**: Musical key of the track
- **Loudness**: Overall loudness in decibels
- **Mode**: Major or minor modality
- **Speechiness**: Presence of spoken words
- **Acousticness**: Confidence measure of acoustic sound
- **Instrumentalness**: Predicts if track contains no vocals
- **Liveness**: Detects presence of an audience
- **Valence**: Musical positiveness
- **Tempo**: Overall estimated tempo in BPM

## Troubleshooting

### Authentication Issues
- Delete cache files (`.cache-<username>`) and re-authenticate
- Verify credentials in `credentials.json`
- Ensure redirect URI matches in Spotify Developer Dashboard

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.7+)

### Web App Data Not Found
- Run `main.py` first to generate the required CSV files
- Ensure CSV files are in the correct location expected by `routes.py`

## Development Notes

- The project uses the Spotipy library for Spotify API interactions
- Authentication uses OAuth 2.0 with user authorization
- Recommendation algorithm is based on cosine similarity of normalized features

## Contributing

This is a personal project. Feel free to fork and modify for your own use!

## License

This project is for educational and personal use.

