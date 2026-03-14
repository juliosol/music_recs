# Migration Plan: Spotify to YouTube Music Recommender

## Project Goal
Transform the existing Spotify-based music recommendation system into a YouTube-based system that can:
1. Create varied playlists from YouTube music videos
2. Discover new music without requiring a Spotify account
3. Generate recommendations as YouTube video links instead of Spotify URIs

---

## Current System Analysis

### What Currently Works (Spotify-based)
- ✅ Extracts audio features from Spotify tracks (danceability, energy, tempo, etc.)
- ✅ Uses cosine similarity to find similar songs
- ✅ Preprocessing pipeline with sentiment analysis and feature normalization
- ✅ Flask web app for user interaction
- ✅ Recommendation algorithm based on user's listening history

### Key Challenges for YouTube Migration
- ❌ YouTube doesn't provide audio features like Spotify does
- ❌ Need alternative methods to analyze music without audio feature API
- ❌ Must handle video search and identification
- ❌ Need to extract metadata from YouTube videos

---

## Migration Strategy Overview

### Phase 1: Research & Setup
Investigate alternative audio analysis methods and set up YouTube API access.

### Phase 2: Data Layer Replacement
Replace Spotify API calls with YouTube API and audio analysis tools.

### Phase 3: Feature Engineering Adaptation
Adapt feature extraction to work with YouTube videos and alternative audio sources.

### Phase 4: Recommendation Engine Update
Ensure recommendation algorithm works with new data sources.

### Phase 5: UI/UX Updates
Update Flask app to work with YouTube links instead of Spotify URIs.

### Phase 6: Testing & Refinement
Test the full pipeline and refine recommendations.

---

## Detailed Step-by-Step Implementation Plan

## PHASE 1: Research & API Setup

### Task 1.1: Set Up YouTube Data API v3
**Estimated Time:** 30 minutes

**Steps:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable YouTube Data API v3
4. Create credentials (API Key)
5. Save API key to `credentials.json` (add `youtube_api_key` field)

**Deliverables:**
- YouTube API key in `credentials.json`
- Test script to verify API access

**Resources:**
- [YouTube Data API Documentation](https://developers.google.com/youtube/v3)
- [Google API Python Client](https://github.com/googleapis/google-api-python-client)

---

### Task 1.2: Research Audio Analysis Libraries
**Estimated Time:** 2-3 hours

**Options to Evaluate:**

#### Option A: Essentia (Recommended)
- **Pros:** Open-source, extracts features similar to Spotify (tempo, key, danceability)
- **Cons:** Requires audio file download, computationally intensive
- **Installation:** `pip install essentia-tensorflow`

#### Option B: LibROSA
- **Pros:** Python native, good for audio feature extraction
- **Cons:** Lower-level, requires more manual feature engineering
- **Installation:** `pip install librosa`

#### Option C: YouTube-DL + Essentia
- **Pros:** Can download audio directly from YouTube, then analyze
- **Cons:** May violate YouTube ToS if done at scale
- **Installation:** `pip install yt-dlp essentia-tensorflow`

#### Option D: Pre-built Dataset (Million Song Dataset)
- **Pros:** No need to analyze audio, features already computed
- **Cons:** Limited to songs in dataset, may not have new music
- **Resource:** [Million Song Dataset](http://millionsongdataset.com/)

**Decision Point:**
Choose based on:
- Scale: How many songs will you analyze?
- Freshness: Do you need latest releases?
- Computation: Do you have resources to process audio?

**Recommended Approach:**
Hybrid - Use pre-built datasets where available, audio analysis for new songs.

**Deliverables:**
- Document chosen approach with justification
- Install required libraries
- Create test script to extract features from one song

---

### Task 1.3: Install Required Dependencies
**Estimated Time:** 30 minutes

**Steps:**
1. Update `requirements.txt` with new dependencies:
```txt
# Existing dependencies
pandas
numpy
scikit-learn
seaborn
matplotlib
tqdm
flask

# YouTube dependencies
google-api-python-client
google-auth-oauthlib
google-auth-httplib2

# Audio analysis (choose based on Task 1.2)
essentia-tensorflow  # Option A
# OR
librosa  # Option B
soundfile

# Audio download (if needed)
yt-dlp

# Additional utilities
youtube-search-python
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Deliverables:**
- Updated `requirements.txt`
- All dependencies installed successfully

---

## PHASE 2: YouTube Data Extraction

### Task 2.1: Create YouTube API Wrapper
**Estimated Time:** 2-3 hours

**Create:** `youtube_extraction/youtube_api.py`

**Required Functions:**
```python
def search_video(query, max_results=10):
    """
    Search YouTube for music videos
    Args:
        query: Artist + Song name (e.g., "The Beatles Yesterday")
        max_results: Number of results to return
    Returns:
        List of video IDs and metadata
    """
    pass

def get_video_details(video_id):
    """
    Get detailed information about a video
    Args:
        video_id: YouTube video ID
    Returns:
        Dict with: title, channel, duration, views, likes, description
    """
    pass

def get_playlist_videos(playlist_id):
    """
    Get all videos from a YouTube playlist
    Args:
        playlist_id: YouTube playlist ID
    Returns:
        List of video IDs
    """
    pass

def extract_audio_url(video_id):
    """
    Get audio stream URL for a video
    Args:
        video_id: YouTube video ID
    Returns:
        Audio stream URL
    """
    pass
```

**Testing:**
Create test script `test_youtube_api.py`:
```python
from youtube_extraction.youtube_api import search_video, get_video_details

# Test search
results = search_video("Radiohead Creep")
print(f"Found {len(results)} videos")

# Test details
if results:
    details = get_video_details(results[0]['id'])
    print(f"Title: {details['title']}")
    print(f"Duration: {details['duration']}")
```

**Deliverables:**
- `youtube_extraction/youtube_api.py` with all functions
- `test_youtube_api.py` with passing tests
- Documentation of API rate limits

---

### Task 2.2: Create Audio Feature Extractor
**Estimated Time:** 4-6 hours

**Create:** `youtube_extraction/audio_features.py`

**Approach depends on Task 1.2 decision. Example for Essentia:**

```python
import essentia.standard as es

def extract_audio_features(audio_file_path):
    """
    Extract Spotify-like features from audio file
    Args:
        audio_file_path: Path to audio file (mp3, wav, etc.)
    Returns:
        Dict with features matching Spotify schema:
        {
            'danceability': float,
            'energy': float,
            'key': int,
            'loudness': float,
            'mode': int,
            'speechiness': float,
            'acousticness': float,
            'instrumentalness': float,
            'liveness': float,
            'valence': float,
            'tempo': float,
            'duration_ms': int
        }
    """
    pass

def download_and_analyze(video_id):
    """
    Download audio from YouTube video and analyze
    Args:
        video_id: YouTube video ID
    Returns:
        Feature dict from extract_audio_features()
    """
    pass

def get_features_from_cache(video_id):
    """
    Check if features already computed and cached
    Args:
        video_id: YouTube video ID
    Returns:
        Feature dict or None if not cached
    """
    pass
```

**Feature Mapping Guide (Essentia → Spotify equivalent):**

| Spotify Feature | Essentia Algorithm | Notes |
|----------------|-------------------|-------|
| danceability | `Danceability` | Direct equivalent |
| energy | `Energy` | Direct equivalent |
| key | `KeyExtractor` | 0-11 for C, C#, D, etc. |
| loudness | `Loudness` | Convert to dB scale |
| mode | `KeyExtractor` | Major=1, Minor=0 |
| speechiness | `SpectralComplexity` | Custom mapping needed |
| acousticness | `SpectralCentroid` | Custom mapping needed |
| instrumentalness | Inverse of vocal detection | Custom algorithm |
| liveness | `SpectralFlux` variance | Custom mapping needed |
| valence | `Mood` classifier | Use happiness/sadness |
| tempo | `RhythmExtractor2013` | BPM |

**Testing:**
```python
# Test with a local audio file first
features = extract_audio_features("test_song.mp3")
print(features)

# Verify all features are in expected range
assert 0 <= features['danceability'] <= 1
assert 0 <= features['energy'] <= 1
# etc.
```

**Deliverables:**
- `youtube_extraction/audio_features.py` with all functions
- Feature extraction working for test audio files
- Documentation of feature mapping decisions
- Caching system to avoid re-analyzing same videos

---

### Task 2.3: Create YouTube Data Pipeline
**Estimated Time:** 3-4 hours

**Create:** `youtube_extraction/youtube_pipeline.py`

**Purpose:** Combine YouTube API + Audio Analysis into one pipeline

```python
def extract_track_features_from_youtube(query, use_cache=True):
    """
    Complete pipeline: Search → Get video → Analyze audio → Return features
    Args:
        query: "Artist - Song Name"
        use_cache: Use cached features if available
    Returns:
        Dict with all track information + audio features
        {
            'id': video_id,
            'track_name': title,
            'track_artist_name': channel_name,
            'youtube_url': full_url,
            'duration_ms': duration,
            'views': view_count,
            'likes': like_count,
            'track_danceability': float,
            'track_energy': float,
            # ... all other audio features
        }
    """
    pass

def extract_playlist_features(playlist_url):
    """
    Extract features for all videos in a YouTube playlist
    Args:
        playlist_url: Full YouTube playlist URL
    Returns:
        List of track feature dicts
    """
    pass

def extract_popular_music_videos(category='Music', region='US', max_results=50):
    """
    Get features for trending/popular music videos
    Args:
        category: YouTube category
        region: Region code
        max_results: Number of videos
    Returns:
        List of track feature dicts
    """
    pass
```

**Testing:**
```python
# Test single song
features = extract_track_features_from_youtube("Queen - Bohemian Rhapsody")
print(f"Analyzed: {features['track_name']}")
print(f"Danceability: {features['track_danceability']}")

# Test playlist
playlist_features = extract_playlist_features("youtube_playlist_url_here")
print(f"Extracted {len(playlist_features)} songs from playlist")
```

**Deliverables:**
- `youtube_extraction/youtube_pipeline.py` working end-to-end
- Test script showing successful extraction
- Error handling for common issues (video not available, age-restricted, etc.)

---

## PHASE 3: Adapt Feature Engineering

### Task 3.1: Update Data Extraction Functions
**Estimated Time:** 2-3 hours

**Modify:** `data_extraction/extraction_fns.py`

**Changes needed:**
1. Replace `user_track_feature_extraction()` to work with YouTube instead of Spotify
2. Remove Spotify-specific fields (album info, explicit flag)
3. Add YouTube-specific fields (video URL, views, likes)

**New function signature:**
```python
def youtube_track_feature_extraction(video_queries, use_cache=True):
    """
    Extract track features from YouTube videos
    Args:
        video_queries: List of search queries ["Artist - Song", ...]
        use_cache: Use cached audio features if available
    Returns:
        DataFrame with all track features
    """
    track_data = []

    for query in tqdm(video_queries):
        try:
            # Use youtube_pipeline from Task 2.3
            features = extract_track_features_from_youtube(query, use_cache)
            track_data.append(features)
        except Exception as e:
            print(f"Error processing {query}: {e}")
            continue

    return pd.DataFrame(track_data)
```

**Deliverables:**
- Updated `extraction_fns.py` with YouTube support
- Backward compatibility preserved (optional)
- Test showing DataFrame creation from YouTube videos

---

### Task 3.2: Update Feature Engineering Pipeline
**Estimated Time:** 2-3 hours

**Modify:** `data_extraction/feature_eng.py`

**Changes needed:**
1. Update `drop_duplicates_df()` - use YouTube video ID instead of track+artist
2. Update `datetime_converter()` - handle YouTube upload date instead of release date
3. Update `sentiment_analysis()` - analyze video titles and descriptions
4. Remove `track_explicit` handling
5. Add views/likes normalization

**New/Modified functions:**
```python
def drop_duplicates_df_youtube(dataframe):
    """
    Drop duplicate videos based on video ID
    """
    # YouTube video IDs are unique, but same song may have multiple videos
    # Strategy: Keep highest view count version
    dataframe = dataframe.sort_values('views', ascending=False)
    dataframe = dataframe.drop_duplicates(subset=['track_name', 'track_artist_name'], keep='first')
    return dataframe

def normalize_engagement_metrics(dataframe):
    """
    Normalize views and likes to 0-1 scale
    """
    scaler = MinMaxScaler()
    dataframe['normalized_views'] = scaler.fit_transform(dataframe[['views']])
    dataframe['normalized_likes'] = scaler.fit_transform(dataframe[['likes']])
    return dataframe
```

**Deliverables:**
- Updated `feature_eng.py` working with YouTube data
- All preprocessing functions adapted
- Test showing successful preprocessing of YouTube tracks

---

### Task 3.3: Create Data Collection Script
**Estimated Time:** 2-3 hours

**Create:** `collect_youtube_music_dataset.py`

**Purpose:** Equivalent to `main.py` but for YouTube

```python
from youtube_extraction.youtube_pipeline import *
from data_extraction.feature_eng import playlist_preprocessing
import pandas as pd

def collect_popular_music():
    """
    Collect popular music videos from YouTube
    """
    print("Collecting popular music videos...")

    # Get trending music videos
    trending = extract_popular_music_videos(max_results=100)

    # Get videos from popular music channels
    popular_channels = [
        "VEVO",
        "Official Music Channel",
        # Add more
    ]

    all_tracks = trending

    # Save raw data
    df = pd.DataFrame(all_tracks)
    df.to_csv('dataset/youtube_music_raw.csv', index=False)

    # Preprocess
    processed_df, normalized_df = playlist_preprocessing(df, 'youtube_music')

    print(f"Collected {len(df)} tracks")
    return processed_df, normalized_df

def collect_from_playlists(playlist_urls):
    """
    Collect music from specific YouTube playlists
    Args:
        playlist_urls: List of YouTube playlist URLs
    """
    all_tracks = []

    for url in playlist_urls:
        print(f"Processing playlist: {url}")
        tracks = extract_playlist_features(url)
        all_tracks.extend(tracks)

    df = pd.DataFrame(all_tracks)
    processed_df, normalized_df = playlist_preprocessing(df, 'custom_playlists')

    return processed_df, normalized_df

if __name__ == '__main__':
    # Collect popular music dataset
    popular_df, popular_normalized = collect_popular_music()

    # Optional: Add your own playlists
    my_playlists = [
        "https://youtube.com/playlist?list=...",
        # Add your favorite playlists
    ]

    if my_playlists:
        custom_df, custom_normalized = collect_from_playlists(my_playlists)
```

**Deliverables:**
- `collect_youtube_music_dataset.py` script
- Successfully collected dataset of 100+ songs
- CSV files: `youtube_music_raw.csv`, `youtube_music.csv`, `normalized_youtube_music.csv`

---

## PHASE 4: Update Recommendation Engine

### Task 4.1: Verify Recommendation Algorithm
**Estimated Time:** 1-2 hours

**Review:** `recommendation_app/application/model.py`

**Good news:** The recommendation algorithm should work as-is! It uses:
- Cosine similarity on normalized features
- Feature vectors (doesn't care about source)

**Testing needed:**
```python
from recommendation_app.application.model import recommend_from_playlist
import pandas as pd

# Load YouTube music dataset
youtube_df = pd.read_csv('data/youtube_music.csv')
youtube_normalized = pd.read_csv('data/normalized_youtube_music.csv')

# Create test user playlist (simulate user likes)
user_likes = youtube_df.sample(10)  # Random 10 songs user "likes"

# Get recommendations
recommendations = recommend_from_playlist(
    youtube_df,
    youtube_normalized,
    user_likes
)

print("Top 10 recommendations:")
for i, row in recommendations.head(10).iterrows():
    print(f"{row['track_artist_name']} - {row['track_name']}")
    print(f"  URL: https://youtube.com/watch?v={row['id']}")
    print(f"  Similarity: {row['sim']:.3f}")
```

**Deliverables:**
- Test script verifying recommendations work
- Documentation of any issues found
- If issues found: modified recommendation algorithm

---

### Task 4.2: Add Diversity/Variety Logic
**Estimated Time:** 3-4 hours

**Create:** `recommendation_app/application/diversity.py`

**Problem:** Cosine similarity alone may give repetitive recommendations

**Solution:** Add diversity constraints

```python
def diversify_recommendations(recommendations_df, diversity_weight=0.3):
    """
    Re-rank recommendations to increase variety
    Args:
        recommendations_df: DataFrame with similarity scores
        diversity_weight: How much to weight diversity (0-1)
    Returns:
        Re-ranked DataFrame
    """
    # Calculate feature diversity
    # Penalize songs too similar to already recommended
    # Boost songs from different artists
    # Boost songs from different genres (if available)
    pass

def create_varied_playlist(all_songs_df, normalized_df, seed_songs, playlist_length=30, diversity=0.5):
    """
    Create a varied playlist that explores different musical spaces
    Args:
        all_songs_df: Complete song database
        normalized_df: Normalized features
        seed_songs: Songs user likes
        playlist_length: Target playlist length
        diversity: 0 = very similar, 1 = very diverse
    Returns:
        DataFrame of recommended songs with variety
    """
    # Phase 1: Get top candidates (2x playlist length)
    candidates = recommend_from_playlist(all_songs_df, normalized_df, seed_songs)
    candidates = candidates.head(playlist_length * 2)

    # Phase 2: Select diverse subset
    selected = []
    remaining = candidates.copy()

    while len(selected) < playlist_length and len(remaining) > 0:
        if len(selected) == 0:
            # Add highest similarity first
            next_song = remaining.iloc[0]
        else:
            # Add song that balances similarity + diversity
            scores = calculate_diversity_scores(remaining, selected, diversity)
            next_song = remaining.loc[scores.idxmax()]

        selected.append(next_song)
        remaining = remaining[remaining['id'] != next_song['id']]

    return pd.DataFrame(selected)
```

**Deliverables:**
- `diversity.py` with variety algorithms
- Test comparing standard vs. diverse recommendations
- Documentation of diversity strategies

---

## PHASE 5: Update Web Application

### Task 5.1: Update Flask Routes
**Estimated Time:** 2-3 hours

**Modify:** `recommendation_app/application/routes.py`

**Changes needed:**

```python
from application.features import extract_features_from_youtube_playlist
from application.model import recommend_from_playlist
from application.diversity import create_varied_playlist

# Load YouTube dataset instead of Spotify
allSongDF = pd.read_csv("./data/youtube_music.csv")
allSongFeatureSetDF = pd.read_csv("./data/normalized_youtube_music.csv")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    # Get YouTube playlist URL from form
    youtube_url = request.form['youtube_playlist_url']

    # Extract features from user's playlist
    user_playlist_df = extract_features_from_youtube_playlist(youtube_url)

    # Get number of recommendations
    number_of_recs = int(request.form['number-of-recs'])

    # Get diversity preference
    diversity = float(request.form.get('diversity', 0.3))

    # Generate recommendations
    if diversity > 0:
        recommendations = create_varied_playlist(
            allSongDF,
            allSongFeatureSetDF,
            user_playlist_df,
            playlist_length=number_of_recs,
            diversity=diversity
        )
    else:
        recommendations = recommend_from_playlist(
            allSongDF,
            allSongFeatureSetDF,
            user_playlist_df
        ).head(number_of_recs)

    # Format as list of [title, url]
    my_songs = []
    for i, row in recommendations.iterrows():
        title = f"{row['track_artist_name']} - {row['track_name']}"
        url = f"https://youtube.com/watch?v={row['id']}"
        my_songs.append([title, url])

    return render_template('results.html', songs=my_songs)

@app.route("/discover", methods=["POST"])
def discover():
    """
    New route: Discover music based on moods/genres instead of playlist
    """
    mood = request.form.get('mood', 'energetic')
    genre = request.form.get('genre', 'any')
    number_of_recs = int(request.form.get('number-of-recs', 20))

    # Filter songs by criteria
    filtered_songs = filter_by_mood_and_genre(allSongDF, mood, genre)

    # Random diverse selection
    recommendations = filtered_songs.sample(min(number_of_recs, len(filtered_songs)))

    my_songs = []
    for i, row in recommendations.iterrows():
        title = f"{row['track_artist_name']} - {row['track_name']}"
        url = f"https://youtube.com/watch?v={row['id']}"
        my_songs.append([title, url])

    return render_template('results.html', songs=my_songs)
```

**Deliverables:**
- Updated `routes.py` with YouTube support
- New `/discover` route for exploration
- Error handling for invalid YouTube URLs

---

### Task 5.2: Update Flask Feature Extraction
**Estimated Time:** 2 hours

**Modify:** `recommendation_app/application/features.py`

**Changes needed:**
Replace Spotify extraction with YouTube:

```python
from youtube_extraction.youtube_pipeline import extract_playlist_features
from data_extraction.feature_eng import playlist_preprocessing

def extract_features_from_youtube_playlist(playlist_url):
    """
    Extract features from YouTube playlist URL
    Args:
        playlist_url: Full YouTube playlist URL
    Returns:
        DataFrame with normalized features matching allSongFeatureSetDF schema
    """
    # Extract raw features
    tracks = extract_playlist_features(playlist_url)
    df = pd.DataFrame(tracks)

    # Preprocess (must match training data preprocessing)
    processed_df = playlist_preprocessing(df)

    return processed_df

def extract_features_from_video_list(video_ids):
    """
    Extract features from list of video IDs
    Args:
        video_ids: List of YouTube video IDs
    Returns:
        DataFrame with features
    """
    pass
```

**Deliverables:**
- Updated `features.py` for YouTube
- Removed Spotify dependencies
- Testing with real YouTube playlists

---

### Task 5.3: Update HTML Templates
**Estimated Time:** 2-3 hours

**Modify:** `recommendation_app/templates/home.html`

**Changes needed:**

```html
<!-- Old Spotify input -->
<!-- <input name="SpotUserName" placeholder="Enter Spotify Username"> -->

<!-- New YouTube input -->
<form action="/recommend" method="POST">
    <label for="youtube_playlist_url">YouTube Playlist URL:</label>
    <input type="url"
           name="youtube_playlist_url"
           placeholder="https://youtube.com/playlist?list=..."
           required>

    <label for="number-of-recs">Number of Recommendations:</label>
    <input type="number"
           name="number-of-recs"
           min="1"
           max="50"
           value="10">

    <label for="diversity">Playlist Variety (0-1):</label>
    <input type="range"
           name="diversity"
           min="0"
           max="1"
           step="0.1"
           value="0.3">
    <span id="diversity-value">0.3</span>

    <button type="submit">Get Recommendations</button>
</form>

<h2>Or Discover New Music</h2>
<form action="/discover" method="POST">
    <label for="mood">Mood:</label>
    <select name="mood">
        <option value="energetic">Energetic</option>
        <option value="calm">Calm</option>
        <option value="happy">Happy</option>
        <option value="melancholic">Melancholic</option>
    </select>

    <label for="number-of-recs">Number of Songs:</label>
    <input type="number" name="number-of-recs" value="20">

    <button type="submit">Discover</button>
</form>
```

**Modify:** `recommendation_app/templates/results.html`

```html
<!-- Update to embed YouTube player -->
<h1>Your Recommendations</h1>

{% for song in songs %}
<div class="song-card">
    <h3>{{ song[0] }}</h3>

    <!-- Embed YouTube player -->
    <iframe width="560"
            height="315"
            src="https://www.youtube.com/embed/{{ song[1].split('v=')[1] }}"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen>
    </iframe>

    <!-- Direct link -->
    <a href="{{ song[1] }}" target="_blank">Open in YouTube</a>

    <!-- Optional: Add to playlist button -->
    <button onclick="addToPlaylist('{{ song[1].split('v=')[1] }}')">
        Add to My Playlist
    </button>
</div>
{% endfor %}

<!-- Export playlist -->
<button onclick="exportPlaylist()">Export Playlist</button>
```

**Add:** Create playlist management features (optional)
```html
<!-- New template: my_playlist.html -->
<h1>My Playlist</h1>
<div id="playlist">
    <!-- JavaScript-managed playlist -->
</div>
```

**Deliverables:**
- Updated `home.html` with YouTube inputs
- Updated `results.html` with YouTube embeds
- Improved UI/UX for music discovery
- (Optional) Playlist management feature

---

## PHASE 6: Testing & Refinement

### Task 6.1: End-to-End Testing
**Estimated Time:** 2-3 hours

**Create:** `tests/test_youtube_pipeline.py`

**Test cases:**

```python
import pytest
from youtube_extraction.youtube_pipeline import *
from recommendation_app.application.model import *

def test_youtube_search():
    """Test YouTube video search"""
    results = search_video("The Beatles Hey Jude")
    assert len(results) > 0
    assert 'id' in results[0]

def test_feature_extraction():
    """Test audio feature extraction"""
    features = extract_track_features_from_youtube("Queen - Bohemian Rhapsody")
    assert 'track_danceability' in features
    assert 0 <= features['track_danceability'] <= 1

def test_recommendation_generation():
    """Test recommendation algorithm"""
    # Load test dataset
    all_songs = pd.read_csv('tests/test_data/youtube_music_test.csv')
    normalized = pd.read_csv('tests/test_data/normalized_test.csv')
    user_likes = all_songs.sample(5)

    # Get recommendations
    recs = recommend_from_playlist(all_songs, normalized, user_likes)

    assert len(recs) > 0
    assert 'sim' in recs.columns
    assert recs['sim'].iloc[0] >= recs['sim'].iloc[-1]  # Sorted by similarity

def test_web_app():
    """Test Flask app routes"""
    from recommendation_app.start import app
    client = app.test_client()

    # Test home page
    response = client.get('/')
    assert response.status_code == 200

    # Test recommendation (would need mock data)
    # ...

if __name__ == '__main__':
    pytest.main([__file__])
```

**Manual test checklist:**
- [ ] YouTube API authentication works
- [ ] Can search for and find music videos
- [ ] Audio feature extraction produces valid features
- [ ] Features are normalized correctly
- [ ] Recommendation algorithm produces sensible results
- [ ] Flask app loads without errors
- [ ] Can submit YouTube playlist URL
- [ ] Recommendations are displayed as YouTube links
- [ ] YouTube embeds play correctly
- [ ] Diversity slider affects results

**Deliverables:**
- Automated test suite
- Test coverage report
- Manual testing checklist completed
- Bug reports for any issues found

---

### Task 6.2: Performance Optimization
**Estimated Time:** 3-4 hours

**Bottlenecks to address:**

#### 1. Audio Feature Extraction (Slow!)
**Problem:** Downloading and analyzing audio takes time

**Solutions:**
```python
# A. Batch processing
def batch_extract_features(video_ids, batch_size=10):
    """Process multiple videos in parallel"""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        features = list(executor.map(extract_track_features_from_youtube, video_ids))
    return features

# B. Persistent caching
import pickle
import hashlib

def get_cached_features(video_id, cache_dir='cache/audio_features'):
    """Load features from disk cache"""
    cache_file = f"{cache_dir}/{video_id}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_cached_features(video_id, features, cache_dir='cache/audio_features'):
    """Save features to disk cache"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{video_id}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(features, f)

# C. Pre-compute popular songs
def precompute_popular_songs():
    """Run once to build initial dataset"""
    # Get top 10,000 popular music videos
    # Extract features
    # Save to database
    pass
```

#### 2. YouTube API Rate Limits
**Problem:** YouTube API has quota limits (10,000 units/day)

**Solutions:**
- Cache API responses
- Use YouTube search library (no API key needed) for search
- Reserve API quota for video details only

```python
from youtubesearchpython import VideosSearch

def search_without_api(query, limit=10):
    """Search YouTube without using API quota"""
    search = VideosSearch(query, limit=limit)
    results = search.result()
    return results['result']
```

#### 3. Recommendation Speed
**Already fast!** Cosine similarity is efficient.

**Optional improvement:**
```python
# Use approximate nearest neighbors for very large datasets
from annoy import AnnoyIndex

def build_annoy_index(features_df):
    """Build approximate nearest neighbor index"""
    dim = features_df.shape[1]
    index = AnnoyIndex(dim, 'angular')

    for i, row in features_df.iterrows():
        index.add_item(i, row.values)

    index.build(10)  # 10 trees
    return index
```

**Deliverables:**
- Implemented caching system
- Batch processing for feature extraction
- Performance benchmarks (before/after)
- Documentation of optimization strategies

---

### Task 6.3: Error Handling & Edge Cases
**Estimated Time:** 2-3 hours

**Common errors to handle:**

```python
# 1. Video not available
def safe_extract_features(video_id):
    try:
        return extract_track_features_from_youtube(video_id)
    except VideoNotAvailable:
        logger.warning(f"Video {video_id} not available")
        return None
    except AgeRestricted:
        logger.warning(f"Video {video_id} age restricted")
        return None

# 2. Invalid playlist URL
def validate_playlist_url(url):
    import re
    pattern = r'(?:youtube\.com/playlist\?list=|youtu\.be/)([a-zA-Z0-9_-]+)'
    match = re.search(pattern, url)
    if not match:
        raise ValueError("Invalid YouTube playlist URL")
    return match.group(1)

# 3. Empty results
def recommend_with_fallback(all_songs_df, normalized_df, user_df):
    try:
        recs = recommend_from_playlist(all_songs_df, normalized_df, user_df)
        if len(recs) == 0:
            # Fallback: return popular songs
            return all_songs_df.sort_values('views', ascending=False).head(20)
        return recs
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        return all_songs_df.sample(20)  # Random songs as last resort

# 4. Audio analysis failures
def extract_with_retry(video_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            return download_and_analyze(video_id)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

**User-facing error messages:**
```python
# In Flask routes
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html',
                         message="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html',
                         message="Something went wrong. Please try again."), 500

# In recommendation route
try:
    recommendations = generate_recommendations(...)
except InvalidPlaylistURL:
    return render_template('error.html',
                         message="Invalid YouTube playlist URL. Please check and try again.")
except QuotaExceeded:
    return render_template('error.html',
                         message="Daily API limit reached. Please try again tomorrow.")
```

**Deliverables:**
- Comprehensive error handling throughout codebase
- User-friendly error messages
- Logging system for debugging
- Error recovery strategies

---

### Task 6.4: Documentation & Final Polish
**Estimated Time:** 2-3 hours

**Documentation updates:**

1. **Update main README.md** with new YouTube instructions

2. **Create YOUTUBE_SETUP.md:**
```markdown
# YouTube Music Recommender Setup Guide

## Quick Start
1. Get YouTube API key
2. Install dependencies: `pip install -r requirements.txt`
3. Add API key to credentials.json
4. Build music database: `python collect_youtube_music_dataset.py`
5. Run web app: `cd recommendation_app && python start.py`

## Detailed Instructions
[Rest of guide...]
```

3. **Create API_USAGE.md:**
```markdown
# API Usage & Quotas

## YouTube Data API v3
- Daily quota: 10,000 units
- Search: 100 units per request
- Video details: 1 unit per video
- Playlist items: 1 unit per request

## Rate Limit Strategy
- Cache all API responses
- Use youtubesearchpython for search (no quota)
- Pre-compute popular songs
- Expected usage: ~500 units per day

## Monitoring
Check quota usage: https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas
```

4. **Add code comments** to complex functions

5. **Create example notebooks:**
```python
# example_usage.ipynb
"""
Jupyter notebook showing:
1. How to search for songs
2. How to extract features
3. How to generate recommendations
4. How to create diverse playlists
"""
```

**Deliverables:**
- Updated README.md
- YOUTUBE_SETUP.md guide
- API_USAGE.md reference
- Code comments added
- Example Jupyter notebook
- Video tutorial (optional)

---

## PHASE 7: Advanced Features (Optional)

### Task 7.1: Genre/Mood Classification
**Estimated Time:** 4-6 hours

**Goal:** Automatically classify songs by genre and mood

**Approach:**
```python
# Use audio features to predict genre/mood
from sklearn.ensemble import RandomForestClassifier

def train_genre_classifier():
    """
    Train classifier using labeled dataset
    Option 1: Use Million Song Dataset (has genre labels)
    Option 2: Manually label 100-200 songs for training
    """
    pass

def predict_genre(audio_features):
    """
    Predict genre from audio features
    Returns: ['rock', 'pop', 'electronic', etc.]
    """
    pass

def predict_mood(audio_features):
    """
    Predict mood from audio features
    Based on: valence (happiness), energy (intensity), tempo
    Returns: 'energetic', 'calm', 'happy', 'melancholic'
    """
    # Simple rule-based approach
    if audio_features['valence'] > 0.6 and audio_features['energy'] > 0.6:
        return 'energetic'
    elif audio_features['valence'] < 0.4 and audio_features['energy'] < 0.4:
        return 'melancholic'
    elif audio_features['valence'] > 0.6:
        return 'happy'
    else:
        return 'calm'
```

---

### Task 7.2: Playlist Export Feature
**Estimated Time:** 2-3 hours

**Goal:** Export recommendations as YouTube playlist

**Implementation:**
```python
# Requires YouTube OAuth (not just API key)
from google_auth_oauthlib.flow import InstalledAppFlow

def create_youtube_playlist(video_ids, playlist_name='My Recommendations'):
    """
    Create new YouTube playlist with recommended songs
    Args:
        video_ids: List of YouTube video IDs
        playlist_name: Name for new playlist
    Returns:
        Playlist URL
    """
    # Authenticate user
    # Create playlist
    # Add videos to playlist
    pass
```

---

### Task 7.3: User Preference Learning
**Estimated Time:** 4-6 hours

**Goal:** Remember user likes/dislikes to improve recommendations

**Implementation:**
```python
# Use browser localStorage or simple database
import sqlite3

def save_user_feedback(user_id, video_id, liked=True):
    """Save user like/dislike"""
    pass

def get_user_history(user_id):
    """Get user's listening history"""
    pass

def personalized_recommendations(user_id, candidate_songs):
    """
    Adjust recommendations based on user history
    - Boost songs similar to liked songs
    - Filter out disliked songs
    - Learn preferred genres/moods
    """
    pass
```

---

## Implementation Timeline

### Minimum Viable Product (MVP) - 2-3 weeks
**Goal:** Basic YouTube recommendations working

**Priority tasks:**
- Phase 1: Research & Setup (2-4 days)
- Phase 2: YouTube Data Extraction (5-7 days)
- Phase 3: Feature Engineering (3-4 days)
- Phase 4: Recommendation Engine (1-2 days)
- Phase 5: Web App Update (3-4 days)
- Phase 6: Testing (2-3 days)

### Full Featured Version - 4-6 weeks
Add:
- Performance optimization
- Advanced diversity algorithms
- Genre/mood classification
- Playlist export
- User preference learning

---

## Alternative Approaches

### Approach A: Hybrid (Spotify + YouTube)
**Idea:** Use Spotify API for features, YouTube for playback
- **Pros:** Spotify has better audio features, easier
- **Cons:** Still requires Spotify account, more complex

### Approach B: Pre-computed Dataset Only
**Idea:** Use Million Song Dataset, no real-time analysis
- **Pros:** Fast, no API limits
- **Cons:** Limited to songs in dataset (~1M songs from pre-2011)

### Approach C: Audio Fingerprinting
**Idea:** Use AcoustID or similar to match YouTube videos to known songs
- **Pros:** Can leverage existing song databases
- **Cons:** Additional complexity, API dependencies

**Recommended:** Start with main approach, consider alternatives if blocked

---

## Success Criteria

### Technical Metrics
- [ ] Can extract audio features from 100 YouTube videos in < 1 hour
- [ ] Recommendation accuracy: >70% of recommendations rated as "good match"
- [ ] Web app response time: < 5 seconds for 20 recommendations
- [ ] API quota usage: < 1,000 units per day

### User Experience Metrics
- [ ] User can get recommendations without Spotify account
- [ ] Recommendations include YouTube video links
- [ ] Playlists have good variety (not repetitive)
- [ ] Can discover new music outside comfort zone

### Deliverables Checklist
- [ ] Working YouTube data extraction pipeline
- [ ] Audio feature extraction from YouTube videos
- [ ] Updated recommendation algorithm
- [ ] Flask web app with YouTube integration
- [ ] Comprehensive documentation
- [ ] Test suite with >80% coverage

---

## Resources & References

### APIs & Libraries
- [YouTube Data API v3](https://developers.google.com/youtube/v3)
- [Essentia](https://essentia.upf.edu/)
- [LibROSA](https://librosa.org/)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [youtube-search-python](https://github.com/alexmercerind/youtube-search-python)

### Datasets
- [Million Song Dataset](http://millionsongdataset.com/)
- [Free Music Archive](https://github.com/mdeff/fma)
- [MusicBrainz](https://musicbrainz.org/)

### Research Papers
- [Music Recommendation Systems: Techniques, Use Cases, and Challenges](https://arxiv.org/abs/2107.00307)
- [Content-Based Music Recommendation](https://link.springer.com/chapter/10.1007/978-1-0716-2197-4_2)

### Tutorials
- [Building a Music Recommendation System](https://towardsdatascience.com/building-a-music-recommendation-engine-with-spotify-data-d1d8e81f5e0)
- [Audio Feature Extraction with Essentia](https://mtg.github.io/essentia-labs/)

---

## Next Steps

**Start with:**
1. Task 1.1: Set up YouTube API (30 min)
2. Task 1.2: Research audio analysis libraries (2-3 hours)
3. Task 2.1: Create YouTube API wrapper (2-3 hours)

**First milestone:** Successfully extract features from 10 YouTube videos

**Questions to answer before starting:**
1. How many songs do you want in your database? (100? 1,000? 10,000?)
2. Do you want to analyze audio in real-time or pre-compute?
3. What's your computational budget? (Local processing or cloud?)
4. Do you need the latest music or is older catalog OK?

**Ready to start? Let's begin with Task 1.1!**
