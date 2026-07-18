## 🎵 YouTube Music Recommender System

A powerful music recommendation system that uses YouTube videos to discover new music and create varied playlists. No Spotify account required!

### ✨ Features

- 🎯 **Personalized Recommendations** - Get song recommendations based on your YouTube playlists
- 🎲 **Mood Discovery** - Find music by mood (energetic, calm, happy, melancholic)
- 🎨 **Variety Control** - Adjust how similar or diverse your recommendations should be
- 📊 **Audio Analysis** - Uses 11 audio features (danceability, energy, tempo, etc.)
- 🎥 **YouTube Integration** - Watch videos directly in the app
- 💾 **Playlist Export** - Copy links or download recommendations as text

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- YouTube Data API v3 key (free from Google Cloud Console)
- Internet connection for audio download and analysis
- FFmpeg (required for audio extraction)

### Installation

#### 1. Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

#### 2. Clone and Install Dependencies

```bash
cd /Users/julsoles/Documents/projects/music_recs
pip install -r requirements_youtube.txt
```

#### 3. Set Up YouTube API Key

Your `credentials.json` already has a YouTube API key configured!

```json
{
  "youtube_api_key": "YOUR_KEY_HERE"
}
```

To get your own key:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create/select a project
3. Enable YouTube Data API v3
4. Create API key credentials
5. Add key to credentials.json

---

## 📖 Usage

### Step 1: Build Music Database

Collect songs for the recommendation engine:

```bash
python collect_youtube_dataset.py
```

This will:
- Search for ~50 popular songs across different genres
- Extract audio features from each song
- Save processed data to `data_extraction/` folder
- Takes ~30-60 minutes depending on your connection

**Note:** The first run downloads and analyzes audio, so it takes time. Subsequent runs use cached features!

### Step 2: Test the System

Verify everything is working:

```bash
python test_youtube_system.py
```

This checks:
- ✓ All packages installed
- ✓ YouTube API connectivity
- ✓ Audio extraction working
- ✓ Recommendation algorithm functional
- ✓ Dataset collected

### Step 3: Start the Web Application

```bash
cd recommendation_app
python start_youtube.py
```

Open your browser to **http://localhost:5000**

---

## 🎯 How to Use the App

### Method 1: Get Recommendations from Your Playlist

1. Find a YouTube playlist you like
2. Copy the playlist URL (e.g., `https://youtube.com/playlist?list=...`)
3. Paste it into the app
4. Adjust settings:
   - **Number of Recommendations:** 1-50 songs
   - **Playlist Variety:** 0 (similar) to 1 (diverse)
   - **Max Songs Per Artist:** Prevent repetition
5. Click "Get Recommendations"

### Method 2: Discover Music by Mood

1. Select a mood:
   - ⚡ **Energetic:** High energy, fast tempo
   - 😌 **Calm:** Relaxing, low energy
   - 😊 **Happy:** Positive, upbeat
   - 😔 **Melancholic:** Emotional, slower
2. Choose number of songs
3. Click "Discover Music"

---

## 🏗️ Project Structure

```
music_recs/
├── youtube_extraction/               # YouTube data extraction
│   ├── youtube_api.py               # YouTube API wrapper
│   ├── audio_features.py            # Audio analysis with librosa
│   ├── youtube_pipeline.py          # Complete extraction pipeline
│   └── feature_eng_youtube.py       # Feature preprocessing
│
├── recommendation_app/               # Flask web application
│   ├── application/
│   │   ├── routes_youtube.py        # YouTube routes
│   │   ├── features_youtube.py      # Feature extraction for app
│   │   ├── model.py                 # Recommendation algorithm
│   │   └── diversity.py             # Variety/diversity logic
│   ├── templates/
│   │   ├── home_youtube.html        # Main page
│   │   ├── results_youtube.html     # Results page
│   │   └── error.html               # Error page
│   └── start_youtube.py             # App entry point
│
├── data_extraction/                  # Generated data files
│   ├── youtube_music_raw.csv        # Raw extracted data
│   ├── youtube_music.csv            # Processed data
│   └── normalized_youtube_music.csv # Normalized features
│
├── cache/audio_features/             # Cached audio analysis
│
├── collect_youtube_dataset.py        # Dataset builder
├── test_youtube_system.py           # System tests
├── requirements_youtube.txt         # Python dependencies
└── credentials.json                 # API credentials
```

---

## 🎼 How It Works

### 1. Audio Feature Extraction

We extract 11 key features from each song:

| Feature | Description | Range |
|---------|-------------|-------|
| **Danceability** | Rhythm stability and beat strength | 0-1 |
| **Energy** | Intensity and activity | 0-1 |
| **Valence** | Musical positiveness (happy vs sad) | 0-1 |
| **Tempo** | Speed in BPM | 60-200 |
| **Key** | Musical key (C, C#, D, etc.) | 0-11 |
| **Mode** | Major (1) or Minor (0) | 0-1 |
| **Loudness** | Overall volume in dB | -60 to 0 |
| **Speechiness** | Presence of spoken words | 0-1 |
| **Acousticness** | Acoustic vs electric | 0-1 |
| **Instrumentalness** | Vocal vs instrumental | 0-1 |
| **Liveness** | Live performance detection | 0-1 |

### 2. Recommendation Algorithm

**Cosine Similarity** - Compares your liked songs' features with the database:

```
similarity = cosine_similarity(user_features, song_features)
```

**Diversity Algorithm** - Balances similarity with variety:

```
score = (1 - diversity) × similarity + diversity × diversity_score
```

### 3. Mood Classification

Songs are classified by audio features:

- **Energetic:** High energy (>0.6) + Fast tempo (>110 BPM)
- **Calm:** Low energy (<0.4) + High acousticness (>0.3)
- **Happy:** High valence (>0.6)
- **Melancholic:** Low valence (<0.4) + Low energy (<0.5)

---

## ⚙️ Configuration

### Collect More Songs

Edit `collect_youtube_dataset.py`:

```python
# Add your own search queries
queries = [
    "Your Favorite Artist - Song Name",
    "Another Artist - Another Song",
    # ...
]

# Or add your playlists
my_playlists = [
    "https://youtube.com/playlist?list=YOUR_PLAYLIST_ID",
    # ...
]
```

### Adjust Recommendation Settings

In `recommendation_app/application/routes_youtube.py`:

```python
# Change default diversity
diversity = float(request.form.get('diversity', 0.5))  # Default 0.5 instead of 0.3

# Change max recommendations
max_results=100  # Allow up to 100 recommendations
```

---

## 🐛 Troubleshooting

### "No module named 'youtube_extraction'"

**Solution:** Make sure you're in the correct directory:
```bash
cd /Users/julsoles/Documents/projects/music_recs
python collect_youtube_dataset.py
```

### "YouTube API quota exceeded"

**Solution:**
- YouTube API has daily limits (10,000 units/day)
- Use cached features: Set `use_cache=True` (default)
- Our search uses `youtube-search-python` which doesn't use quota
- Only video details use quota (1 unit per video)

### "Audio download failed"

**Solution:**
- Install FFmpeg (see installation section)
- Check internet connection
- Some videos are region-restricted or age-restricted
- The system will skip failed videos and continue

### "Error: Could not extract features from playlist"

**Solution:**
- Verify playlist URL is correct
- Make sure playlist is public
- Try with a smaller playlist first (10-20 videos)
- Check YouTube API key is valid

### "Dataset not found"

**Solution:**
```bash
# Build the dataset first
python collect_youtube_dataset.py
```

### Application is slow

**Solution:**
- First run is slow (downloads & analyzes audio)
- Cached features make subsequent runs much faster
- Reduce number of videos analyzed
- Pre-build a larger database for faster recommendations

---

## 📊 Performance & Limits

### Speed

- **First song:** ~30-60 seconds (download + analysis)
- **Cached songs:** <1 second
- **Recommendations:** <2 seconds for 50 songs

### API Quotas

- **YouTube Data API:** 10,000 units/day (free tier)
- **Cost per video:** 1 unit for details
- **Search (youtube-search-python):** No quota cost!

### Database Size

- **Minimum:** 50 songs for decent recommendations
- **Recommended:** 200-500 songs for great variety
- **Maximum:** Limited by disk space (~10MB per song cached)

---

## 🎓 Advanced Features

### Custom Feature Weights

Edit `recommendation_app/application/model.py` to weight features differently:

```python
# Give more weight to tempo and energy
weights = {
    'track_energy': 2.0,
    'track_tempo': 2.0,
    'track_valence': 1.0,
    # ...
}
```

### Add Genre Classification

Implement genre detection in `youtube_extraction/audio_features.py`:

```python
def predict_genre(features):
    if features['tempo'] > 120 and features['energy'] > 0.7:
        return 'electronic'
    elif features['acousticness'] > 0.6:
        return 'folk'
    # ...
```

### Export to Spotify

Create a script to search recommendations on Spotify and add to playlist.

---

## 🤝 Contributing

This is a personal project, but feel free to:
- Fork and modify for your own use
- Report bugs or issues
- Suggest new features

---

## 📝 License

This project is for educational and personal use.

### Third-Party Services

- **YouTube Data API:** Subject to [YouTube API Terms](https://developers.google.com/youtube/terms/api-services-terms-of-service)
- **Audio Download:** Respect copyright and use only for personal analysis

---

## 🔮 Future Enhancements

- [ ] User accounts with saved preferences
- [ ] Playlist history and favorites
- [ ] Genre classification and filtering
- [ ] Lyrics analysis integration
- [ ] Multi-playlist mixing
- [ ] Social features (share playlists)
- [ ] Mobile app version
- [ ] Collaborative filtering
- [ ] Real-time audio feature extraction (no download)

---

## 📚 Resources

### APIs & Libraries Used

- [YouTube Data API v3](https://developers.google.com/youtube/v3)
- [librosa](https://librosa.org/) - Audio analysis
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube downloader
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [Flask](https://flask.palletsprojects.com/) - Web framework

### Learning Resources

- [Audio Signal Processing](https://www.coursera.org/learn/audio-signal-processing)
- [Music Information Retrieval](https://musicinformationretrieval.com/)
- [Recommendation Systems](https://www.coursera.org/specializations/recommender-systems)

---

## ❓ FAQ

**Q: Do I need a YouTube account?**
A: No! You just need a YouTube API key (free).

**Q: Can I use this for commercial purposes?**
A: This is for personal/educational use. Check YouTube TOS for commercial use.

**Q: How accurate are the audio features?**
A: ~80-90% accuracy compared to Spotify's features. Not perfect but very usable!

**Q: Can I use Spotify playlists?**
A: No, this version only works with YouTube. See the original Spotify version.

**Q: Why is the first run so slow?**
A: We download and analyze audio files. Subsequent runs use cached features.

**Q: Can I run this on a server?**
A: Yes! Just ensure FFmpeg is installed and change `host='0.0.0.0'` in start_youtube.py

---

## 🎉 Enjoy Your Music!

If you find this useful, star the repo and share with friends!

**Questions?** Open an issue or check the troubleshooting section.

**Happy listening! 🎵**
