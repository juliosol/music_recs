# 🚀 Quick Start Guide - YouTube Music Recommender

Get up and running in 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
# Install FFmpeg (required for audio extraction)
brew install ffmpeg  # macOS
# or: sudo apt-get install ffmpeg  # Linux

# Install Python packages
pip install -r requirements_youtube.txt
```

## Step 2: Verify Setup (1 minute)

```bash
# Test that everything is installed correctly
python test_youtube_system.py
```

You should see all tests passing! ✓

## Step 3: Build Music Database (30-60 minutes)

```bash
# This will download and analyze ~50 songs
# First run is slow, but features are cached!
python collect_youtube_dataset.py
```

**What's happening:**
- Searching for popular songs across genres
- Downloading audio from YouTube
- Extracting audio features (danceability, energy, etc.)
- Saving to database

**Note:** Grab a coffee! ☕ This takes 30-60 minutes on first run.

## Step 4: Start the Web App (30 seconds)

```bash
cd recommendation_app
python start_youtube.py
```

Open your browser to: **http://localhost:5000**

## Step 5: Get Recommendations!

### Method A: From Your Playlist
1. Find a YouTube playlist you like
2. Copy the URL
3. Paste into the app
4. Adjust variety slider
5. Get recommendations!

### Method B: Discover by Mood
1. Select a mood (energetic, calm, happy, melancholic)
2. Choose number of songs
3. Discover new music!

---

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements_youtube.txt
```

### "FFmpeg not found"
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### "Dataset not found"
```bash
# Run this first to build the database
python collect_youtube_dataset.py
```

### "YouTube API quota exceeded"
- Don't worry! Most functions don't use quota
- Cached features don't need re-downloading
- Free tier gives 10,000 units/day

---

## What's Next?

### Customize Your Database
Edit `collect_youtube_dataset.py` and add your favorite songs:

```python
queries = [
    "Your Favorite Artist - Song",
    "Another Artist - Song",
    # Add 20-50 songs you like!
]
```

Then run:
```bash
python collect_youtube_dataset.py
```

### Adjust Settings
- **Variety Slider:** 0 = similar songs, 1 = very diverse
- **Max Per Artist:** Prevent too many songs from one artist
- **Number of Recs:** Get 1-50 recommendations

---

## Need Help?

1. Check `README_YOUTUBE.md` for detailed documentation
2. Run `python test_youtube_system.py` to diagnose issues
3. Check the troubleshooting section

---

## 🎉 That's It!

You now have a working music recommender system!

**Enjoy discovering new music! 🎵**
