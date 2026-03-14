# Expanding the YouTube Music Dataset

This project builds its recommendation database from YouTube metadata and audio features. You can increase the dataset size by adding more seed queries, more playlists, and/or trending music by region, then re-running the collector.

## How it works
The collector in `collect_youtube_dataset.py` now supports:
- Curated queries (built-in defaults)
- Seed queries from a text file
- Seed playlists from a text file
- Optional trending music per region
- Append mode to merge with existing raw data

The outputs are regenerated so the app can use the larger database:
- `data_extraction/youtube_music_raw.csv`
- `data_extraction/youtube_music.csv`
- `data_extraction/normalized_youtube_music.csv`

## Add more music sources
### 1) Seed queries
Add your own queries (one per line) in:
- `data_extraction/seed_queries.txt`

Example lines:
- `Billie Eilish bad guy`
- `Metallica Enter Sandman`
- `A.R. Rahman Jai Ho`

### 2) Seed playlists
Add public YouTube playlist URLs (one per line) in:
- `data_extraction/seed_playlists.txt`

Example lines:
- `https://www.youtube.com/playlist?list=PL...`

### 3) Trending music (optional)
Include trending music by region using `--include-popular` and provide region codes with `--regions`.

## Commands to increase the database
Run the collector from the project root.

- **Basic expansion (curated + seed files):**
  `python collect_youtube_dataset.py --append`

- **Include trending music (US + GB):**
  `python collect_youtube_dataset.py --append --include-popular --regions US,GB --max-per-region 100`

- **Increase playlist coverage:**
  `python collect_youtube_dataset.py --append --max-per-playlist 200`

- **Skip curated defaults (use only seed files):**
  `python collect_youtube_dataset.py --append --skip-curated`

## Notes
- `--append` merges new tracks with the existing raw dataset and removes duplicate video IDs.
- Larger runs will take longer because audio analysis downloads are performed per video.
- API quota is only used for trending and detailed video metadata calls.
