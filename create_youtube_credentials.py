"""
Create or update local credentials.json with youtube_api_key.

This script keeps existing keys (e.g., Spotify keys) and only adds/updates
"youtube_api_key" so deleted credentials can be restored quickly.
"""
import argparse
import json
import os
import sys


def load_existing_credentials(path):
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Create or update credentials.json with youtube_api_key"
    )
    parser.add_argument(
        "--key",
        default=os.getenv("YOUTUBE_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        help="YouTube Data API key (defaults to YOUTUBE_API_KEY/GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--path",
        default="credentials.json",
        help="Path to credentials json file (default: credentials.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing youtube_api_key without prompt",
    )
    args = parser.parse_args()

    api_key = args.key
    if not api_key:
        api_key = input("Enter your YouTube API key: ").strip()

    if not api_key:
        print("No API key provided. Nothing was written.")
        return 1

    credentials = load_existing_credentials(args.path)

    if "youtube_api_key" in credentials and credentials["youtube_api_key"] and not args.force:
        print(f"A youtube_api_key already exists in {args.path}.")
        response = input("Replace it? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborted. Existing key was kept.")
            return 0

    credentials["youtube_api_key"] = api_key

    with open(args.path, "w") as f:
        json.dump(credentials, f, indent=2)

    print(f"Updated {args.path} with youtube_api_key.")
    print("Tip: credentials.json is gitignored, so this stays local.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
