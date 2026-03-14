"""
Entry point for YouTube Music Recommender Flask Application
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from application import app

# Import YouTube routes (this will register them with the app)
import application.routes_youtube

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎵 YouTube Music Recommender Server Starting...")
    print("="*60)
    print("\nServer will be available at: http://localhost:8000")
    print("\nMake sure you have:")
    print("  1. Run 'collect_youtube_dataset.py' to build music database")
    print("  2. Installed all requirements from 'requirements_youtube.txt'")
    print("\nPress Ctrl+C to stop the server\n")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=8000)
