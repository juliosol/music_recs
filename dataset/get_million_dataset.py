import kagglehub

# Download latest version
path = kagglehub.dataset_download("himanshuwagh/spotify-million")

print("Path to dataset files:", path)