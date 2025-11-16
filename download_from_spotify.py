import os
import subprocess
from helper import mp3_to_wav

def create_og_playlist(playlist_url, playlistName):
    playlist_dir = f"static/playlists/og/{playlistName}"
    # venv_spotdl_path = os.path.join(project_root, "venv", "bin", "spotdl")
    spotdl_file_dir = f"static/playlists/og/{playlistName}"

    # Ensure directories exist
    os.makedirs(playlist_dir, exist_ok=True)
    os.makedirs(spotdl_file_dir, exist_ok=True)

    print("made directories, getting song")

    subprocess.run([
        "spotdl",
        "sync",
        playlist_url,
        "--save-file", f"{spotdl_file_dir}/playlist.spotdl",
        "--output", f"{playlist_dir}/{{title}}.{{output-ext}}",
        "--format", "mp3"  # suppress stderr
    ], check=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("songs retrieved")

    mp3_to_wave(playlist_dir)
    print("Converted all mp3 to wav")



    
def mp3_to_wave(playlist_dir,logger=print):
    for file in os.listdir(playlist_dir):
        if file.endswith(".mp3"):
            mp3_path = os.path.join(playlist_dir, file)
            mp3_to_wav(mp3_path,mp3_path.replace(".mp3", ".wav"))
            os.remove(mp3_path)

    logger("Converted all files to wav")


import os
import subprocess
from helper import mp3_to_wav

def download_track(track_id, track_name,logger=print):
    track_dir = f"static/tracks"
    os.makedirs(track_dir, exist_ok=True)

    print(f"Downloading track {track_id}...")

    # Use spotdl to download the track
    try:
        subprocess.run([
            "spotdl",
            f"https://open.spotify.com/track/{track_id}",
            "--output", f"{track_dir}/{track_name}.{{output-ext}}",
            "--format", "mp3"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download {track_id}: {e}", flush=True)
        raise

    print("Track downloaded, converting to WAV...")
    mp3_path = os.path.join(track_dir, f"{track_name}.mp3")
    wav_path = os.path.join(track_dir, f"{track_name}.wav")
    mp3_to_wav(mp3_path, wav_path)
    os.remove(mp3_path)
    print(f"Conversion complete: {wav_path}")



# Example usage:
# download_track("3n3Ppam7vgaVa1iaRUc9Lp", "example_track")


#create_og_playlist("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M","test")
#mp3_to_wave("app/playlists/test/notlofied/")