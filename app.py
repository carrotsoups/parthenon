from flask import Flask, request, render_template, jsonify, Response, stream_with_context
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import threading
import queue
from download_from_spotify import download_track
from m import transform_to_lofi  # Your lofi function

app = Flask(__name__)
app.config['STATIC_TRACKS_PATH'] = 'static/tracks'
os.makedirs(app.config['STATIC_TRACKS_PATH'], exist_ok=True)

# Spotify credentials
SPOTIFY_CLIENT_ID = "192fc22852a845cbb46f731d0acbd538"
SPOTIFY_CLIENT_SECRET = "831ed3fc3d8b41bb88affa43525add5d"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

# ------------------------------
# SSE / Logging Helper
# ------------------------------
def stream_lofi(track_id):
    log_queue = queue.Queue()

    def log_transform():
        def log(msg):
            log_queue.put(msg)
            print(msg, flush=True)

        original_path = os.path.join(app.config['STATIC_TRACKS_PATH'], f"{track_id}.wav")
        if not os.path.exists(original_path):
            log("Downloading track...")
            download_track(track_id, track_id,logger=log)
            log("Download complete.")

        stem_dir = os.path.join(app.config['STATIC_TRACKS_PATH'], f"{track_id}_stems")
        os.makedirs(stem_dir, exist_ok=True)
        log("Starting Lo-fi transformation...")
        transform_to_lofi(original_path, stem_dir, track_id, logger=log)
        log_queue.put(f"LOFI_DONE::{track_id}")
        log("Lo-fi transformation finished!")

    threading.Thread(target=log_transform).start()

    while True:
        msg = log_queue.get()
        yield f"data: {msg}\n\n"
        if msg.startswith("LOFI_DONE"):
            break

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_playlist', methods=['POST'])
def get_playlist():
    data = request.get_json()
    playlist_url = data['playlistUrl']

    if "playlist/" in playlist_url:
        playlist_id = playlist_url.split("playlist/")[1].split("?")[0]
    else:
        return jsonify({"error": "Invalid playlist URL"}), 400

    tracks_data = sp.playlist_items(playlist_id)
    tracks = []
    for item in tracks_data['items']:
        track = item['track']
        tracks.append({
            "name": track['name'],
            "artist": track['artists'][0]['name'],
            "track_id": track['id']
        })

    return jsonify({"tracks": tracks})


@app.route('/process_song', methods=['POST'])
def process_song():
    data = request.get_json()
    track_id = data['track_id']

    download_track(track_id, track_id)
    processed_filename = f"{track_id}.wav"

    return jsonify({"audio_url": f"/static/tracks/{processed_filename}"})


# ------------------------------
# SSE Endpoint for live logs
# ------------------------------
@app.route('/lofi_song_stream', methods=['GET'])
def lofi_song_stream():
    track_id = request.args.get("track_id")
    return Response(stream_with_context(stream_lofi(track_id)), mimetype='text/event-stream')




@app.route('/library')
def library():
    tracks = {}
    for file in os.listdir(app.config['STATIC_TRACKS_PATH']):
        if file.endswith(".wav") and "_lofi" not in file:
            track_id = file.replace(".wav", "")
            lofi_file = f"{track_id}_lofi.wav"
            stem_dir = os.path.join(app.config['STATIC_TRACKS_PATH'], f"{track_id}_stems")
            stems = []
            if os.path.exists(stem_dir):
                stems = [s for s in os.listdir(stem_dir) if s.endswith(".wav")]

            # Fetch Spotify metadata
            try:
                track_info = sp.track(track_id)
                name = track_info['name']
                artist = track_info['artists'][0]['name']
                album_img = track_info['album']['images'][0]['url'] if track_info['album']['images'] else None
            except Exception as e:
                print(f"Error fetching Spotify data for {track_id}: {e}")
                name, artist, album_img = track_id, "Unknown", None

            tracks[track_id] = {
                "original": file,
                "lofi": lofi_file if os.path.exists(os.path.join(app.config['STATIC_TRACKS_PATH'], lofi_file)) else None,
                "stems": stems,
                "name": name,
                "artist": artist,
                "album_img": album_img
            }

    return render_template("library.html", tracks=tracks)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
