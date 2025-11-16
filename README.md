# parthenon
decompose your fav spotify songs into lofi ish. get the stems too!<br>
so uh.... its too memory intensive to host on free sites. Thus, please install it manually

Dependencies: 
- **python 3.11** (make sure `C:<PYTHON_DIR>\Python\Python310\python.exe` is added to `Path` in the `SYSTEM ENVIRONMENT VARIABLES`)
- ffmpeg
    - mac: run `brew install ffmpeg`
    - windows: download source code from https://www.ffmpeg.org/download.html. Extract the zip contents. Add `C:<EXTRACTED_FOLDER>\bin\` to `Path` in the `SYSTEM ENVIRONMENT VARIABLES`
    - linux: `sudo apt install ffmpeg`


Getting the website:
  1. Clone the repo `git clone https://github.com/carrotsoups/parthenon.git`
  2. Enter directory: `cd parthenon`
  3. Create a virtual environment `py -3.11 -m venv venv`
  4. Activate environment: `.\venv\Scripts\activate` (windows), `source venv/bin/activate` (linux & mac)
  5. Install all dependencies: `py -3.11 -m pip install -r requirements.txt`
  6. Host the Flask app: `py -3.11 app.py`
  7. Open the website! (http://127.0.0.1:5000/)

# Demo
<figure class="video_container">
  <iframe src="/demo.mp4" frameborder="0" allowfullscreen="true"> 
</iframe>
</figure>
