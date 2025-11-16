import subprocess
import torch
import soundfile as sf
import os
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import numpy as np
import torch
from torchaudio.functional import lowpass_biquad, resample
from pedalboard import time_stretch
import numpy as np

import librosa

import librosa
import numpy as np

def detect_bpm_from_waveform(audio, sample_rate, logger=print):
    """
    waveform: torch.Tensor [channels, samples] or [samples]
    sample_rate: int
    """

    # If stereo → convert to mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)

    # Librosa BPM detection
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate)

    # Fix half-time/double-time issues
    if tempo > 160:
        tempo /= 2
    if tempo < 50:
        tempo *= 2
    logger(f"BMP: {tempo}")
    return float(tempo)

def mp3_to_wav(input_mp3, output_wav):
    subprocess.run([
        "ffmpeg",
        "-y", 
        "-i", input_mp3,
        "-ac", "2", 
        "-ar", "44100",
        "-sample_fmt","s16",
        output_wav
    ])

def separate_and_save_stems(wav_path, output_dir, song_name, logger=print):
    logger("Running separation...")

    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sample_rate = bundle.sample_rate

    # Helper function to process chunks
    def separate_sources(model, mix, segment=10.0, overlap=0.1, device=None):
        if device is None:
            device = mix.device
        else:
            device = torch.device(device)
        batch, channels, length = mix.shape
        chunk_len = int(sample_rate * segment * (1 + overlap))
        start = 0
        end = chunk_len
        overlap_frames = int(overlap * sample_rate)
        final = torch.zeros(batch, len(model.sources), channels, length, device=device)
        fade = torch.nn.functional.hardtanh  # Placeholder for fade function if needed

        while start < length - overlap_frames:
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = model.forward(chunk)
            final[:, :, :, start:end] += out
            start += int(chunk_len - overlap_frames)
            end += chunk_len
            if end >= length:
                break
        return final

    # Load audio
    data, sr = sf.read(wav_path)
    waveform = torch.from_numpy(data.T).float().to(device)
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()

    sources = separate_sources(model, waveform[None], device=device, segment=10, overlap=0.1)[0]
    sources = sources * ref.std() + ref.mean()

    # Save stems
    stem_dir = os.path.join(output_dir)
    os.makedirs(stem_dir, exist_ok=True)
    sources_list = model.sources
    for stem_name, stem_audio in zip(sources_list, sources):
        stem_path = os.path.join(stem_dir, f"{stem_name}.wav")
        sf.write(stem_path, stem_audio.cpu().numpy().T, sample_rate)
        logger(f"Saved stem: {stem_path}")

    return stem_dir, sample_rate


def transform_saved_stems(stem_dir, output_path, transformations, target_bpm, sr,target_sample_rate=44100, logger=print):
    """
    Apply transformations to saved stems and merge them.

    stem_dir: folder containing stem wavs named 'bass.wav','drums.wav','other.wav','vocals.wav'
    transformations: dict of {'bass': func, 'drums': func, 'other': func, 'vocals': func}
                     Each function must accept (waveform, sample_rate, target_bpm)
                     and return (processed_waveform, processed_sample_rate).
                     processed_waveform may be a numpy.ndarray (shape [channels,samples])
                     or a torch.Tensor (shape [channels,samples]).
    """
    logger("Applying transformations to saved stems...")

    # Load stems (keep per-stem sample rate sr)
    audios = {}
    sample_rates = {}
    for stem_name in ["bass", "drums", "other", "vocals"]:
        stem_path = os.path.join(stem_dir, f"{stem_name}.wav")
        if not os.path.exists(stem_path):
            raise FileNotFoundError(f"Stem file not found: {stem_path}")
        data, sr = sf.read(stem_path)           # data shape = [samples, channels] or [samples]
        # convert to shape [channels, samples] torch.Tensor for transformations
        arr = np.asarray(data).T
        audios[stem_name] = torch.from_numpy(arr).float()
        sample_rates[stem_name] = sr

    # Apply transformations (use actual sr from file)
    processed_stems = {}
    for stem_name, waveform in audios.items():
        sr = sample_rates[stem_name]
        if stem_name in transformations:
            # <-- PASS the original sr (not target_sample_rate) and the target_bpm
            processed_waveform,processed_sr = transformations[stem_name](waveform, sr, target_bpm)
            #logger(stem_name,processed_waveform)
            #processed_waveform, processed_sr = processed_waveform
        else:
            # No transform -> keep original waveform & sr
            processed_waveform, processed_sr = waveform[0], sr

        # Accept either numpy or torch; normalize dtype/shape
        if isinstance(processed_waveform, np.ndarray):
            # ensure shape [channels, samples]
            proc = torch.from_numpy(processed_waveform.astype(np.float32))
        elif isinstance(processed_waveform, torch.Tensor):
            proc = processed_waveform.float()
        else:
            raise TypeError(f"Unsupported processed waveform type for {stem_name}: {type(processed_waveform)}")

        processed_stems[stem_name] = (proc, int(processed_sr))

    # Merge stems (bass, drums, other)
    waveformb, sample_rateb = processed_stems["bass"]
    waveformd, sample_rated = processed_stems["drums"]
    waveformo, sample_rateo = processed_stems["other"]

    merged_waveform, merged_sample_rate = merge_audio(
        waveformb, waveformd, waveformo, sample_rateb, sample_rated, sample_rateo
    )

    # Optionally merge vocals back (if present)
    """if "vocals" in processed_stems:
        waveformv, sample_ratev = processed_stems["vocals"]
        # resample vocals to merged_sample_rate if necessary
        if sample_ratev != merged_sample_rate:
            waveformv = resample(waveformv, orig_freq=sample_ratev, new_freq=merged_sample_rate)
        min_len = min(merged_waveform.shape[-1], waveformv.shape[-1])
        merged_waveform = merged_waveform[..., :min_len] + waveformv[..., :min_len]"""

    # Normalize to prevent clipping (work on tensor)
    max_val = merged_waveform.abs().max()
    if max_val > 1.0:
        merged_waveform = merged_waveform / max_val

    # Save final output (convert to numpy [samples, channels])
    out_np = merged_waveform.cpu().numpy().T
    sf.write(output_path, out_np, merged_sample_rate)
    logger(f"Saved transformed track to {output_path}")
    return output_path


def transformation(waveform, sample_rate, target_sample_rate, lowpass_cutoff_freq, board, target_bpm, logger=print):
    logger(f"Processing waveform at {sample_rate} Hz")

    # --- Low-pass filter ---
    waveform = lowpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=lowpass_cutoff_freq)
    logger("Applied low-pass filter")

    # --- Resample to lower sample rate ---
    if sample_rate != target_sample_rate:
        waveform = resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)
        sample_rate = target_sample_rate
        logger(f"Resampled to {sample_rate} Hz")

    #stretch_factor = detect_bpm_from_waveform(waveform, sample_rate) / target_bpm  # Assuming 120 BPM is the original tempo
    # --- Convert to numpy for Pedalboard ---
    audio_numpy = waveform.squeeze().cpu().numpy()
    stretch_factor =  target_bpm / detect_bpm_from_waveform(audio_numpy, sample_rate)
    stretched_audio = time_stretch( # Slows down audio
        input_audio = audio_numpy,
        samplerate=sample_rate,
        stretch_factor = stretch_factor
    )
    
    # --- Apply effects ---
    processed = board(stretched_audio, sample_rate)

    # --- Save to output file ---

    # Ensure stereo (2D) format and float32 dtype
    if processed.ndim == 1:
        processed = np.expand_dims(processed, axis=0)  # [1, time]
    processed = processed.astype(np.float32)

    return processed, sample_rate


def merge_audio(waveformb, waveformd, waveformo, sample_rateb, sample_rated, sample_rateo):
    if isinstance(waveformb, np.ndarray):
        waveformb = torch.from_numpy(waveformb)
    if isinstance(waveformd, np.ndarray):
        waveformd = torch.from_numpy(waveformd)
    if isinstance(waveformo, np.ndarray):
        waveformo = torch.from_numpy(waveformo)

    target_sample_rate = min(sample_rateb, sample_rated, sample_rateo)

    if sample_rateb != target_sample_rate:
        waveformb = resample(waveformb, orig_freq=sample_rateb, new_freq=target_sample_rate)
    if sample_rated != target_sample_rate:
        waveformd = resample(waveformd, orig_freq=sample_rated, new_freq=target_sample_rate)
    if sample_rateo != target_sample_rate:
        waveformo = resample(waveformo, orig_freq=sample_rateo, new_freq=target_sample_rate)

    min_len = min(waveformb.shape[-1], waveformd.shape[-1], waveformo.shape[-1])
    waveformb = waveformb[..., :min_len]
    waveformd = waveformd[..., :min_len]
    waveformo = waveformo[..., :min_len]

    merged_waveform = waveformb + waveformd + waveformo
    max_val = merged_waveform.abs().max()
    if max_val > 1.0:
        merged_waveform = merged_waveform / max_val

    return merged_waveform, target_sample_rate