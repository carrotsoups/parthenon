import os
import torch
import torchaudio
import soundfile as sf
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade
from lofi_audio_transformation import merge_audio, drums_transformation, bass_transformation, other_transformation

def music_seperation(wav_path, output_path, song):
    print("running separation")

    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sample_rate = bundle.sample_rate
    print(f"sample rate: {sample_rate}")

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
        fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

        final = torch.zeros(batch, len(model.sources), channels, length, device=device)

        while start < length - overlap_frames:
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out

            if start == 0:
                fade.fade_in_len = overlap_frames
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0

        return final

    data, sr = sf.read(wav_path)
    waveform = torch.from_numpy(data.T).float()
    waveform = waveform.to(device)
    mixture = waveform

    segment = 10
    overlap = 0.1

    print("separating track..")
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()

    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap
    )[0]

    sources = sources * ref.std() + ref.mean()
    sources_list = model.sources
    audios = dict(zip(sources_list, list(sources)))

    if True:
        stem_dir = os.path.join(os.path.dirname(output_path), song + "_stems")
        os.makedirs(stem_dir, exist_ok=True)
        for stem_name, stem_audio in audios.items():
            stem_path = os.path.join(stem_dir, f"{stem_name}.wav")
            sf.write(stem_path, stem_audio.cpu().numpy().T, sample_rate)
            print(f"Saved stem: {stem_path}")
            

def musicSeperationForSheet(wav_path, output_path, song):
    print("Running music separation (for sheet)...")
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sample_rate = bundle.sample_rate

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
        fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

        final = torch.zeros(batch, len(model.sources), channels, length, device=device)

        while start < length - overlap_frames:
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out

            if start == 0:
                fade.fade_in_len = overlap_frames
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0

        return final

    # Load audio
    data, sr = sf.read(wav_path)
    waveform = torch.from_numpy(data.T).float().to(device)

    # Normalization
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()

    sources = separate_sources(model, waveform[None], device=device, segment=10, overlap=0.1)[0]
    sources = sources * ref.std() + ref.mean()
    audios = dict(zip(model.sources, list(sources)))

    return audios["vocals"].cpu(), audios["other"].cpu(), sample_rate

music_seperation("app/playlists/test/notlofied/APT.wav", "siolofi.wav", "song1")
