import os
from helper import separate_and_save_stems, transform_saved_stems
from lofi_audio_transformation import drums_transformation, bass_transformation, other_transformation

def transform_to_lofi(input_path, output_dir, name, logger=print):
    """
    Separates audio into stems and applies lo-fi transformations.
    All internal logs are sent through the `logger` function.
    """
    stem_dir = output_dir
    if not os.path.exists(stem_dir) or not any(f.endswith(".wav") for f in os.listdir(stem_dir)):
        logger(f"[LOFI] Separating stems for {name}...")
        separate_and_save_stems(input_path, stem_dir, name, logger=logger)
        logger(f"[LOFI] Stem separation complete.")
    lofi_output_path = os.path.join(os.path.dirname(input_path), f"{name}_lofi.wav")
    if os.path.exists(lofi_output_path):
        logger(f"[LOFI] Lo-fi track already exists: {lofi_output_path}")
        return

    logger(f"[LOFI] Applying lo-fi transformations...")
    transform_saved_stems(
        stem_dir=stem_dir,
        output_path=lofi_output_path,
        transformations={
            "bass": lambda w, sr, tb, logger=logger: bass_transformation(w, sr, tb, logger=logger),
            "drums": lambda w, sr, tb, logger=logger: drums_transformation(w, sr, tb, logger=logger),
            "other": lambda w, sr, tb, logger=logger: other_transformation(w, sr, tb, logger=logger),
            "vocals": lambda w, sr, tb, logger=logger: (w, sr)  # keep vocals unchanged
        },
        sr=44100,
        target_sample_rate=44100,
        target_bpm=75,
        logger=logger
    )
    logger(f"[LOFI] Lo-fi transformation saved to {lofi_output_path}")
