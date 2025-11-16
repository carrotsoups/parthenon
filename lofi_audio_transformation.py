from torchaudio.functional import highpass_biquad
from pedalboard import (Pedalboard, Reverb, Gain, Compressor,
                        HighpassFilter, HighShelfFilter, PitchShift, LowShelfFilter)
from helper import transformation
import numpy as np
import torch


def drums_transformation(waveform, sample_rate, target_bpm,logger=print):

    # Convert to mono numpy
    waveform_np = waveform.squeeze().cpu().numpy().astype(np.float32)
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=0)


    pitched = PitchShift(semitones=10)(waveform_np, float(sample_rate))

    if isinstance(pitched, tuple):
        pitched = pitched[0]  # Only take the processed audio
    pitched_tensor = torch.from_numpy(pitched[None, :]).float()


    board = Pedalboard([
        Gain(gain_db=-4),
        HighpassFilter(cutoff_frequency_hz=80),
        Compressor(threshold_db=-28, ratio=4.0, attack_ms=1, release_ms=20)
        #Reverb(room_size=0.6, damping=0.8, wet_level=0.35)
    ])

    return transformation(pitched_tensor, sample_rate,
                          target_sample_rate=20000,
                          lowpass_cutoff_freq=600,
                          board=board,
                          target_bpm=target_bpm)
    

def bass_transformation(waveform, sample_rate, target_bpm,logger=print):

    # Only highpass ONCE
    waveform = highpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=100)

    board = Pedalboard([
        Gain(gain_db=6),
        LowShelfFilter(cutoff_frequency_hz=600, gain_db=-14),
        HighpassFilter(cutoff_frequency_hz=200),
        HighShelfFilter(cutoff_frequency_hz=2000, gain_db=-6),
        Compressor(threshold_db=-22, ratio=2.0, attack_ms=10, release_ms=200),
        Reverb(room_size=0.3)
    ])

    return transformation(waveform, sample_rate,
                          target_sample_rate=20000,
                          lowpass_cutoff_freq=2000,
                          board=board,
                          target_bpm=target_bpm)



def other_transformation(waveform, sample_rate, target_bpm,logger=print):

    waveform_np = waveform.squeeze().cpu().numpy().astype(np.float32)
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=0)

    pitched = PitchShift(semitones=10)(waveform_np, float(sample_rate))

    if isinstance(pitched, tuple):
        pitched = pitched[0]  # Only take the processed audio
    pitched_tensor = torch.from_numpy(pitched[None, :]).float()


    board = Pedalboard([
        Gain(gain_db=-10), 
        HighpassFilter(cutoff_frequency_hz=150),
        Compressor(threshold_db=-36, ratio=5),
        Compressor(threshold_db=-25, ratio=3.5),
        Compressor(threshold_db=-12, ratio=2),
        Reverb(room_size=0.5, damping=0.8, wet_level=0.5)
    ])

    return transformation(pitched_tensor, sample_rate,
                          target_sample_rate=20000,
                          lowpass_cutoff_freq=1400,
                          board=board,
                          target_bpm=target_bpm)
