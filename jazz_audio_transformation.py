from pedalboard import Pedalboard, Reverb, Gain, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter
from helper import transformation
def drums_transformation(waveform, sample_rate):
    board = Pedalboard([
        Gain(gain_db=1),  # Slight boost
        HighpassFilter(cutoff_frequency_hz=60),  # Clean sub rumble
        Reverb(room_size=0.3, damping=0.5, wet_level=0.25),  # Natural jazz room
        Compressor(threshold_db=-20, ratio=2.0, attack_ms=10, release_ms=150)  # Soft dynamics
    ])
    processed, sample_rate = transformation(waveform, sample_rate, target_sample_rate=16000, lowpass_cutoff_freq=18000, board=board)
    return processed, sample_rate


def bass_transformation(waveform, sample_rate):
    board = Pedalboard([
        LowShelfFilter(cutoff_frequency_hz=100, gain_db=2),  # Slight warmth
        HighShelfFilter(cutoff_frequency_hz=5000, gain_db=-3),  # Smooth highs
        Compressor(threshold_db=-18, ratio=2.0, attack_ms=10, release_ms=200),  # Gentle punch
        Reverb(room_size=0.2, damping=0.4, wet_level=0.2)  # Slight ambient
    ])
    processed, sample_rate = transformation(waveform, sample_rate, target_sample_rate=16000, lowpass_cutoff_freq=20000, board=board)
    return processed, sample_rate


def other_transformation(waveform, sample_rate):
    board = Pedalboard([
        Gain(gain_db=2),
        Reverb(room_size=0.4, damping=0.5, wet_level=0.3),
        HighpassFilter(cutoff_frequency_hz=80),
        Compressor(threshold_db=-25, ratio=2.5, attack_ms=5, release_ms=150)
    ])
    processed, sample_rate = transformation(waveform, sample_rate, target_sample_rate=16000, lowpass_cutoff_freq=20000, board=board)
    return processed, sample_rate

