import numpy as np
from scipy import signal


def design_lowpass_filter(cutoff_freq: float, sample_rate: float, order: int = 4):
    nyquist = sample_rate / 2
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def design_bandpass_filter(
    lowcut: float, highcut: float, sample_rate: float, order: int = 4
):
    nyquist = sample_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band", analog=False)
    return b, a


def normalize_power(signal: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    current_power = np.mean(np.abs(signal) ** 2)
    scale = np.sqrt(target_power / current_power)
    return signal * scale


def db_to_linear(db_value: float) -> float:
    return 10 ** (db_value / 10)


def linear_to_db(linear_value: float) -> float:
    return 10 * np.log10(linear_value)


def frequency_to_index(frequency: float, sample_rate: float, nfft: int) -> int:
    return int(round(frequency * nfft / sample_rate))
