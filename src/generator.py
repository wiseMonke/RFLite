import numpy as np
from typing import Optional, Tuple


class SignalGenerator:
    def __init__(self, sample_rate: float = 2.4e6):
        self.sample_rate = sample_rate

    def generate_tone(
        self, frequency: float, duration: float, amplitude: float = 1.0
    ) -> np.ndarray:
        # Complex exponential for IQ tone
        t = np.arange(0, duration, 1 / self.sample_rate)
        iq_signal = amplitude * np.exp(1j * 2 * np.pi * frequency * t)
        return iq_signal

    def generate_am(
        self,
        carrier_freq: float,
        message_freq: float,
        modulation_index: float = 0.5,
        duration: float = 1.0,
    ) -> np.ndarray:
        t = np.arange(0, duration, 1 / self.sample_rate)

        carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
        message = np.cos(2 * np.pi * message_freq * t)

        # AM: carrier * (1 + m*message)
        am_signal = carrier * (1 + modulation_index * message)
        return am_signal

    def generate_fm(
        self,
        carrier_freq: float,
        message_freq: float,
        modulation_index: float = 5.0,
        duration: float = 1.0,
    ) -> np.ndarray:
        t = np.arange(0, duration, 1 / self.sample_rate)

        message = np.cos(2 * np.pi * message_freq * t)
        frequency_deviation = modulation_index * message_freq

        # FM: phase = integral of instantaneous frequency
        instantaneous_phase = (
            2
            * np.pi
            * (
                carrier_freq * t
                + frequency_deviation * np.cumsum(message) / self.sample_rate
            )
        )

        fm_signal = np.exp(1j * instantaneous_phase)
        return fm_signal

    def add_noise(self, signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        # Add complex Gaussian noise for specified SNR
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
        )

        return signal + noise

    def generate_chirp(
        self, start_freq: float, end_freq: float, duration: float = 1.0
    ) -> np.ndarray:
        # Linear frequency sweep for spectrogram testing
        t = np.arange(0, duration, 1 / self.sample_rate)

        instantaneous_frequency = start_freq + (end_freq - start_freq) * t / duration
        phase = 2 * np.pi * np.cumsum(instantaneous_frequency) / self.sample_rate

        chirp_signal = np.exp(1j * phase)
        return chirp_signal
