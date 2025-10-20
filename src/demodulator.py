# src/demodulator.py
import numpy as np
from scipy import signal


class Demodulator:
    def __init__(self, sample_rate: float = 2.4e6):
        self.sample_rate = sample_rate

    def demodulate_am(
        self, iq_data: np.ndarray, cutoff_freq: float = 5e3
    ) -> np.ndarray:
        """Envelope detection with zero-phase filtering"""
        # Use real part for Hilbert (standard for real-valued AM)
        from scipy.signal import hilbert

        analytic_signal = hilbert(np.real(iq_data))
        envelope = np.abs(analytic_signal)

        # Remove DC offset
        envelope_ac = envelope - np.mean(envelope)

        # Zero-phase low-pass filter
        nyquist = self.sample_rate / 2
        b, a = signal.butter(4, cutoff_freq / nyquist, btype="low")
        message = signal.filtfilt(b, a, envelope_ac)
        return message

    def demodulate_am_coherent(
        self, iq_data: np.ndarray, carrier_freq: float, cutoff_freq: float = 5e3
    ) -> np.ndarray:
        """Coherent AM demodulation"""
        t = np.arange(len(iq_data)) / self.sample_rate
        lo = np.exp(-1j * 2 * np.pi * carrier_freq * t)
        baseband = iq_data * lo

        nyquist = self.sample_rate / 2
        b, a = signal.butter(4, cutoff_freq / nyquist, btype="low")
        message = signal.filtfilt(b, a, np.real(baseband))
        return message

    def demodulate_fm_quadrature(
        self, iq_data: np.ndarray, cutoff_freq: float = 5e3
    ) -> np.ndarray:
        """Robust quadrature FM demodulation"""
        I = np.real(iq_data)
        Q = np.imag(iq_data)
        dI = np.diff(I)
        dQ = np.diff(Q)

        numerator = I[1:] * dQ - Q[1:] * dI
        denominator = I[1:] ** 2 + Q[1:] ** 2

        # Avoid division by zero without adding bias
        message = np.zeros_like(numerator)
        valid = denominator > 1e-6
        message[valid] = numerator[valid] / denominator[valid]

        # Zero-phase low-pass filter
        nyquist = self.sample_rate / 2
        b, a = signal.butter(4, cutoff_freq / nyquist, btype="low")
        message_filtered = signal.filtfilt(b, a, message)
        return message_filtered

    def demodulate_fm_phase(
        self, iq_data: np.ndarray, cutoff_freq: float = 5e3
    ) -> np.ndarray:
        """FM demod via phase differentiation"""
        phase = np.unwrap(np.angle(iq_data))
        instantaneous_freq = np.diff(phase) * self.sample_rate / (2 * np.pi)
        freq_deviation = instantaneous_freq - np.mean(instantaneous_freq)

        nyquist = self.sample_rate / 2
        b, a = signal.butter(4, cutoff_freq / nyquist, btype="low")
        message = signal.filtfilt(b, a, freq_deviation)
        return message

    def estimate_carrier_frequency(self, iq_data: np.ndarray) -> float:
        from analyzer import SpectrumAnalyzer

        analyzer = SpectrumAnalyzer(self.sample_rate)
        peak_freqs, _ = analyzer.find_peaks(iq_data, height=-40)
        if len(peak_freqs) > 0:
            return peak_freqs[0]
        freqs, psd = analyzer.compute_psd(iq_data)
        return np.sum(freqs * 10 ** (psd / 10)) / np.sum(10 ** (psd / 10))
