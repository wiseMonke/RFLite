import numpy as np
from scipy import signal


class SpectrumAnalyzer:
    def __init__(self, sample_rate: float = 2.4e6):
        self.sample_rate = sample_rate

    def compute_psd(self, iq_data: np.ndarray, window: str = "hann", nfft: int = 1024):
        # Welch's method for smooth PSD
        freqs, psd = signal.welch(
            iq_data,
            fs=self.sample_rate,
            window=window,
            nperseg=nfft,
            return_onesided=False,
        )

        # Convert to dB and center frequencies
        psd_db = 10 * np.log10(psd)
        freqs_shifted = np.fft.fftshift(freqs)
        psd_shifted = np.fft.fftshift(psd_db)

        return freqs_shifted, psd_shifted

    def compute_spectrogram(
        self,
        iq_data: np.ndarray,
        window: str = "hann",
        nfft: int = 1024,
        overlap: int = 512,
    ):
        # STFT for time-frequency analysis
        f, t, Sxx = signal.stft(
            iq_data,
            fs=self.sample_rate,
            window=window,
            nperseg=nfft,
            noverlap=overlap,
            return_onesided=False,
        )

        # Convert to dB and center frequencies
        Sxx_db = 10 * np.log10(np.abs(Sxx))
        f_shifted = np.fft.fftshift(f)
        Sxx_shifted = np.fft.fftshift(Sxx_db, axes=0)

        return f_shifted, t, Sxx_shifted

    def estimate_snr(self, iq_data: np.ndarray, signal_bw: float = 10e3) -> float:
        freqs, psd_db = self.compute_psd(iq_data, nfft=1024)

        # Find strongest peak
        peak_idx = np.argmax(psd_db)
        peak_freq = freqs[peak_idx]

        # Signal region around peak
        signal_region = (freqs >= peak_freq - signal_bw / 2) & (
            freqs <= peak_freq + signal_bw / 2
        )

        # Power in linear scale
        psd_linear = 10 ** (psd_db / 10)
        total_power = np.sum(psd_linear)
        signal_power = np.sum(psd_linear[signal_region])
        noise_power = total_power - signal_power

        if noise_power <= 0:
            return 60.0

        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        return snr_db

    def measure_power(self, iq_data: np.ndarray) -> float:
        # RMS power to dBm
        rms_power = np.mean(np.abs(iq_data) ** 2)
        power_dbm = 10 * np.log10(rms_power / 1e-3)
        return power_dbm

    def find_peaks(self, iq_data: np.ndarray, height: float = -50, distance: int = 10):
        freqs, psd = self.compute_psd(iq_data)

        # Find spectral peaks
        peak_indices, properties = signal.find_peaks(
            psd, height=height, distance=distance
        )

        peak_freqs = freqs[peak_indices]
        peak_magnitudes = properties["peak_heights"]

        return peak_freqs, peak_magnitudes
