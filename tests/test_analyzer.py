import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from generator import SignalGenerator
from analyzer import SpectrumAnalyzer


def test_psd_accuracy():
    print("=== PSD Accuracy Test ===")

    gen = SignalGenerator(sample_rate=1e6)
    analyzer = SpectrumAnalyzer(sample_rate=1e6)

    tone = gen.generate_tone(frequency=100e3, duration=0.01)
    freqs, psd = analyzer.compute_psd(tone, nfft=1024)

    peak_idx = np.argmax(psd)
    measured_freq = freqs[peak_idx]
    peak_power = psd[peak_idx]

    error_hz = abs(measured_freq - 100e3)
    error_percent = (error_hz / 100e3) * 100

    print(f"Target frequency: 100.0 kHz")
    print(f"Measured frequency: {measured_freq/1000:.1f} kHz")
    print(f"Peak power: {peak_power:.1f} dB")
    print(f"Frequency error: {error_hz:.0f} Hz ({error_percent:.2f}%)")

    assert np.isclose(measured_freq, 100e3, rtol=0.01)
    print("✓ PSD accuracy: PASS\n")


def test_spectrogram_time_frequency():
    print("=== Spectrogram Test ===")

    gen = SignalGenerator(sample_rate=100e3)
    analyzer = SpectrumAnalyzer(sample_rate=100e3)

    chirp = gen.generate_chirp(start_freq=1e3, end_freq=10e3, duration=0.1)
    f, t, Sxx = analyzer.compute_spectrogram(chirp, nfft=256, overlap=128)

    print(f"Time bins: {len(t)}, Frequency bins: {len(f)}")
    print(f"Spectrogram shape: {Sxx.shape}")
    print(f"Time range: {t[0]:.3f}s to {t[-1]:.3f}s")
    print(f"Frequency range: {f[0]/1000:.1f} kHz to {f[-1]/1000:.1f} kHz")

    assert len(t) > 0
    assert len(f) > 0
    assert Sxx.shape == (len(f), len(t))
    print("✓ Spectrogram: PASS\n")


def test_snr_estimation():
    print("=== SNR Estimation Test ===")

    gen = SignalGenerator(sample_rate=1e6)
    analyzer = SpectrumAnalyzer(sample_rate=1e6)

    clean_tone = gen.generate_tone(100e3, duration=0.01)
    snr_clean = analyzer.estimate_snr(clean_tone, signal_bw=10e3)

    print(f"Clean signal SNR: {snr_clean:.1f} dB")

    noisy_signal = gen.add_noise(clean_tone, snr_db=10.0)
    snr_noisy = analyzer.estimate_snr(noisy_signal, signal_bw=10e3)

    print(f"Noisy signal SNR: {snr_noisy:.1f} dB")
    print(f"SNR degradation: {snr_clean - snr_noisy:.1f} dB")

    assert snr_clean > 0
    assert np.isfinite(snr_clean)
    assert snr_noisy < snr_clean
    print("✓ SNR estimation: PASS\n")


def test_peak_detection():
    print("=== Peak Detection Test ===")

    gen = SignalGenerator(sample_rate=2e6)
    analyzer = SpectrumAnalyzer(sample_rate=2e6)

    duration = 0.01
    t = np.arange(0, duration, 1 / 2e6)

    tone1 = 0.8 * np.exp(1j * 2 * np.pi * 100e3 * t)
    tone2 = 0.5 * np.exp(1j * 2 * np.pi * 300e3 * t)
    tone3 = 0.3 * np.exp(1j * 2 * np.pi * 500e3 * t)

    multi_tone = tone1 + tone2 + tone3
    peak_freqs, peak_mags = analyzer.find_peaks(multi_tone, height=-50)

    print(f"Found {len(peak_freqs)} spectral peaks")

    for i, (freq, mag) in enumerate(zip(peak_freqs[:3], peak_mags[:3])):
        print(f"Peak {i+1}: {freq/1000:.1f} kHz, {mag:.1f} dB")

    expected_freqs = [100e3, 300e3, 500e3]
    for i, expected in enumerate(expected_freqs):
        closest = min(peak_freqs, key=lambda x: abs(x - expected))
        error = abs(closest - expected)
        print(
            f"Tone {i+1}: expected {expected/1000:.0f} kHz, found {closest/1000:.1f} kHz, error {error:.0f} Hz"
        )

    assert len(peak_freqs) >= 3
    print("✓ Peak detection: PASS\n")


def run_analyzer_tests():
    print("RFLite Spectrum Analyzer Test Suite")
    print("=" * 40)

    test_psd_accuracy()
    test_spectrogram_time_frequency()
    test_snr_estimation()
    test_peak_detection()

    print("=" * 40)
    print("SUMMARY: All analyzer tests completed successfully")
    print("Frequency analysis is operational and accurate")


if __name__ == "__main__":
    run_analyzer_tests()
