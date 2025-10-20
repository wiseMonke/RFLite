import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from generator import SignalGenerator
from analyzer import SpectrumAnalyzer


def plot_psd_comparison():
    gen = SignalGenerator(sample_rate=2e6)
    analyzer = SpectrumAnalyzer(sample_rate=2e6)

    duration = 0.01

    # Generate different signal types
    tone = gen.generate_tone(500e3, duration, amplitude=1.0)

    am_signal = gen.generate_am(
        carrier_freq=500e3, message_freq=10e3, modulation_index=0.7, duration=duration
    )

    t = np.arange(0, duration, 1 / 2e6)
    multi_tone = (
        0.7 * np.exp(1j * 2 * np.pi * 300e3 * t)
        + 0.5 * np.exp(1j * 2 * np.pi * 600e3 * t)
        + 0.3 * np.exp(1j * 2 * np.pi * 900e3 * t)
    )

    noisy_tone = gen.add_noise(tone, snr_db=15)

    signals = [
        ("Pure Tone (500 kHz)", tone),
        ("AM Signal", am_signal),
        ("Multi-Tone", multi_tone),
        ("Noisy Tone (15 dB SNR)", noisy_tone),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (title, signal_data) in enumerate(signals):
        ax = axes[idx // 2, idx % 2]

        freqs, psd = analyzer.compute_psd(signal_data, nfft=1024)
        ax.plot(freqs / 1000, psd, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("PSD (dB)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1000, 1000)

        if "Noisy" in title:
            snr = analyzer.estimate_snr(signal_data)
            ax.text(
                0.05,
                0.95,
                f"SNR: {snr:.1f} dB",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

    plt.tight_layout()
    plt.savefig("output/psd_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_spectrogram_analysis():
    gen = SignalGenerator(sample_rate=200e3)
    analyzer = SpectrumAnalyzer(sample_rate=200e3)

    duration = 0.5
    t = np.arange(0, duration, 1 / 200e3)

    # Create complex signal with time-varying components
    chirp_phase = 2 * np.pi * (20e3 * t + (80e3 - 20e3) * t**2 / (2 * duration))
    chirp_component = 0.8 * np.exp(1j * chirp_phase)

    pulse_times = (t > 0.1) & (t < 0.3)
    pulse_component = 0.6 * np.exp(1j * 2 * np.pi * 40e3 * t) * pulse_times

    hop_freq = np.where(t < 0.4, 10e3, 60e3)
    hop_component = 0.5 * np.exp(1j * 2 * np.pi * np.cumsum(hop_freq) / 200e3)

    combined_signal = chirp_component + pulse_component + hop_component

    f, t_spec, Sxx = analyzer.compute_spectrogram(
        combined_signal, nfft=512, overlap=384
    )

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.pcolormesh(t_spec, f / 1000, Sxx, shading="gouraud", cmap="viridis")
    plt.colorbar(label="Power (dB)")
    plt.title("Time-Frequency Analysis")
    plt.ylabel("Frequency (kHz)")
    plt.ylim(0, 100)

    plt.subplot(2, 1, 2)
    plt.plot(t * 1000, np.real(combined_signal))
    plt.title("Time Domain")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/advanced_spectrogram.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_peak_detection_demo():
    gen = SignalGenerator(sample_rate=2e6)
    analyzer = SpectrumAnalyzer(sample_rate=2e6)

    duration = 0.02
    t = np.arange(0, duration, 1 / 2e6)

    components = [
        (800e3, 1.0),
        (750e3, 0.3),
        (850e3, 0.3),
        (600e3, 0.2),
        (950e3, 0.15),
    ]

    signal_sum = np.zeros_like(t, dtype=complex)
    for freq, amp in components:
        signal_sum += amp * np.exp(1j * 2 * np.pi * freq * t)

    noisy_signal = gen.add_noise(signal_sum, snr_db=25)

    peak_freqs, peak_mags = analyzer.find_peaks(noisy_signal, height=-40, distance=5)

    freqs, psd = analyzer.compute_psd(noisy_signal, nfft=2048)

    plt.figure(figsize=(12, 6))
    plt.plot(freqs / 1000, psd, "b-", linewidth=1, label="PSD")
    plt.plot(peak_freqs / 1000, peak_mags, "ro", markersize=8, label="Peaks")

    for freq, mag in zip(peak_freqs, peak_mags):
        plt.annotate(
            f"{freq/1000:.0f} kHz\n{mag:.1f} dB",
            xy=(freq / 1000, mag),
            xytext=(10, 10),
            textcoords="offset points",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    plt.title("Spectral Peak Detection")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("PSD (dB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(500, 1000)

    plt.tight_layout()
    plt.savefig("output/peak_detection_demo.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("Generating analyzer validation plots...")

    plot_psd_comparison()
    plot_spectrogram_analysis()
    plot_peak_detection_demo()

    print("All analyzer plots generated")
