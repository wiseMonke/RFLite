import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from generator import SignalGenerator
from demodulator import Demodulator
from analyzer import SpectrumAnalyzer


def plot_am_demodulation_pipeline():
    gen = SignalGenerator(sample_rate=500e3)
    demod = Demodulator(sample_rate=500e3)
    analyzer = SpectrumAnalyzer(sample_rate=500e3)

    duration = 0.02
    t = np.arange(0, duration, 1 / 500e3)

    message = 0.6 * np.sin(2 * np.pi * 1e3 * t) + 0.4 * np.sin(2 * np.pi * 3e3 * t)

    am_signal = gen.generate_am(
        carrier_freq=100e3, message_freq=1e3, modulation_index=0.8, duration=duration
    )

    noisy_am = gen.add_noise(am_signal, snr_db=20)

    message_env = demod.demodulate_am(noisy_am, cutoff_freq=5e3)
    message_coh = demod.demodulate_am_coherent(
        noisy_am, carrier_freq=100e3, cutoff_freq=5e3
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # AM signal with envelope
    axes[0, 0].plot(t * 1000, np.real(am_signal[: len(t)]), "b-", alpha=0.7)
    axes[0, 0].plot(
        t * 1000, np.abs(am_signal[: len(t)]), "r-", linewidth=2, label="Envelope"
    )
    axes[0, 0].set_title("AM Signal with Envelope")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Noisy AM signal
    axes[0, 1].plot(t * 1000, np.real(noisy_am[: len(t)]), "b-", alpha=0.7)
    axes[0, 1].plot(t * 1000, np.abs(noisy_am[: len(t)]), "r-", linewidth=2)
    axes[0, 1].set_title("Noisy AM Signal (20 dB SNR)")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].grid(True, alpha=0.3)

    # Envelope detection
    axes[1, 0].plot(t * 1000, message_env[: len(t)], "g-", linewidth=2)
    axes[1, 0].set_title("Envelope Detection")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].grid(True, alpha=0.3)

    # Coherent detection
    axes[1, 1].plot(t * 1000, message_coh[: len(t)], "g-", linewidth=2)
    axes[1, 1].set_title("Coherent Detection")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[1, 1].grid(True, alpha=0.3)

    # AM spectrum
    freqs, psd_am = analyzer.compute_psd(am_signal)
    axes[2, 0].plot(freqs / 1000, psd_am)
    axes[2, 0].set_title("AM Spectrum")
    axes[2, 0].set_xlabel("Frequency (kHz)")
    axes[2, 0].set_xlim(50, 150)
    axes[2, 0].grid(True, alpha=0.3)

    # Demodulated spectrum
    freqs_msg, psd_msg = analyzer.compute_psd(message_coh)
    axes[2, 1].plot(freqs_msg / 1000, psd_msg)
    axes[2, 1].set_title("Demodulated Spectrum")
    axes[2, 1].set_xlabel("Frequency (kHz)")
    axes[2, 1].set_xlim(-10, 10)
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/am_demodulation_pipeline.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_fm_demodulation_pipeline():
    gen = SignalGenerator(sample_rate=1e6)
    demod = Demodulator(sample_rate=1e6)
    analyzer = SpectrumAnalyzer(sample_rate=1e6)

    duration = 0.01
    t = np.arange(0, duration, 1 / 1e6)

    message = 0.7 * np.sin(2 * np.pi * 1e3 * t) + 0.3 * np.sin(2 * np.pi * 2e3 * t)

    fm_signal = gen.generate_fm(
        carrier_freq=200e3, message_freq=1e3, modulation_index=4.0, duration=duration
    )

    noisy_fm = gen.add_noise(fm_signal, snr_db=25)

    message_phase = demod.demodulate_fm_phase(noisy_fm, cutoff_freq=5e3)
    message_quad = demod.demodulate_fm_quadrature(noisy_fm, cutoff_freq=5e3)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # FM signal
    axes[0, 0].plot(t * 1000, np.real(fm_signal), "b-", alpha=0.7)
    axes[0, 0].set_title("FM Signal")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].grid(True, alpha=0.3)

    # Noisy FM
    axes[0, 1].plot(t * 1000, np.real(noisy_fm), "b-", alpha=0.7)
    axes[0, 1].set_title("Noisy FM (25 dB SNR)")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].grid(True, alpha=0.3)

    # Phase demodulation
    t_phase = np.arange(len(message_phase)) / 1e6 * 1000
    axes[1, 0].plot(t_phase, message_phase, "g-", linewidth=2)
    axes[1, 0].set_title("Phase Demodulation")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].grid(True, alpha=0.3)

    # Quadrature demodulation
    t_quad = np.arange(len(message_quad)) / 1e6 * 1000
    axes[1, 1].plot(t_quad, message_quad, "g-", linewidth=2)
    axes[1, 1].set_title("Quadrature Demodulation")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[1, 1].grid(True, alpha=0.3)

    # FM spectrum
    freqs, psd_fm = analyzer.compute_psd(fm_signal)
    axes[2, 0].plot(freqs / 1000, psd_fm)
    axes[2, 0].set_title("FM Spectrum")
    axes[2, 0].set_xlabel("Frequency (kHz)")
    axes[2, 0].set_xlim(150, 250)
    axes[2, 0].grid(True, alpha=0.3)

    # Demodulated spectrum
    freqs_msg, psd_msg = analyzer.compute_psd(message_phase)
    axes[2, 1].plot(freqs_msg / 1000, psd_msg)
    axes[2, 1].set_title("Demodulated Spectrum")
    axes[2, 1].set_xlabel("Frequency (kHz)")
    axes[2, 1].set_xlim(-10, 10)
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/fm_demodulation_pipeline.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("Generating demodulator validation plots...")

    plot_am_demodulation_pipeline()
    plot_fm_demodulation_pipeline()

    print("All demodulator plots generated")
