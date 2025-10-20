import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from generator import SignalGenerator


def plot_tone_validation():
    gen = SignalGenerator(sample_rate=10e3)

    tone = gen.generate_tone(frequency=1e3, duration=0.1, amplitude=0.7)
    time = np.arange(len(tone)) / gen.sample_rate

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Time domain plots
    axes[0, 0].plot(time * 1000, np.real(tone))
    axes[0, 0].set_title("Real Component")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].grid(True)

    axes[0, 1].plot(time * 1000, np.imag(tone))
    axes[0, 1].set_title("Imaginary Component")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].grid(True)

    # Frequency domain
    fft_data = np.fft.fft(tone)
    freqs = np.fft.fftfreq(len(tone), 1 / gen.sample_rate)
    axes[1, 0].plot(freqs / 1000, 20 * np.log10(np.abs(fft_data)))
    axes[1, 0].set_title("Frequency Spectrum")
    axes[1, 0].set_xlabel("Frequency (kHz)")
    axes[1, 0].set_xlim(-2, 2)
    axes[1, 0].grid(True)

    # Constellation
    axes[1, 1].scatter(np.real(tone), np.imag(tone), alpha=0.5, s=1)
    axes[1, 1].set_title("IQ Constellation")
    axes[1, 1].set_xlabel("I Component")
    axes[1, 1].set_ylabel("Q Component")
    axes[1, 1].grid(True)
    axes[1, 1].axis("equal")

    plt.tight_layout()
    plt.savefig("output/tone_validation.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_am_validation():
    gen = SignalGenerator(sample_rate=100e3)

    am_signal = gen.generate_am(
        carrier_freq=20e3, message_freq=1e3, modulation_index=0.7, duration=0.01
    )
    time = np.arange(len(am_signal)) / gen.sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # AM signal with envelope
    axes[0].plot(time * 1000, np.real(am_signal), label="AM Signal", alpha=0.7)
    axes[0].plot(time * 1000, np.abs(am_signal), "r-", label="Envelope", linewidth=2)
    axes[0].set_title("AM Signal with Envelope")
    axes[0].set_xlabel("Time (ms)")
    axes[0].legend()
    axes[0].grid(True)

    # AM spectrum
    fft_data = np.fft.fft(am_signal)
    freqs = np.fft.fftfreq(len(am_signal), 1 / gen.sample_rate) / 1000
    axes[1].plot(freqs, 20 * np.log10(np.abs(fft_data)))
    axes[1].set_title("AM Spectrum")
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_xlim(15, 25)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("output/am_validation.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_fm_validation():
    gen = SignalGenerator(sample_rate=500e3)

    fm_signal = gen.generate_fm(
        carrier_freq=100e3, message_freq=2e3, modulation_index=3.0, duration=0.005
    )
    time = np.arange(len(fm_signal)) / gen.sample_rate

    # Extract instantaneous frequency
    phase = np.unwrap(np.angle(fm_signal))
    instantaneous_freq = np.diff(phase) * gen.sample_rate / (2 * np.pi)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # FM signal
    axes[0].plot(time * 1000, np.real(fm_signal))
    axes[0].set_title("FM Signal")
    axes[0].set_xlabel("Time (ms)")
    axes[0].grid(True)

    # Frequency variation
    axes[1].plot(time[1:] * 1000, instantaneous_freq / 1000)
    axes[1].axhline(y=100, color="r", linestyle="--", label="Carrier")
    axes[1].set_title("Instantaneous Frequency")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Frequency (kHz)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("output/fm_validation.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_chirp_spectrogram_preview():
    gen = SignalGenerator(sample_rate=100e3)

    chirp = gen.generate_chirp(start_freq=1e3, end_freq=10e3, duration=0.1)

    plt.figure(figsize=(10, 6))
    plt.specgram(chirp, Fs=gen.sample_rate, NFFT=1024, noverlap=512)
    plt.title("Chirp Signal Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.savefig("output/chirp_spectrogram.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("Generating signal validation plots...")

    plot_tone_validation()
    plot_am_validation()
    plot_fm_validation()
    plot_chirp_spectrogram_preview()

    print("All validation plots generated")
