import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from generator import SignalGenerator
from demodulator import Demodulator


def debug_am_demodulation():
    print("=== AM Demodulation Debug ===")

    gen = SignalGenerator(sample_rate=500e3)
    demod = Demodulator(sample_rate=500e3)

    duration = 0.01
    t = np.arange(0, duration, 1 / 500e3)

    modulation_index = 0.8
    original_message = modulation_index * np.cos(2 * np.pi * 1e3 * t)

    print(f"Original message: {len(original_message)} samples")
    print(f"Message frequency: 1 kHz")
    print(f"Message amplitude: {modulation_index}")

    am_signal = gen.generate_am(
        carrier_freq=100e3,
        message_freq=1e3,
        modulation_index=modulation_index,
        duration=duration,
    )

    print(f"AM signal: {len(am_signal)} samples")
    print(
        f"AM signal stats: mean={np.mean(np.real(am_signal)):.3f}, std={np.std(np.real(am_signal)):.3f}"
    )

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t[:500], original_message[:500])
    plt.title("Original Message (1 kHz cosine)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t[:500], np.real(am_signal[:500]))
    plt.plot(t[:500], np.abs(am_signal[:500]), "r--", label="Envelope")
    plt.title("AM Signal (carrier + envelope)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    message_env = demod.demodulate_am(am_signal, cutoff_freq=5e3)
    message_coh = demod.demodulate_am_coherent(
        am_signal, carrier_freq=100e3, cutoff_freq=5e3
    )

    print(f"Envelope demod: {len(message_env)} samples")
    print(f"Coherent demod: {len(message_coh)} samples")

    # Recover messages
    min_len = min(len(message_env), len(original_message))
    recovered_env = (message_env[:min_len] - 1) / modulation_index
    recovered_coh = message_coh[:min_len] - np.mean(message_coh[:min_len])

    plt.subplot(3, 1, 3)
    t_plot = t[:min_len]
    plt.plot(t_plot, recovered_env, label="Envelope Demod (Recovered)")
    plt.plot(t_plot, recovered_coh, label="Coherent Demod (Recovered)")
    plt.plot(t_plot, original_message[:min_len], "k--", label="Original", alpha=0.7)
    plt.title("Demodulated vs Original (Recovered Message)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("output/debug_am_demodulation.png")
    print("Saved debug plot to output/debug_am_demodulation.png")

    corr_env = np.corrcoef(original_message[:min_len], recovered_env)[0, 1]
    corr_coh = np.corrcoef(original_message[:min_len], recovered_coh)[0, 1]
    print(f"Envelope correlation with original: {corr_env:.3f}")
    print(f"Coherent correlation with original: {corr_coh:.3f}")

    return corr_env > 0.95 and corr_coh > 0.95


def debug_fm_demodulation():
    print("\n=== FM Demodulation Debug ===")

    gen = SignalGenerator(sample_rate=1e6)
    demod = Demodulator(sample_rate=1e6)

    duration = 0.01
    t = np.arange(0, duration, 1 / 1e6)

    modulation_index = 3.0
    message_freq = 500
    original_message = np.cos(2 * np.pi * message_freq * t)

    print(f"Original FM message: {message_freq} Hz cosine")

    fm_signal = gen.generate_fm(
        carrier_freq=200e3,
        message_freq=message_freq,
        modulation_index=modulation_index,
        duration=duration,
    )

    message_phase = demod.demodulate_fm_phase(fm_signal, cutoff_freq=3e3)
    message_quad = demod.demodulate_fm_quadrature(fm_signal, cutoff_freq=3e3)

    scaling = modulation_index * message_freq

    min_len_phase = min(len(message_phase), len(original_message))
    recovered_phase = message_phase[:min_len_phase] / scaling

    min_len_quad = min(len(message_quad), len(original_message))
    recovered_quad = message_quad[:min_len_quad] / scaling

    # Remove transient and DC for better correlation
    transient = 100
    if len(recovered_phase) > transient:
        recovered_phase = recovered_phase[transient:] - np.mean(
            recovered_phase[transient:]
        )
        original_phase = original_message[transient : len(recovered_phase) + transient]
    else:
        original_phase = original_message[: len(recovered_phase)]

    if len(recovered_quad) > transient:
        recovered_quad = recovered_quad[transient:] - np.mean(
            recovered_quad[transient:]
        )
        original_quad = original_message[transient : len(recovered_quad) + transient]
    else:
        original_quad = original_message[: len(recovered_quad)]

    print(
        f"Phase demod range (scaled): {np.min(recovered_phase):.3f} to {np.max(recovered_phase):.3f}"
    )
    print(
        f"Quadrature demod range (scaled): {np.min(recovered_quad):.6f} to {np.max(recovered_quad):.6f}"
    )

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t[:1000], original_message[:1000])
    plt.title("Original FM Message")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t[:1000], np.real(fm_signal[:1000]))
    plt.title("FM Signal")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    t_phase = np.arange(len(recovered_phase)) / 1e6
    t_quad = np.arange(len(recovered_quad)) / 1e6
    plt.plot(t_phase[:1000], recovered_phase[:1000], label="Phase Demod (Recovered)")
    plt.plot(t_quad[:1000], recovered_quad[:1000], label="Quadrature Demod (Recovered)")
    plt.plot(t[:1000], original_message[:1000], "k--", label="Original")
    plt.title("FM Demodulated (Recovered Message)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("output/debug_fm_demodulation.png")
    print("Saved debug plot to output/debug_fm_demodulation.png")

    corr_phase = (
        np.corrcoef(original_phase, recovered_phase)[0, 1]
        if len(recovered_phase) > 0
        else 0
    )
    corr_quad = (
        np.corrcoef(original_quad, recovered_quad)[0, 1]
        if len(recovered_quad) > 0
        else 0
    )
    print(f"Phase correlation: {corr_phase:.3f}")
    print(f"Quadrature correlation: {corr_quad:.3f}")

    return corr_phase > 0.9 and corr_quad > 0.9


def run_debug_tests():
    print("RFLite Demodulation Debug Suite")
    print("=" * 45)

    am_ok = debug_am_demodulation()
    fm_ok = debug_fm_demodulation()

    print("\n" + "=" * 45)
    print("DEBUG SUMMARY:")
    print(f"AM Demodulation: {'WORKING' if am_ok else 'BROKEN'}")
    print(f"FM Demodulation: {'WORKING' if fm_ok else 'BROKEN'}")


if __name__ == "__main__":
    run_debug_tests()
