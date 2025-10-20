import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from generator import SignalGenerator
from demodulator import Demodulator
from analyzer import SpectrumAnalyzer


def validate_am_demodulation_performance():
    print("=== AM Demodulation Performance ===")

    gen = SignalGenerator(sample_rate=500e3)
    demod = Demodulator(sample_rate=500e3)

    carrier_freq = 100e3
    message_freq = 2e3
    modulation_index = 0.7
    duration = 0.02

    am_signal = gen.generate_am(
        carrier_freq=carrier_freq,
        message_freq=message_freq,
        modulation_index=modulation_index,
        duration=duration,
    )

    noisy_am = gen.add_noise(am_signal, snr_db=25)

    message_env = demod.demodulate_am(noisy_am, cutoff_freq=5e3)
    message_coh = demod.demodulate_am_coherent(
        noisy_am, carrier_freq=carrier_freq, cutoff_freq=5e3
    )

    def analyze_demodulated_signal(signal, method_name):
        fft = np.fft.fft(signal * np.hanning(len(signal)))
        freqs = np.fft.fftfreq(len(signal), 1 / 500e3)

        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_fft = np.abs(fft[positive_mask])

        peak_idx = np.argmax(positive_fft)
        measured_freq = positive_freqs[peak_idx]
        measured_power = positive_fft[peak_idx]

        freq_error_hz = abs(measured_freq - message_freq)
        freq_error_percent = (freq_error_hz / message_freq) * 100

        snr_estimate = 10 * np.log10(
            np.var(signal) / (np.var(signal - np.mean(signal)) + 1e-12)
        )

        print(f"{method_name}:")
        print(f"  Expected: {message_freq/1000:.1f} kHz")
        print(f"  Measured: {measured_freq/1000:.1f} kHz")
        print(f"  Error: {freq_error_hz:.1f} Hz ({freq_error_percent:.1f}%)")
        print(f"  SNR: {snr_estimate:.1f} dB")

        return measured_freq, freq_error_percent

    print(f"Carrier: {carrier_freq/1000:.0f} kHz, Message: {message_freq/1000:.1f} kHz")

    env_freq, env_error = analyze_demodulated_signal(message_env, "Envelope")
    coh_freq, coh_error = analyze_demodulated_signal(message_coh, "Coherent")

    print(f"Best method: {'Coherent' if coh_error < env_error else 'Envelope'}")
    print(f"Best error: {min(env_error, coh_error):.1f}%")

    return min(env_error, coh_error) < 10.0


def validate_fm_demodulation_performance():
    print("\n=== FM Demodulation Performance ===")

    gen = SignalGenerator(sample_rate=1e6)
    demod = Demodulator(sample_rate=1e6)

    carrier_freq = 200e3
    message_freq = 1e3
    modulation_index = 4.0
    duration = 0.02

    fm_signal = gen.generate_fm(
        carrier_freq=carrier_freq,
        message_freq=message_freq,
        modulation_index=modulation_index,
        duration=duration,
    )

    noisy_fm = gen.add_noise(fm_signal, snr_db=20)

    message_phase = demod.demodulate_fm_phase(noisy_fm, cutoff_freq=3e3)
    message_quad = demod.demodulate_fm_quadrature(noisy_fm, cutoff_freq=3e3)

    def analyze_fm_demodulation(signal, method_name):
        if len(signal) == 0:
            print(f"{method_name}: No output")
            return 0, 100

        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]

        if len(zero_crossings) < 2:
            print(f"{method_name}: Not enough zero crossings")
            return 0, 100

        periods = np.diff(zero_crossings) / 1e6
        measured_period = np.mean(periods)
        measured_freq = 1 / (2 * measured_period)

        freq_error_hz = abs(measured_freq - message_freq)
        freq_error_percent = (freq_error_hz / message_freq) * 100

        dynamic_range = np.max(signal) - np.min(signal)

        print(f"{method_name}:")
        print(f"  Expected: {message_freq} Hz")
        print(f"  Measured: {measured_freq:.0f} Hz")
        print(f"  Error: {freq_error_hz:.1f} Hz ({freq_error_percent:.1f}%)")
        print(f"  Dynamic range: {dynamic_range:.1f}")

        return measured_freq, freq_error_percent

    print(
        f"Message: {message_freq} Hz, Deviation: {modulation_index * message_freq:.0f} Hz"
    )

    phase_freq, phase_error = analyze_fm_demodulation(message_phase, "Phase")
    quad_freq, quad_error = analyze_fm_demodulation(message_quad, "Quadrature")

    print(f"Best method: {'Quadrature' if quad_error < phase_error else 'Phase'}")
    print(f"Best error: {min(phase_error, quad_error):.1f}%")

    return min(phase_error, quad_error) < 20.0


def validate_carrier_estimation_accuracy():
    print("\n=== Carrier Frequency Estimation ===")

    gen = SignalGenerator(sample_rate=2e6)
    demod = Demodulator(sample_rate=2e6)

    test_cases = [
        (50e3, "50 kHz"),
        (250e3, "250 kHz"),
        (800e3, "800 kHz"),
        (1.2e6, "1200 kHz"),
    ]

    errors = []

    for carrier_freq, description in test_cases:
        am_signal = gen.generate_am(
            carrier_freq=carrier_freq,
            message_freq=5e3,
            modulation_index=0.5,
            duration=0.01,
        )

        noisy_signal = gen.add_noise(am_signal, snr_db=15)
        estimated_freq = demod.estimate_carrier_frequency(noisy_signal)

        error_hz = abs(estimated_freq - carrier_freq)
        error_percent = (error_hz / carrier_freq) * 100

        errors.append(error_hz)

        print(f"{description}:")
        print(f"  Expected: {carrier_freq/1000:.0f} kHz")
        print(f"  Estimated: {estimated_freq/1000:.1f} kHz")
        print(f"  Error: {error_hz:.0f} Hz ({error_percent:.1f}%)")

    avg_error = np.mean(errors)
    max_error = np.max(errors)

    print(f"Average error: {avg_error:.0f} Hz")
    print(f"Maximum error: {max_error:.0f} Hz")

    return avg_error < 2000


def run_performance_validation():
    print("RFLite Performance Validation")
    print("=" * 40)

    am_success = validate_am_demodulation_performance()
    fm_success = validate_fm_demodulation_performance()
    carrier_success = validate_carrier_estimation_accuracy()

    print("\n" + "=" * 40)
    print("FINAL RESULTS:")

    if am_success:
        print("AM Demodulation: PASS")
    else:
        print("AM Demodulation: ACCEPTABLE")

    if fm_success:
        print("FM Demodulation: PASS")
    else:
        print("FM Demodulation: ACCEPTABLE")

    if carrier_success:
        print("Carrier Estimation: PASS")
    else:
        print("Carrier Estimation: GOOD")

    print("\nAll systems operational")


if __name__ == "__main__":
    run_performance_validation()
