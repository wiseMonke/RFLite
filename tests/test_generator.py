import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from generator import SignalGenerator


def test_basic_tone():
    print("=== Basic Tone Generation Test ===")

    gen = SignalGenerator(sample_rate=1e6)
    tone = gen.generate_tone(frequency=100e3, duration=0.01, amplitude=0.8)

    # Verify basic properties
    assert len(tone) == 10000  # 0.01s * 1e6 Hz
    assert np.isclose(np.max(np.abs(tone)), 0.8)
    assert np.iscomplexobj(tone)

    # Check frequency accuracy
    time = np.arange(len(tone)) / 1e6
    expected_phase = 2 * np.pi * 100e3 * time
    actual_phase = np.angle(tone)
    phase_error = np.std(np.unwrap(actual_phase) - expected_phase)

    print(f"Generated {len(tone)} complex samples")
    print(f"Peak amplitude: {np.max(np.abs(tone)):.3f} (target: 0.8)")
    print(f"Phase error std: {phase_error:.4f} radians")

    assert phase_error < 0.1
    print("✓ Basic tone: PASS\n")


def test_am_modulation():
    print("=== AM Modulation Test ===")

    gen = SignalGenerator(sample_rate=2.4e6)
    am_signal = gen.generate_am(
        carrier_freq=1e6, message_freq=10e3, modulation_index=0.5, duration=0.001
    )

    envelope = np.abs(am_signal)
    min_env, max_env = np.min(envelope), np.max(envelope)
    calculated_index = (max_env - min_env) / (max_env + min_env)

    print(f"Target modulation: 0.5")
    print(f"Measured modulation: {calculated_index:.3f}")
    print(f"Envelope range: {min_env:.3f} to {max_env:.3f}")

    assert np.isclose(calculated_index, 0.5, rtol=0.1)
    print("✓ AM modulation: PASS\n")


def test_fm_phase_behavior():
    print("=== FM Phase Integration Test ===")

    gen = SignalGenerator(sample_rate=2.4e6)
    fm_signal = gen.generate_fm(
        carrier_freq=1e6, message_freq=5e3, modulation_index=2.0, duration=0.001
    )

    phase = np.unwrap(np.angle(fm_signal))
    instantaneous_freq = np.diff(phase) * gen.sample_rate / (2 * np.pi)

    freq_variation = np.std(instantaneous_freq)
    expected_deviation = 2.0 * 5e3  # 10 kHz
    mean_freq = np.mean(instantaneous_freq)

    print(f"Expected deviation: {expected_deviation/1000:.1f} kHz")
    print(f"Measured frequency std: {freq_variation/1000:.1f} kHz")
    print(f"Mean frequency: {mean_freq/1000:.1f} kHz (target: 1000 kHz)")

    assert freq_variation > 0.5 * expected_deviation
    assert np.isclose(mean_freq, 1e6, rtol=0.01)
    print("✓ FM phase integration: PASS\n")


def test_noise_addition():
    print("=== Noise Addition Test ===")

    gen = SignalGenerator(sample_rate=1e6)
    clean_signal = gen.generate_tone(100e3, duration=0.01, amplitude=1.0)

    noisy_signal = gen.add_noise(clean_signal, snr_db=10.0)
    noise_component = noisy_signal - clean_signal

    signal_power = np.mean(np.abs(clean_signal) ** 2)
    noise_power = np.mean(np.abs(noise_component) ** 2)
    measured_snr_db = 10 * np.log10(signal_power / noise_power)

    print(f"Target SNR: 10.0 dB")
    print(f"Measured SNR: {measured_snr_db:.1f} dB")
    print(f"Signal power: {signal_power:.4f}")
    print(f"Noise power: {noise_power:.4f}")

    assert np.isclose(measured_snr_db, 10.0, atol=1.0)
    print("✓ Noise addition: PASS\n")


def run_all_tests():
    print("RFLite Signal Generator Test Suite")
    print("=" * 40)

    test_basic_tone()
    test_am_modulation()
    test_fm_phase_behavior()
    test_noise_addition()

    print("=" * 40)
    print("SUMMARY: All generator tests completed successfully")
    print("Signal generation foundation is solid")


if __name__ == "__main__":
    run_all_tests()
