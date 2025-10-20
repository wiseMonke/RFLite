# examples/real_time_demo.py
"""
Simulated Real-Time Processing
Process signal in chunks like a real SDR stream.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from generator import SignalGenerator
from analyzer import SpectrumAnalyzer

fs = 200e3
gen = SignalGenerator(sample_rate=fs)
signal = gen.generate_chirp(10e3, 50e3, 0.1)  # 100 ms chirp

# Simulate real-time: process in 10 ms chunks
chunk_size = int(0.01 * fs)  # 10 ms
psd_list = []
analyzer = SpectrumAnalyzer(sample_rate=fs)

for i in range(0, len(signal), chunk_size):
    chunk = signal[i : i + chunk_size]
    if len(chunk) < chunk_size:
        continue
    _, psd = analyzer.compute_psd(chunk, nfft=512)
    psd_list.append(psd)

# Plot spectrogram-like view
psd_array = np.array(psd_list).T
plt.figure(figsize=(10, 4))
plt.imshow(psd_array, aspect="auto", cmap="viridis", origin="lower")
plt.colorbar(label="PSD (dB)")
plt.title("Simulated Real-Time Spectrogram (Chunked Processing)")
plt.xlabel("Time Chunk")
plt.ylabel("Frequency Bin")
plt.savefig("output/real_time_demo.png", dpi=150)
plt.show()
