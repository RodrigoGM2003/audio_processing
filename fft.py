import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pyaudio
import utils
import wave
import time


# Load the audio file
p = pyaudio.PyAudio()
audio = wave.open("./audios/record.wav", "rb")

# Read the audio frames
frames = audio.readframes(audio.getnframes())
frames = np.frombuffer(frames, dtype=np.int16)
audio.close()

# Parameters
NFFT = 3200
Fs = int(audio.getframerate())

# Calculate the spectrogram with lower resolution
start_time = time.time()
spectrogram_1 = utils.calculate_spectrogram(frames, Fs, NFFT, int(NFFT // 1.5))
end_time = time.time()

print("Time consumed by small overlap: ", end_time - start_time)
print("Shape: ", spectrogram_1.shape)
print("Memory: ", spectrogram_1.nbytes / 1024**2, "MB")

# Calculate the spectrogram with higher resolution
start_time = time.time()
spectrogram_2 = utils.calculate_spectrogram(frames, Fs, NFFT, int(NFFT // 1.1))
end_time = time.time()

print("Time consumed by big overlap: ", end_time - start_time)
print("Shape: ", spectrogram_2.shape)
print("Memory: ", spectrogram_2.nbytes / 1024**2, "MB")


# Plot the spectrograms
fig, ax = plt.subplots(2, 1, figsize=(15, 15))
ax[0].imshow(spectrogram_1, aspect='auto', origin='lower', extent=[0, len(frames) / Fs, 0, Fs / 2])
ax[0].set_title('My Spectrogram of Live Audio')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Frequency')
ax[0].set_ylim(0, 7000)

ax[1].imshow(spectrogram_2, aspect='auto', origin='lower', extent=[0, len(frames) / Fs, 0, Fs / 2])
ax[1].set_title('My Spectrogram of Live Audio')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Frequency')
ax[1].set_ylim(0, 7000)

plt.show()