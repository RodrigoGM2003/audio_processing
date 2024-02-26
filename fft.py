import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pyaudio
import utils
import wave
import time

p = pyaudio.PyAudio()
audio = wave.open("./audios/live_record.wav", "rb")

frames = audio.readframes(audio.getnframes())
frames = np.frombuffer(frames, dtype=np.int16)
audio.close()

def calculate_spectrogram(signal, fs, window_size, overlap):
    # Calculate the step size and number of segments
    step_size = window_size - overlap
    n_segments = (len(signal) - window_size) // step_size + 1

    # Initialize the spectrogram with zeros
    spectrogram = np.zeros((window_size // 2 + 1, n_segments))

    # Apply a Hamming window function
    window = np.hamming(window_size)

    # Calculate the spectrogram
    for i in range(n_segments):
        start = i * step_size
        end = start + window_size
        segment = signal[start:end] * window
        _, _, Z = stft(segment, fs, nperseg=window_size)
        Z_resized = np.resize(Z, (window_size // 2 + 1, Z.shape[1]))  # Resize Z to match the expected size

        # spectrogram[:, i] = np.mean(10 * np.log10(np.abs(Z)**2), axis=1)
        spectrogram[:, i] = np.mean(10 * np.log10(np.abs(Z_resized)**2), axis=1)
        # print(spectrogram.shape, flush=True)
        
    return spectrogram



# def calculate_spectrogram(signal, fs, window_size, overlap):
#     # Calculate the step size and number of segments
#     step_size = window_size - overlap
#     n_segments = (len(signal) - window_size) // step_size + 1

#     # Initialize the spectrogram with zeros
#     spectrogram = np.zeros((window_size // 2 + 1, n_segments))

#     # Apply a Hamming window function
#     window = np.hamming(window_size)

#     # Calculate the spectrogram
#     for i in range(n_segments):
#         start = i * step_size
#         end = start + window_size
#         segment = signal[start:end] * window
#         _, _, Z = stft(segment, fs, nperseg=window_size)
#         spectrogram[:, i] = np.mean(10 * np.log10(np.abs(Z)**2), axis=1)
#     return spectrogram


NFFT = 3200
Fs = int(audio.getframerate())

# print(frames.shape)

#calculate the difference in time consumed by both functions

start_time = time.time()
spectrogram_1 = calculate_spectrogram(frames, Fs, NFFT, int(NFFT // 1.5))
end_time = time.time()

print("Time consumed by small overlap: ", end_time - start_time)
print("Shape: ", spectrogram_1.shape)
print("Memory: ", spectrogram_1.nbytes / 1024**2, "MB")

start_time = time.time()
spectrogram_2 = calculate_spectrogram(frames, Fs, NFFT, int(NFFT // 1.1))
end_time = time.time()

print("Time consumed by big overlap: ", end_time - start_time)
print("Shape: ", spectrogram_2.shape)
print("Memory: ", spectrogram_2.nbytes / 1024**2, "MB")


# spectrogram = spectrogram[200, :]
# print(spectrogram.shape)


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
# # ax[0].set_ylim(0, 10000)
# ax[1].specgram(frames, NFFT=NFFT, Fs=Fs, noverlap=900)
# ax[1].set_title('Plt´s Spectrogram of Live Audio')
# ax[1].set_xlabel('Time')
# ax[1].set_ylabel('Frequency')
# ax[1].set_ylim(0, 10000)
plt.show()

