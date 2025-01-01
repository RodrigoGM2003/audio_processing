import numpy as np
from collections import deque
import utils 
import pyaudio
import wave
import matplotlib.pyplot as plt
from queue import Queue
import math
import utils

# Parameters
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
_WINDOW_SIZE = 16384
_OVERLAP = 16384 // 4
_THRESHOLD = 0.01
threshold = 0.02

# Read the audio file and get the frames
p = pyaudio.PyAudio()
audio = wave.open("./audios/record.wav", "rb")
frames = audio.readframes(audio.getnframes())
frames = np.frombuffer(frames, dtype=np.int16)
fs = audio.getframerate()
rate = audio.getframerate()
audio.close()

# Normalize the audio
normalized_frames = frames / np.max(np.abs(frames))

# Calculate the energy VAD
vad_result, energy_vad = utils.mean_energy_vad(data=normalized_frames, threshold=threshold, window_size=16384, overlap=16384//4)

# Extract the speech
speech = utils.postprocess_speech(data=frames, vad_result=vad_result)

# Calculate the spectrogram	
spectrogram = utils.calculate_spectrogram(data=speech, fs=fs)


#Plot audio and energy
fig, ax = plt.subplots(3,1 , figsize=(10,10))
ax[0].plot(frames)
ax[0].set_title("Speech")


#Plot energy
ax[1].plot(energy_vad)
ax[1].set_title("Energy")
ax[1].axhline(threshold, color='r', linestyle='--')

#Plot spectrogram
ax[2].imshow(spectrogram, aspect='auto', origin='lower', extent=[0, len(frames) / fs, 0, fs / 2])
ax[2].set_title("Spectrogram")
ax[2].set_xlabel("Time")
ax[2].set_ylabel("Frequency")
ax[2].set_ylim(0, 8000)
plt.show()

p = pyaudio.PyAudio()
utils.save_audio(p=p, audio_data=speech, rate=rate,filename="./audios/speech.wav", channels=1, format=pyaudio.paInt16)
p.terminate()

