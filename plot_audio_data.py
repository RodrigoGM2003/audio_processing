import numpy as np
from collections import deque
import utils 
import pyaudio
import wave
import matplotlib.pyplot as plt
from queue import Queue
import math

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10

p = pyaudio.PyAudio()
audio = wave.open("./audios/record.wav", "rb")

threshold = 0.02

frames = audio.readframes(audio.getnframes())
frames = np.frombuffer(frames, dtype=np.int16)
fs = audio.getframerate()
rate = audio.getframerate()

audio.close()

_WINDOW_SIZE = 16384
_OVERLAP = 16384 // 4
_THRESHOLD = 0.01


def mean_energy_vad(data: np.ndarray, window_size: int=_WINDOW_SIZE, overlap: int=_OVERLAP, threshold: float=_THRESHOLD):
    vad_mask = []
    energy_array = []

    for i in range(0, len(data) - window_size, overlap):
    # for i in range(0, window_size, overlap):
        chunk = data[i:i+window_size]
        energy = np.mean(np.abs(chunk))
        energy_array.append(energy)
        vad_mask.append(energy > threshold)

    return np.array(vad_mask, dtype=int), energy_array


def postprocess_speech(data: np.ndarray, vad_result: np.ndarray):

    window_length = math.ceil(len(data) / len(vad_result))
    expanded_vad_result = np.repeat(vad_result, window_length)
    expanded_vad_result = expanded_vad_result[:len(data)]
    # speech_data = expanded_vad_result * data
    speech_data = data[expanded_vad_result.astype(bool)]
    
    return speech_data




normalized_frames = frames / np.max(np.abs(frames))

vad_result, energy_vad = mean_energy_vad(data=normalized_frames, threshold=threshold, window_size=16384, overlap=16384//4)
speech = postprocess_speech(data=frames, vad_result=vad_result)

print(len(speech), len(frames))
print(speech[:10], frames[:10], type(speech[0]), type(frames[0]))


# speech = utils.extract_speech(data=normalized_frames, window_size=16384, overlap=16384//4, 
#                               threshold=0.1, vad_function=utils.mean_energy_vad)

spectrogram = utils.calculate_spectrogram(data=speech, fs=fs)
# spectrogram = utils.calculate_spectrogram(data=frames, fs=fs)




#plot audio and energy
fig, ax = plt.subplots(3,1 , figsize=(10,10))
ax[0].plot(frames)
ax[0].set_title("Speech")
ax[1].plot(energy_vad)
ax[1].set_title("Energy")
#plot threshold
ax[1].axhline(threshold, color='r', linestyle='--')
ax[2].imshow(spectrogram, aspect='auto', origin='lower', extent=[0, len(frames) / fs, 0, fs / 2])
ax[2].set_title("Spectrogram")
ax[2].set_xlabel("Time")
ax[2].set_ylabel("Frequency")
ax[2].set_ylim(0, 8000)
plt.show()

#save the audio with the correct format
# speech = (speech * (2**15 - 1)).astype(np.int16)


p = pyaudio.PyAudio()
utils.save_audio(p=p, audio_data=speech, rate=rate,filename="./audios/speech.wav", channels=1, format=pyaudio.paInt16)
p.terminate()

