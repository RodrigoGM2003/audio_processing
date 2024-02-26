import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wave
from queue import Queue
import utils
import math

obj = wave.open('./audios/live_record_plot.wav','rb')

frames = obj.readframes(-1)
data = np.frombuffer(frames, dtype=np.int16)
rate = obj.getframerate()

obj.close()

# Plot the audio data
# utils.plot_audio(data, rate)

normalized_data = data / np.max(np.abs(data)) 

#Buenos resultados 2048 - 128

vad_result, energy_vad = utils.energy_vad(normalized_data, threshold=20, window_size=16384, overlap=16384//2)

window_length = math.ceil(len(normalized_data) / len(vad_result))

# Repeat each value in vad_result for the length of each window
expanded_vad_result = np.repeat(vad_result, window_length)

# Trim expanded_vad_result to the length of normalized_data
expanded_vad_result = expanded_vad_result[:len(normalized_data)]

# Now, expanded_vad_result has the same shape as normalized_data, and you can multiply them
speech_data = expanded_vad_result * normalized_data
# speech_data = data[expanded_vad_result == 1]

fig,ax = plt.subplots(4,1, figsize=(15, 15))
ax[0].plot(data)
ax[0].set_title('Audio Signal')
ax[1].plot(vad_result)
ax[1].set_title('Speech')
ax[2].plot(energy_vad)
ax[2].set_title('Energy VAD')
ax[3].plot(speech_data)
ax[3].set_title('Speech Data')

plt.show()


# Save the audio data without silence
#change the format of the audio to be able to save it
speech_data = (speech_data * (2**15 - 1)).astype(np.int16)


p = pyaudio.PyAudio()
utils.save_audio(p, speech_data, rate=44100, filename='./audios/speech_squared.wav', format=pyaudio.paInt16, channels=1)
p.terminate()