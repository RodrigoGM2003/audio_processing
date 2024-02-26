import utils
import numpy as np
import matplotlib.pyplot as plt
import pyaudio

p = pyaudio.PyAudio()

seconds = 10

spectrogram = utils.calculate_live_spectrogam(p=p, input_device_index=utils.select_microphone(p=p), seconds=seconds,)

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno',  extent=[0, seconds, 0, 44100 / 2])
ax.set_title('Live Spectrogram')
ax.set_xlabel('Time')
ax.set_ylabel('Frequency')
ax.set_ylim(0, 7000)
plt.show()