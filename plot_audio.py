import wave
import matplotlib.pyplot as plt
import numpy as np
import utils

# Load the audio file
obj = wave.open('./audios/record.wav','rb')
signal = obj.readframes(-1)

# Get the frames and the framerate
nframes = obj.getnframes()
framerate = obj.getframerate()

# Calculate the time
time = nframes / framerate
times = np.linspace(0, time, num=nframes)

# Print the time
print(time)

# Convert the signal to a numpy array
signal_array = np.frombuffer(signal, dtype=np.int16)

# Plot the audio
utils.plot_audio(signal_array, framerate)

obj.close()