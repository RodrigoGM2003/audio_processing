import wave
import matplotlib.pyplot as plt
import numpy as np
import utils

obj = wave.open('./audios/live_record_plot.wav','rb')

signal = obj.readframes(-1)
nframes = obj.getnframes()
framerate = obj.getframerate()


time = nframes / framerate
times = np.linspace(0, time, num=nframes)

print(time)

signal_array = np.frombuffer(signal, dtype=np.int16)


utils.plot_audio(signal_array, framerate)




obj.close()