import wave
import matplotlib.pyplot as plt
import numpy as np
import pyaudio

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10


p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    if device_info['maxInputChannels'] > 0:
        print(f"Device index: {i} - {device_info['name']}")

input_device_index = int(input("Enter the index of the input device you want to use: "))


stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER,
    input_device_index=input_device_index
)

print("Recording...")

frames = []

for i in range(0, int(RATE / FRAMES_PER_BUFFER * RECORD_SECONDS)):
    data = stream.read(FRAMES_PER_BUFFER)
    frames.append(data)
    
stream.stop_stream()
stream.close()

p.terminate()

obj = wave.open('./audios/record.wav','wb')
obj.setnchannels(CHANNELS)
obj.setsampwidth(p.get_sample_size(FORMAT))
obj.setframerate(RATE)
obj.writeframes(b''.join(frames))
obj.close()
