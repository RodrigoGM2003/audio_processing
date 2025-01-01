# import speech_recognition as sr
# import matplotlib.pyplot as plt
# import numpy as np

def select_microphone():
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(str(index) + " Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

    mic_index = int(input("Enter the index of the microphone you want to use: "))
    return mic_index


# def plot_audio_waveform(audio):
#     audio_data = np.frombuffer(audio.frame_data, np.int16)
#     time = np.arange(0, len(audio_data) / audio.sample_rate, 1/audio.sample_rate)
#     plt.plot(time, audio_data)
#     plt.show()

# def plot_audio_waveforms_overlapped(audio1, audio2):
#     # Plot the first audio
#     audio_data1 = np.frombuffer(audio1.frame_data, np.int16)
#     time1 = np.arange(0, len(audio_data1) / audio1.sample_rate, 1/audio1.sample_rate)
#     plt.plot(time1, audio_data1, label='Audio 1')

#     # Plot the second audio
#     audio_data2 = np.frombuffer(audio2.frame_data, np.int16)
#     time2 = np.arange(0, len(audio_data2) / audio2.sample_rate, 1/audio2.sample_rate)
#     plt.plot(time2, audio_data2, label='Audio 2')

#     # Add a legend
#     plt.legend()

#     # Display the plot
#     plt.show()
    
# def plot_audio_waveforms(audio1, audio2):
#     # Create the first subplot for the first audio
#     plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
#     audio_data1 = np.frombuffer(audio1.frame_data, np.int16)
#     time1 = np.arange(0, len(audio_data1) / audio1.sample_rate, 1/audio1.sample_rate)
#     plt.plot(time1, audio_data1)
#     plt.title('Audio 1')

#     # Create the second subplot for the second audio
#     plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
#     audio_data2 = np.frombuffer(audio2.frame_data, np.int16)
#     time2 = np.arange(0, len(audio_data2) / audio2.sample_rate, 1/audio2.sample_rate)
#     plt.plot(time2, audio_data2)
#     plt.title('Audio 2')

#     # Display the plots
#     plt.tight_layout()
#     plt.show()


# def capture_and_plot_audio(mic_index):
#     mic = sr.Microphone(device_index=mic_index)
#     recognizer = sr.Recognizer()

#     with mic as source:
#         audio = recognizer.record(source, duration=5)
#         audio2 = recognizer.record(source, duration=5)

#     plot_audio_waveform(audio)
#     plot_audio_waveform(audio2)
#     plot_audio_waveforms(audio, audio2)
    
# if __name__ == "__main__":
#     mic_index = select_microphone()
#     capture_and_plot_audio(mic_index=mic_index)

import speech_recognition as sr
import queue
import threading
import time

def audio_listener(q, mic_index):
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=mic_index)

    with mic as source:
        print("Habla ahora...")

        recognizer.adjust_for_ambient_noise(source)

        while True:
            try:
                audio = recognizer.listen(source, timeout=None)
                q.put(audio)
            except KeyboardInterrupt:
                break

def process_audio(q):
    recognizer = sr.Recognizer()

    while True:
        if not q.empty():
            audio = q.get()

            text = ""
            try:
                text = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                text = ""
                
            print("Transcripción: " + text)
                

if __name__ == "__main__":
    audio_queue = queue.Queue()
    
    mic_index = select_microphone()

    listener_thread = threading.Thread(target=audio_listener, args=(audio_queue, mic_index,))
    processor_thread = threading.Thread(target=process_audio, args=(audio_queue,))

    listener_thread.start()
    processor_thread.start()

    try:
        listener_thread.join()
        processor_thread.join()
    except KeyboardInterrupt:
        print("Proceso detenido por el usuario.")