# import speech_recognition as sr
# import matplotlib.pyplot as plt
# import numpy as np

def select_microphone():
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(str(index) + " Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

    mic_index = int(input("Enter the index of the microphone you want to use: "))
    return mic_index



import speech_recognition as sr
import queue
import threading
import time
import utils

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
    
    mic_index = utils.select_microphone()

    listener_thread = threading.Thread(target=audio_listener, args=(audio_queue, mic_index,))
    processor_thread = threading.Thread(target=process_audio, args=(audio_queue,))

    listener_thread.start()
    processor_thread.start()

    try:
        listener_thread.join()
        processor_thread.join()
    except KeyboardInterrupt:
        print("Proceso detenido por el usuario.")