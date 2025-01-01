import speech_recognition as sr
import time
import sys
import os
import threading
import queue
import signal

stop_threads = False
# List all microphones
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(str(index) + " Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

mic_index = int(input("Enter the index of the microphone you want to use: "))

# Create a Recognizer instance
r = sr.Recognizer()

mic = sr.Microphone(device_index=mic_index)

audio_queue = queue.Queue()
message_queue = queue.Queue()
text = ""



def listen(r, mic):
    while not stop_threads:
        with mic as source:
            audio = r.record(source, duration=1)
            audio_queue.put(audio)
                
def transform():
    while not stop_threads:
        if not audio_queue.empty():
            audio = audio_queue.get()
            try:
                message_queue.put(r.recognize_google(audio, language="es-ES"))
            except sr.UnknownValueError:
                    time.sleep(0.1)
        # else:
        #     time.sleep(0.05)
            
def print_message():
    global stop_threads
    while not stop_threads:
        message = ""
        if not message_queue.empty():
            printed_message = message_queue.get()
            
            if printed_message == "salir":
                stop_threads = True
            if printed_message == "limpia":
                os.system('cls' if os.name == 'nt' else 'clear')
                
            for m in printed_message:
                print(m, end="")
                sys.stdout.flush()
                time.sleep(0.05)
            print(" ", end="")
  


listen_thread = threading.Thread(target=listen, args=(r, mic))
transform_thread = threading.Thread(target=transform, args=())
print_thread = threading.Thread(target=print_message)

listen_thread.start()
transform_thread.start()
print_thread.start()
    

# Wait for all threads to finish
listen_thread.join()
transform_thread.join()
print_thread.join()