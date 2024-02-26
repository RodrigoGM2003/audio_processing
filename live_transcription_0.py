# import speech_recognition as sr
# import time
# import os
# import threading
# import queue

# # List all microphones
# for index, name in enumerate(sr.Microphone.list_microphone_names()):
#     print(str(index) + " Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

# mic_index = int(input("Enter the index of the microphone you want to use: "))

# # Create a Recognizer instance
# r = sr.Recognizer()

# mic = sr.Microphone(device_index=mic_index)

# audio_queue = queue.Queue()
# message_queue = queue.Queue()
# text = ""

# def listen(r, mic):
#     while True:
#         with mic as source:
#             try:
#                 audio = r.listen(source, duration = 2)
#                 audio_queue.append(audio)
#             except sr.UnknownValueError:
#                 print("Google Speech Recognition could not understand audio")
#             except sr.RequestError as e:
#                 print("Could not request results from Google Speech Recognition service; {0}".format(e))
            
# def transform(r):
#     while True:
#         if(len(audio_queue) == 0):
#             time.sleep(0.1)
        
#         else:
#             audio = audio_queue.pop(0)
#             message_queue.append(r.recognize_google(audio, language="es-ES"))
        
# def print_message():
#     message = ""
#     while True:
#         if(len(message_queue) == 0):
#             time.sleep(0.1)
#         else:
#             print(message_queue.pop(0), end="")


 
#             # message.append( " " + message_queue.pop(0))
#             # os.system('cls' if os.name == 'nt' else 'clear')
#             # print(text)
#             # for i in message:
#             #     print(i)  
#             #     time.sleep(0.05)

# # # Start listening to the audio
# # with mic as source:
# #     while True:
# #         audio = r.record(source, duration=2)  # Listen for the first phrase and extract it into audio data

# #         try:
# #             print( r.recognize_google(audio, language="es-ES"))  # Recognize speech using Google Speech Recognition
# #         except sr.UnknownValueError:
# #             print("Google Speech Recognition could not understand audio")
# #         except sr.RequestError as e:
# #             print("Could not request results from Google Speech Recognition service; {0}".format(e))
            
#             # Start listening to the audio
# # mensaje = ""
# # with mic as source:
# #     while True:
# #         audio = r.record(source, duration=2)  # Listen for the first phrase and extract it into audio data
# #         buffer: str = ""
# #         try:
# #             buffer = " " + r.recognize_google(audio, language="es-ES")  # Recognize speech using Google Speech Recognition
# #         except sr.UnknownValueError:
# #              print("Google Speech Recognition could not understand audio")
# #         except sr.RequestError as e:
# #              print("Could not request results from Google Speech Recognition service; {0}".format(e))
        
        
# #         if len(buffer) > 0:
# #             for i in buffer:
# #                 mensaje = mensaje + i
# #                 os.system('cls' if os.name == 'nt' else 'clear')
# #                 print(mensaje)
# #                 time.sleep(0.05)
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