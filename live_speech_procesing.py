import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wave
from queue import Queue
import utils
import math
import threading
from typing import Callable

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10


def _record_live_audio(p: pyaudio.PyAudio, input_device_index: int, frames: Queue,
                      recording_status: list, seconds: int = RECORD_SECONDS,
                      format: int = FORMAT, channels: int = CHANNELS, rate: int = RATE,
                      frames_per_buffer: int = FRAMES_PER_BUFFER):
    stream = p.open(
        format=format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=frames_per_buffer,
        input_device_index=input_device_index
    )

    print("Recording...")

    for i in range(0, int(rate / frames_per_buffer * seconds)):
        data = stream.read(frames_per_buffer)
        frames.put(data)

    frames.put(None)  # Signal the end of the recording
    stream.stop_stream()
    stream.close()
    p.terminate()
    recording_status[0] = False

_WINDOW_SIZE = 16384
_OVERLAP = 16384 // 4
_THRESHOLD = 0.1


def _extract_live_speech(in_frames: Queue, out_frames: Queue, recording_status: list,
                         window_size: int=_WINDOW_SIZE, overlap: int=_OVERLAP, threshold: float=_THRESHOLD, 
                         vad_function: Callable=utils.mean_energy_vad):

    frames = np.array([])

    aux = 0

    while recording_status[0]:
        frame = in_frames.get()
        
        if frame is None:
            break

        frames = np.append(frames, np.frombuffer(frame, dtype=np.int16))
        # print(frames.shape)
        # print(len(frames))
        if len(frames) >= window_size:
            normalized_frames = frames / np.max(np.abs(frames))
            
            _, energy = vad_function(normalized_frames, window_size, overlap, threshold)

            out_frames.put(energy)
            frames = frames[-(overlap * (aux + 1 )):]
            aux = (aux + 1) % 4
            print(len(frames))
    

     


def isolate_live_speech(p: pyaudio.PyAudio, input_device_index: int, seconds: int = RECORD_SECONDS,
                    format: int = FORMAT, channels: int = CHANNELS, rate: int = RATE,
                    frames_per_buffer: int = FRAMES_PER_BUFFER, performance_mode: bool = True,
                    window_time: int = 2, vad_function: Callable=utils.mean_energy_vad):
    frames = Queue()
    speech_frames = Queue()
    recording_status = [True]
    audio = np.array([])



    record_thread = threading.Thread(target=_record_live_audio,
                                     args=(p, input_device_index, frames, recording_status, seconds,
                                           format, channels, rate, frames_per_buffer))

    live_speech_thread = threading.Thread(target=_extract_live_speech,
                                     args=(frames, speech_frames, recording_status, FRAMES_PER_BUFFER, 
                                           int(FRAMES_PER_BUFFER // 4), 0.1, vad_function))



    record_thread.start()
    live_speech_thread.start()
    # audio = _isolate_live_speech(frames=speech_frames, recording_status=recording_status,
    #                          rate=rate, performance_mode=performance_mode, window_time=window_time)

    
    record_thread.join()
    live_speech_thread.join()
    audio = []
    
    while not speech_frames.empty():
        a = speech_frames.get(0)
        if a == []:
            continue
        audio.append(a[0])
        
        
        # audio = np.append(audio, np.frombuffer(a, dtype=np.float64))
        # print(type(a))
            # audio.append(frame)
        # audio.append(speech_frames.get(0))
        
    print(audio)
    
    audio = np.array(audio)
    

    return audio

def main():
    p = pyaudio.PyAudio()
    index = utils.select_microphone(p)
    speech_frames = isolate_live_speech(p, input_device_index=index, seconds=5, vad_function=utils.mean_energy_vad)
    
    print(speech_frames.shape)
    
    # audio = np.frombuffer(speech_frames, dtype=np.int16)
    
    
    # audio_frames = []
    # while not speech_frames.empty():
    #     audio_frames.append(np.frombuffer(speech_frames.get(0), dtype=np.int16))
        
    
    # audio = np.concatenate(audio_frames)

    
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(speech_frames)
    ax.set_title('Speech Data')
    plt.show()
    
    #format audio data so it can be saved
    
    # utils.save_audio(p, audio, rate=44100, filename='./audios/live_speech_isolation.wav', format=pyaudio.paInt16, channels=1)
    
    
    p.terminate()
    
    


if __name__ == '__main__':
    main()