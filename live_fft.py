import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pyaudio
import utils
import wave
from queue import Queue
import threading
from collections import deque


p = pyaudio.PyAudio()
audio = wave.open("./audios/record.wav", "rb")

frames = audio.readframes(audio.getnframes())
frames = np.frombuffer(frames, dtype=np.int16)
audio.close()

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

def calculate_spectrogram(signal, fs, window_size, overlap):
    # Calculate the step size and number of segments
    step_size = window_size - overlap
    n_segments = (len(signal) - window_size) // step_size + 1

    # Initialize the spectrogram with zeros
    spectrogram = np.zeros((window_size // 2 + 1, n_segments))

    # Apply a Hamming window function
    window = np.hamming(window_size)

    # Calculate the spectrogram
    for i in range(n_segments):
        start = i * step_size
        end = start + window_size
        segment = signal[start:end] * window
        _, _, Z = stft(segment, fs, nperseg=window_size)
        Z_resized = np.resize(Z, (window_size // 2 + 1, Z.shape[1]))  # Resize Z to match the expected size

        # spectrogram[:, i] = np.mean(10 * np.log10(np.abs(Z)**2), axis=1)
        spectrogram[:, i] = np.mean(10 * np.log10(np.abs(Z_resized)**2), axis=1)
        # print(spectrogram.shape, flush=True)
        
    return spectrogram

#function that only calculates one step of the spectrogram
def calculate_spectrogram_step(signal, fs, window_size, overlap):

    # Initialize the spectrogram with zeros
    spectrogram = np.zeros((window_size // 2 + 1))

    # Apply a Hamming window function
    window = np.hamming(window_size)

    # Calculate the spectrogram
    start = 0
    end = window_size
    segment = signal * window
    _, _, Z = stft(segment, fs, nperseg=window_size)
    spectrogram = np.mean(10 * np.log10(np.abs(Z)**2), axis=1)

    return spectrogram
    
    

def _record_live_audio(p: pyaudio.PyAudio, input_device_index: int, frames: Queue,
                      recording_status: list, seconds: int = RECORD_SECONDS,
                      format: int = FORMAT, channels: int = CHANNELS, rate: int = RATE,
                      frames_per_buffer: int = FRAMES_PER_BUFFER):
    """
    Records live audio from the selected input device.

    Args:
        p (pyaudio.PyAudio): PyAudio instance.
        input_device_index (int): Index of the input device to use.
        frames (Queue): Queue to store the recorded audio frames.
        recording_status (list): List to track the recording status.
        seconds (int, optional): Duration of the recording in seconds. Defaults to RECORD_SECONDS.
        format (int, optional): Audio format. Defaults to FORMAT.
        channels (int, optional): Number of audio channels. Defaults to CHANNELS.
        rate (int, optional): Sample rate of the audio. Defaults to RATE.
        frames_per_buffer (int, optional): Number of frames per buffer. Defaults to FRAMES_PER_BUFFER.
    """
    audio = []
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
        audio.append(data)
        frames.put(data)
        
    stream.stop_stream()
    stream.close()
    
    frames.put(None)  # Signal the end of the recording
    # # Save the recorded audio to a file
    
    # audio = np.frombuffer(b''.join(audio), dtype=np.int16)
    # utils.save_audio(p, audio, rate=rate, filename="./audios/live_record.wav", channels=channels, format=format)
    
    
    p.terminate()
    recording_status[0] = False
    
    


def _extract_live_spectrogam(in_frames: Queue, output: Queue,  recording_status: list,
                            window_size: int=FRAMES_PER_BUFFER, overlap: int=int(FRAMES_PER_BUFFER // 2)):
    frames = np.array([])

    while recording_status[0]:
        frame = in_frames.get()
        
        if frame is None:
            break

        frames = np.append(frames, np.frombuffer(frame, dtype=np.int16))

        print(len(frames))
        
        if len(frames) >= window_size:
            fs = RATE
            spectrogram_step = calculate_spectrogram(np.array(frames), fs, window_size, overlap)  
            
            output.put(spectrogram_step)
            frames = frames[-overlap:]


 
def calculate_live_spectrogam(p: pyaudio.PyAudio, input_device_index: int, seconds: int = RECORD_SECONDS,
                    format: int = FORMAT, channels: int = CHANNELS, rate: int = RATE,
                    frames_per_buffer: int = FRAMES_PER_BUFFER):
    frames = Queue()
    output = Queue()
    spectrogram = np.empty(((frames_per_buffer // 2) + 1, 0))
    recording_status = [True]



    record_thread = threading.Thread(target=_record_live_audio,
                                     args=(p, input_device_index, frames, recording_status, seconds,
                                           format, channels, rate, frames_per_buffer))

    spectrogram_thread = threading.Thread(target=_extract_live_spectrogam,
                                     args=(frames, output, recording_status, frames_per_buffer, int(frames_per_buffer // 2) ))



    spectrogram_thread.start()
    record_thread.start()

    
    record_thread.join()
    spectrogram_thread.join()
    
    while not output.empty():
        spectrogram = np.hstack((spectrogram, output.get(0)))

    return spectrogram


def main():
    spectrogram = calculate_live_spectrogam(p, input_device_index=utils.select_microphone(p), seconds=10,
                                            frames_per_buffer=FRAMES_PER_BUFFER, rate=RATE)
    
    print("Shape: ", spectrogram.shape)
    print("Memory: ", spectrogram.nbytes / 1024**2, "MB")
    # spectrogram = spectrogram[:160, :] # Tamaño justo
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.imshow(spectrogram, aspect='auto', origin='lower', extent=[0, len(frames) / RATE, 0, RATE / 2])
    ax.set_title('My Spectrogram of Live Audio')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_ylim(0, 7000)
    plt.show()
    return

if __name__ == "__main__":
    main()


