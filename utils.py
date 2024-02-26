import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import wave
from queue import Queue
import math
from typing import Callable
from scipy.signal import stft

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10

def select_microphone(p: pyaudio.PyAudio):
    """
    Selects the microphone input device to use.

    Args:
        p (pyaudio.PyAudio): PyAudio instance.

    Returns:
        int: Index of the selected input device.
    """
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            print(f"Device index: {i} - {device_info['name']}")
    return int(input("Enter the index of the input device you want to use: "))


def save_audio(p: pyaudio.PyAudio, audio_data: np.array, rate: int, 
               filename: str = "./audios/example.wav", channels: int = 1, format: int = pyaudio.paInt16):
    """
    Saves the audio data to a WAV file.

    Args:
        p (pyaudio.PyAudio): PyAudio instance.
        audio_data (np.array): Audio data as a NumPy array.
        rate (int): Sample rate of the audio data.
        filename (str): Name of the output WAV file.
        channels (int, optional): Number of audio channels. Defaults to 1.
        format (int optional): [description]. Defaults to pyaudio.paInt16.
    """
    obj = wave.open(filename, 'wb')
    obj.setnchannels(channels)
    obj.setsampwidth(p.get_sample_size(format))
    obj.setframerate(rate)
    obj.writeframes(b''.join(audio_data))
    obj.close()


def plot_audio(audio_data: np.array, rate: int, channels: int = 1):
    """
    Plots the audio signal.

    Args:
        audio_data (np.array): Audio data as a NumPy array.
        rate (int): Sample rate of the audio data.
        channels (int, optional): Number of audio channels. Defaults to 1.
    """
    time = len(audio_data) / rate
    times = np.linspace(0, time, num=len(audio_data))

    plt.figure(figsize=(15, 5))
    plt.plot(times, audio_data, label='Audio Signal')
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    plt.xlim(0, time)
    plt.title('Audio Signal')
    plt.show()


# def _record_live_audio(p: pyaudio.PyAudio, input_device_index: int, frames: Queue,
#                       recording_status: list, seconds: int = RECORD_SECONDS,
#                       format: int = FORMAT, channels: int = CHANNELS, rate: int = RATE,
#                       frames_per_buffer: int = FRAMES_PER_BUFFER):
#     """
#     Records live audio from the selected input device.

#     Args:
#         p (pyaudio.PyAudio): PyAudio instance.
#         input_device_index (int): Index of the input device to use.
#         frames (Queue): Queue to store the recorded audio frames.
#         recording_status (list): List to track the recording status.
#         seconds (int, optional): Duration of the recording in seconds. Defaults to RECORD_SECONDS.
#         format (int, optional): Audio format. Defaults to FORMAT.
#         channels (int, optional): Number of audio channels. Defaults to CHANNELS.
#         rate (int, optional): Sample rate of the audio. Defaults to RATE.
#         frames_per_buffer (int, optional): Number of frames per buffer. Defaults to FRAMES_PER_BUFFER.
#     """
#     stream = p.open(
#         format=format,
#         channels=channels,
#         rate=rate,
#         input=True,
#         frames_per_buffer=frames_per_buffer,
#         input_device_index=input_device_index
#     )

#     print("Recording...")

#     for i in range(0, int(rate / frames_per_buffer * seconds)):
#         data = stream.read(frames_per_buffer)
#         frames.put(data)

#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#     recording_status[0] = False
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




def _plot_live_audio(frames: Queue, recording_status: list, rate: int = RATE, performance_mode: bool = False,
                     window_time: int = 2):
    """
    Plots live audio from the recorded frames.

    Args:
        frames (Queue): Queue containing the recorded audio frames.
        recording_status (list): List tracking the recording status.
        rate (int, optional): Sample rate of the audio. Defaults to RATE.
        performance_mode (bool, optional): Whether to use performance mode for plotting. Defaults to False.
        window_time (int, optional): Time window for the plot in seconds. Defaults to 2.

    Returns:
        list: List of audio bytes.
    """
    audio_data = np.array([])

    window_size = window_time * rate

    if not performance_mode:
        audio_data = np.zeros(window_size, dtype=np.int16)

    audio_bytes = []

    # Create the figure and the plot once before the loop
    fig, ax = plt.subplots(figsize=(15, 5))
    line, = ax.plot([], [], label='Audio Signal')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time [s]')
    ax.set_title('Audio Signal')

    def update(frame):
        nonlocal audio_data

        while not frames.empty():
            data = frames.get(0)
            audio_data = np.append(audio_data, np.frombuffer(data, dtype=np.int16))
            audio_bytes.append(data)

        if not recording_status[0] and len(audio_data) == 0:
            ani.event_source.stop()

        if not performance_mode:
            audio_data = audio_data[-window_size:]

        _time = len(audio_data) / rate
        _times = np.linspace(0, _time, num=len(audio_data))

        line.set_data(_times, audio_data)

        ax.set_xlim(0, _time)
        ax.set_ylim(audio_data.min(), audio_data.max())

        if not performance_mode:
            ax.draw_artist(line)
            ax.draw_artist(ax.xaxis)

        return line,

    ani = animation.FuncAnimation(fig, update, interval=50, blit=performance_mode, save_count=50)

    plt.show()

    return audio_bytes

def _extract_live_spectrogam(in_frames: Queue, output: Queue, recording_status: list,
                            window_size: int=FRAMES_PER_BUFFER, overlap: int=int(FRAMES_PER_BUFFER // 2)):
    """
    Extracts live spectrogram from input frames and puts it into the output queue.

    Args:
        in_frames (Queue): Input frames queue.
        output (Queue): Output queue to store the spectrogram.
        recording_status (list): List to control the recording status.
        window_size (int, optional): Size of the window for spectrogram calculation. Defaults to FRAMES_PER_BUFFER.
        overlap (int, optional): Number of overlapping frames. Defaults to FRAMES_PER_BUFFER // 2.
    """
    frames = np.array([])

    while recording_status[0]:
        frame = in_frames.get()
        
        if frame is None:
            break

        frames = np.append(frames, np.frombuffer(frame, dtype=np.int16))

        if len(frames) >= window_size:
            fs = RATE
            spectrogram_step = calculate_spectrogram(np.array(frames), fs, window_size, overlap)  
            
            output.put(spectrogram_step)
            frames = frames[-overlap:]

def plot_live_audio(p: pyaudio.PyAudio, input_device_index: int, seconds: int = RECORD_SECONDS,
                    format: int = FORMAT, channels: int = CHANNELS, rate: int = RATE,
                    frames_per_buffer: int = FRAMES_PER_BUFFER, performance_mode: bool = True,
                    window_time: int = 2):
    """
    Records and plots live audio from the selected input device.

    Args:
        p (pyaudio.PyAudio): PyAudio instance.
        input_device_index (int): Index of the input device to use.
        seconds (int, optional): Duration of the recording in seconds. Defaults to RECORD_SECONDS.
        format (int, optional): Audio format. Defaults to FORMAT.
        channels (int, optional): Number of audio channels. Defaults to CHANNELS.
        rate (int, optional): Sample rate of the audio. Defaults to RATE.
        frames_per_buffer (int, optional): Number of frames per buffer. Defaults to FRAMES_PER_BUFFER.
        performance_mode (bool, optional): Whether to use performance mode for plotting. Defaults to True.
        window_time (int, optional): Time window for the plot in seconds. Defaults to 2.

    Returns:
        list: List of audio bytes.
    """
    frames = Queue()
    recording_status = [True]

    record_thread = threading.Thread(target=_record_live_audio,
                                     args=(p, input_device_index, frames, recording_status, seconds,
                                           format, channels, rate, frames_per_buffer))

    record_thread.start()
    audio = _plot_live_audio(frames=frames, recording_status=recording_status,
                             rate=rate, performance_mode=performance_mode, window_time=window_time)

    record_thread.join()

    return audio

def calculate_live_spectrogam(p: pyaudio.PyAudio, input_device_index: int, seconds: int = RECORD_SECONDS,
                    format: int = FORMAT, channels: int = CHANNELS, rate: int = RATE,
                    frames_per_buffer: int = FRAMES_PER_BUFFER):
    """
    Calculates the live spectrogram of the audio input from a specified device.

    Args:
        p (pyaudio.PyAudio): The PyAudio object.
        input_device_index (int): The index of the input device.
        seconds (int, optional): The duration of the recording in seconds. Defaults to RECORD_SECONDS.
        format (int, optional): The audio format. Defaults to FORMAT.
        channels (int, optional): The number of audio channels. Defaults to CHANNELS.
        rate (int, optional): The sample rate of the audio. Defaults to RATE.
        frames_per_buffer (int, optional): The number of frames per buffer. Defaults to FRAMES_PER_BUFFER.

    Returns:
        numpy.ndarray: The live spectrogram of the audio input.
    """
    
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

def squared_energy_vad(data: np.ndarray, window_size: int=16384, overlap: int=128, threshold: float=1):
    """
    Applies squared energy-based Voice Activity Detection (VAD) to the audio data.

    Args:
        data (np.array): Audio data as a NumPy array.
        window_size (int, optional): Size of the analysis window. Defaults to 16384.
        overlap (int, optional): Number of samples to overlap between windows. Defaults to 512.
        threshold (int, optional): Energy threshold for VAD. Defaults to 1.

    Returns:
        np.array: VAD mask as a NumPy array.
        list: Energy values for each analysis window.
    """
    vad_mask = []
    energy_array = []

    for i in range(0, len(data) - window_size, overlap):
        chunk = data[i:i+window_size]
        energy = np.sum(chunk**2)
        energy_array.append(energy)
        vad_mask.append(energy > threshold)

    return np.array(vad_mask, dtype=int), energy_array

_WINDOW_SIZE = 16384
_OVERLAP = 16384 // 4
_THRESHOLD = 0.01

def mean_energy_vad(data: np.ndarray, window_size: int=_WINDOW_SIZE, overlap: int=_OVERLAP, threshold: float=_THRESHOLD):
    """
    Applies average energy-based Voice Activity Detection (VAD) to the audio data.

    Args:
        data (np.array): Audio data as a NumPy array.
        window_size (int, optional): Size of the analysis window. Defaults to 16384.
        overlap (int, optional): Number of samples to overlap between windows. Defaults to 512.
        threshold (float, optional): Energy threshold for VAD. Defaults to 1.

    Returns:
        np.array: VAD mask as a NumPy array.
        list: Energy values for each analysis window.
    """
    vad_mask = []
    energy_array = []

    for i in range(0, len(data) - window_size, overlap):
    # for i in range(0, window_size, overlap):
        chunk = data[i:i+window_size]
        energy = np.mean(np.abs(chunk))
        energy_array.append(energy)
        vad_mask.append(energy > threshold)

    return np.array(vad_mask, dtype=int), energy_array



def postprocess_speech(data: np.ndarray, vad_result: np.ndarray):
    """
    Postprocesses the speech data based on the VAD (Voice Activity Detection) result.

    Args:
        data (np.ndarray): The input speech data.
        vad_result (np.ndarray): The VAD result indicating the presence of speech.

    Returns:
        np.ndarray: The postprocessed speech data.

    """
    window_length = math.ceil(len(data) / len(vad_result))
    expanded_vad_result = np.repeat(vad_result, window_length)
    expanded_vad_result = expanded_vad_result[:len(data)]
    speech_data = expanded_vad_result * data
    return speech_data


def extract_speech(data: np.ndarray, threshold: float, window_size: int, overlap: int, vad_function: Callable):
    """
    Extracts speech from audio data using voice activity detection.

    Args:
        data (np.ndarray): The audio data.
        threshold (float): The threshold value for voice activity detection.
        window_size (int): The size of the analysis window in samples.
        overlap (float): The overlap ratio between consecutive windows.
        vad_function (Callable): The voice activity detection function.

    Returns:
        np.ndarray: The extracted speech data.
    """
    window_size = int(window_size)
    overlap = int(overlap)
    vad_result, _ = vad_function(data, window_size, overlap, threshold)
    
    speech_data = postprocess_speech(data, vad_result)
    return speech_data


def calculate_spectrogram(data: np.ndarray, fs: int=RATE, window_size: int = 1024, overlap: int = 1024 - 200):
    """
    Calculate the spectrogram of a given data.

    Parameters:
    data (np.ndarray): The input data.
    fs (int): The sampling frequency of the data. Default is RATE.
    window_size (int): The size of the window for calculating the spectrogram. Default is 1024.
    overlap (int): The overlap between consecutive windows. Default is 1024 - 200.

    Returns:
    np.ndarray: The spectrogram of the input data.
    """
    
    # Calculate the step size and number of segments
    epsilon = 1e-10
    step_size = window_size - overlap
    n_segments = (len(data) - window_size) // step_size + 1

    # Initialize the spectrogram with zeros
    spectrogram = np.zeros((window_size // 2 + 1, n_segments))

    # Apply a Hamming window function
    window = np.hamming(window_size)

    # Calculate the spectrogram
    for i in range(n_segments):
        start = i * step_size
        end = start + window_size
        segment = data[start:end] * window
        _, _, Z = stft(segment, fs, nperseg=window_size)
        spectrogram[:, i] = np.mean(10 * np.log10(np.abs(Z + epsilon)**2), axis=1)

    return spectrogram