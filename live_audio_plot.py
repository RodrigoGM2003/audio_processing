import pyaudio
import utils
    
def __main__():
    p = pyaudio.PyAudio()
    
    input_device_index = utils.select_microphone(p)
    
    frames = utils.plot_live_audio(p, input_device_index, 
                                  frames_per_buffer=3200, format=pyaudio.paInt16, 
                                  channels=1, rate=44100, seconds=5, performance_mode=True)
    
    utils.save_audio(p, frames, rate=44100, filename='./audios/live_record_plot.wav', format=pyaudio.paInt16, channels=1)
    

if __name__ == "__main__":
    __main__()
    