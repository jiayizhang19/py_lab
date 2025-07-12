
import whisper
from faster_whisper import WhisperModel
import numpy as np
import pyaudio
import keyboard
from scipy.io import wavfile


# Whisper is not an ideal tool for live transcription, need to seek other library for live speech
def process_live_speech(task="transcribe"):
    """
    chunk_size: the number of samples
    sample_rate: the number of samples per second
    chunk_size / sample_rate = xx seconds, e.g. 2048 / 16000 = 0.128s
    """
    # Live audio config
    sample_rate = 16000
    chunk_size = 2048
    min_chunk_seconds = 2
    overlap_duration = 0.5
    min_samples = sample_rate * min_chunk_seconds   # Buffer 2 seconds of audio
    overlap_samples = int(sample_rate * overlap_duration)   # 0.5 second of overlap

    audio, stream, chunk_size = initialize_audio_stream(chunk_size=chunk_size)
    audio_buffer = np.array([], dtype=np.int16)
    prev_tail = np.array([], dtype=np.int16)  
    print("Recording ... Press Enter to stop")
    print("> ", end="")
    try:
        while not keyboard.is_pressed("enter"):
            # Each data is a small NumPy array
            raw_data = stream.read(num_frames=chunk_size, exception_on_overflow=False)
            new_chunk = np.frombuffer(raw_data, dtype=np.int16)
            audio_buffer = np.concatenate((audio_buffer,new_chunk))
            if len(audio_buffer) >= min_samples:
                chunk_with_tail = np.concatenate((prev_tail, audio_buffer))
                text = speech_to_text(chunk_with_tail, task)
                prev_tail = audio_buffer[-overlap_samples:]
                audio_buffer = np.array([], dtype=np.int16)
                if text:
                    print(f"{text}", end="", flush=True)
    finally:
        if len(audio_buffer) > 0:
            final_chunk = np.concatenate((prev_tail, audio_buffer))
            text = speech_to_text(final_chunk, task)
            if text:
                print(f"{text}", end=" ", flush=True)
        print("\n> End...")
        clean_up_audio_stream(stream, audio)
        

def process_wav_recording(file, task="transcribe"):
    """
    Load .wav file and convert to numpy array
    """
    _, audio_data = wavfile.read(file)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    return speech_to_text(audio_data, task)


def speech_to_text(audio_data, task, model_size="base"):
    """
    Convert audio to text with Whisper, with the default task 
    Arg:
        audio_data: audio data in 16-bit integer type in numpy array, ranging from -32768 to 32727.
                    Need to convert audio data from 16-bit int to 32-bit float ranging from -1.0 to 1.0, expected by most audio processing models.
                    If the audio input is a recorded .wav, below line won't work, as the .wav file is a str type.
        task: transcribe or translate
        model_size: the size of whisper, it could be tiny, base (default value), small, large 
    Returns:
        text
    """
    audio_float = audio_data.astype(np.float32) / 32768.0
    # =================================================================================== 
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> using whisper <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # model = whisper.load_model(model)
    # result = model.transcribe(audio_float, fp16=False) # fp16 is not supported on CPU but GPU, usinng fp32 byb default on CPU
    # return result["text"]
    # ===================================================================================
    model = WhisperModel(
        model_size, 
        device="cpu", 
        compute_type="int8"
        )
    segments, _ = model.transcribe(
        audio_float, 
        beam_size=5, # better decoding accuracy
        vad_filter=True, # set to True aviods wasting whisper's time and CPU on silence
        # language="en",
        task=task
        )
    return " ".join(segment.text for segment in segments)
    

def initialize_audio_stream(sample_rate=16000, chunk_size=1024):
    """
    Args:
        sample_rate: audio sample rate (default 16000)
        chunk_size: audio frames per buffer 
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        input=True,
        rate=sample_rate,
        frames_per_buffer=chunk_size
    )
    return audio, stream, chunk_size


def clean_up_audio_stream(stream, audio):
    """
    Clean up stream and audio to release resources
    """
    stream.stop_stream()
    stream.close()
    audio.terminate()



    
