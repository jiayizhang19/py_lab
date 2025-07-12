from faster_whisper import WhisperModel
from zhconv import convert
import numpy as np
import pyaudio
import keyboard
from googletrans import Translator
import edge_tts
import playsound
import os
import tempfile
import asyncio

en_to_ch = {"speech": "en", "src": "en", "dest": "zh-CN", "voice": "zh-CN-XiaoxiaoNeural"}
ch_to_en = {"speech": "zh", "src": "zh-CN", "dest": "en", "voice": "en-US-JennyNeural"}


def main():
    choice = get_choice()
    text = speech_to_text(choice["speech"])
    print("> Speech: ", text)
    translation = translate_text(text=text, src=choice["src"], dest=choice["dest"])
    print("> Translation: ", translation)
    asyncio.run(text_to_speech(translation, voice=choice["voice"]))


def get_choice():
    while True:
        text = input("Press 1 for en to ch, 2 for ch to en: ")
        if "1" in text:
            return en_to_ch
        elif "2" in text:
            return ch_to_en
        else:
            continue


def translate_text(text, src, dest):
    translator = Translator()
    result = translator.translate(text, src=src, dest=dest)
    return result.text


async def text_to_speech(text, voice):
    """
    create a temp .mp3 file, then play it instantly and remote it
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
        output_file = temp.name
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_file)
    playsound.playsound(output_file)
    os.remove(output_file)


def speech_to_text(language):
    """
    Convert audio to text with Whisper, with the default task as transcribe
    Arg:
        audio_data: audio data in 16-bit integer type in numpy array, ranging from -32768 to 32727.
                    Need to convert audio data from 16-bit int to 32-bit float ranging from -1.0 to 1.0, expected by most audio processing models.
                    If the audio input is a recorded .wav, below line won't work, as the .wav file is a str type.
        task: transcribe or translate
        model_size: the size of whisper, it could be tiny, base (default value), small, large 
    Returns:
        text
    """
    audio_data = process_dialogue_data()
    audio_float = audio_data.astype(np.float32) / 32768.0
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        audio_float, 
        beam_size=5, # better decoding accuracy with the larger number
        vad_filter=True, # set to True aviods wasting whisper's time and CPU on silence
        language=language,
        )
    text = " ".join(segment.text for segment in segments)
    return convert(text, "zh-cn") if language == "zh" else text
    

def process_dialogue_data():
    """
    Records audio in chunks until Enter key is pressed.
    Returns Numpy array of recorded audio data for whisper usage in speech_to_text
    """
    audio, stream, chunk_size = initialize_audio_stream()
    frames = []
    print("Recording ... Press Enter to stop")
    while not keyboard.is_pressed("enter"):
        # Each data is a small NumPy array
        data = stream.read(num_frames=chunk_size, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))
    # combine each small array to one long array for whisper audio input
    audio_data = np.concatenate(frames)
    clean_up_audio_stream(stream=stream, audio=audio)
    return audio_data


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


if __name__ == "__main__":
    main()

    
