from faster_whisper import WhisperModel
from zhconv import convert
import numpy as np
from googletrans import Translator
import edge_tts
import asyncio
import gradio as gr
import tempfile

# Initialize models
model = WhisperModel("base", device="cpu")
translator = Translator()

# Language profiles
en_to_ch = {"speech": "en", "src": "en", "dest": "zh-CN", "voice": "zh-CN-XiaoxiaoNeural"}
ch_to_en = {"speech": "zh", "src": "zh-CN", "dest": "en", "voice": "en-US-JennyNeural"}

def process_audio(audio_data, direction):
    """Handle the full pipeline for a given audio file"""
    try:
        choice = en_to_ch if direction == "en_to_ch" else ch_to_en
        
        # 1. Transcribe
        text = speech_to_text(audio_data, language=choice["speech"])
        
        # 2. Translate
        translation = translate_text(text, src=choice["src"], dest=choice["dest"])
        
        # 3. Generate speech
        audio_path = asyncio.run(text_to_speech(translation, voice=choice["voice"]))
        
        return text, translation, audio_path
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

def translate_text(text, src, dest):
    result = translator.translate(text, src=src, dest=dest)
    return result.text

async def text_to_speech(text, voice):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        output_path = f.name
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)
    return output_path

def speech_to_text(audio_data, language):
    audio_float = audio_data.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio_float, 
        beam_size=5,
        vad_filter=True,
        language=language,
        initial_prompt="请用简体中文" if language == "zh" else None
    )
    text = " ".join(segment.text for segment in segments)
    return convert(text, "zh-cn") if language == "zh" else text

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("## 🎤 Real-Time Speech Translator")
    
    with gr.Row():
        direction = gr.Radio(
            choices=[("English → Chinese", "en_to_ch"), 
                    ("Chinese → English", "ch_to_en")],
            value="en_to_ch",
            label="Translation Direction"
        )
        mic = gr.Microphone(type="numpy")
    
    btn = gr.Button("Process", variant="primary")
    
    with gr.Column():
        transcription = gr.Textbox(label="Original Transcription")
        translation = gr.Textbox(label="Translated Text")
        audio_output = gr.Audio(label="Spoken Translation", autoplay=True)

    btn.click(
        fn=process_audio,
        inputs=[mic, direction],
        outputs=[transcription, translation, audio_output]
    )

# Run the app
if __name__ == "__main__":
    app.launch(share=True, show_error=True)  # Set share=False for local only
