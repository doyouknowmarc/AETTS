# synthesis_logic.py
import gradio as gr
import torch
import functools
import tempfile
import soundfile as sf
import numpy as np
import os
import traceback
import re
import zipfile
import warnings  # Enable warning control

# Suppress less useful warnings from dependencies
warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout after all but last recurrent layer*",
)
warnings.filterwarnings(
    "ignore",
    message="*torch.nn.utils.weight_norm*is deprecated*",
)
warnings.filterwarnings(
    "ignore",
    message="Defaulting repo_id*",
)

# --- Import from our project files ---
from config import AppConfig, CUDA_AVAILABLE, MPS_AVAILABLE

try:
    from kokoro.model import KModel
    from kokoro.pipeline import KPipeline
except ImportError:
    raise ImportError("Kokoro library not found. Please install it with: pip install kokoro>=0.9.4 soundfile numpy")
REPO_ID = 'hexgrad/Kokoro-82M'
#LANG_CODE = 'a'
#VOICE = 'am_michael'
#SPEED = 1.3
SAMPLE_RATE = 24000
FRAMES_PER_BUFFER = 1024

@functools.lru_cache(maxsize=1)
def load_local_kmodel(device):
    """
    This function correctly loads the KModel from local files.
    This is the expensive operation we want to cache.
    """
    model_dir = AppConfig.LOCAL_KOKORO_MODEL_PATH
    config_path = os.path.join(model_dir, 'config.json')
    model_path = os.path.join(model_dir, 'kokoro-v1_0.pth')

    if not all(os.path.exists(p) for p in [config_path, model_path]):
        raise FileNotFoundError(
            f"Could not find model files in '{model_dir}'. "
            "Please ensure config.json and model.pth exist."
        )

    print(f"Loading local KModel from: {model_path} onto device: {device}")
    k_model = KModel(repo_id='hexgrad/Kokoro-82M', config=config_path, model=model_path)
    k_model.to(device)
    k_model.eval()
    return k_model

def _synthesize_text_chunk(pipeline, text, voice_id, speed):
    """Helper to synthesize a single chunk of text with a given speed."""
    generator = pipeline(text, voice=voice_id, speed=speed) # Pass speed here
    audio_chunks = [audio for _, _, audio in generator]
    if not audio_chunks:
        return None
    return np.concatenate(audio_chunks)

def step6_synthesize_speech_kokoro(text_to_speak, language_for_tts, kokoro_voice_id, speed, use_gpu, sentence_wise, pause_duration_ms, progress=gr.Progress()):
    """
    Synthesizes speech using the Kokoro TTS library, with sentence-wise processing and speed control.
    """
    progress(0, desc="Starting TTS...")
    if not text_to_speak or not text_to_speak.strip():
        raise gr.Error("Text for speech synthesis cannot be empty.")

    voice_id = kokoro_voice_id

    device = "cpu"
    if use_gpu:
        if CUDA_AVAILABLE:
            device = "cuda"
        elif MPS_AVAILABLE:
            device = "mps"

    progress(0.1, desc=f"Loading model on '{device}'...")

    try:
        loaded_model = load_local_kmodel(device)
        
        kokoro_lang_code = AppConfig.KOKORO_LANG_MAP.get(language_for_tts)
        if not kokoro_lang_code:
            supported_langs = list(AppConfig.KOKORO_LANG_MAP.keys())
            raise gr.Error(
                f"Language '{language_for_tts}' is not supported by Kokoro TTS. "
                f"Please choose a voice for a supported language: {supported_langs}"
            )
        
        print(f"Initializing Kokoro pipeline with lang_code: '{kokoro_lang_code}' (mapped from '{language_for_tts}')")
        pipeline = KPipeline(
            lang_code=kokoro_lang_code,
            model=loaded_model,
            repo_id=REPO_ID,
        )
        
        voice_file_path = os.path.join(AppConfig.LOCAL_KOKORO_MODEL_PATH, "voices", f"{voice_id}.pt")
        if not os.path.exists(voice_file_path):
            raise FileNotFoundError(f"Voice file not found: {voice_file_path}.")

        voice_tensor = torch.load(voice_file_path, map_location=device, weights_only=True)
        pipeline.voices[voice_id] = voice_tensor

        if sentence_wise:
            sentences = re.split('(?<=[.!?]) +', text_to_speak.strip())
            if not sentences:
                raise gr.Error("Could not split text into sentences.")

            all_audio_segments = []
            sentence_files = []
            temp_dir = tempfile.mkdtemp()
            pause_audio = np.zeros(int(24000 * (pause_duration_ms / 1000.0)), dtype=np.float32)

            for i, sentence in enumerate(sentences):
                if not sentence.strip(): continue
                progress(0.2 + (i / len(sentences)) * 0.7, desc=f"Synthesizing sentence {i+1}/{len(sentences)}...")
                
                audio_segment = _synthesize_text_chunk(pipeline, sentence, voice_id, speed)
                
                if audio_segment is not None:
                    all_audio_segments.append(audio_segment)
                    if i < len(sentences) - 1:
                        all_audio_segments.append(pause_audio)
                    
                    sentence_filename = os.path.join(temp_dir, f"sentence_{i+1}.wav")
                    sf.write(sentence_filename, audio_segment, samplerate=24000)
                    sentence_files.append(sentence_filename)

            if not all_audio_segments:
                raise gr.Error("Sentence-wise TTS failed to produce any audio.")

            full_audio = np.concatenate(all_audio_segments)
            zip_path = os.path.join(temp_dir, "sentences.zip")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for f in sentence_files:
                    zf.write(f, os.path.basename(f))
            
            download_path = zip_path

        else:
            progress(0.4, desc="Generating audio from text...")
            
            full_audio = _synthesize_text_chunk(pipeline, text_to_speak, voice_id, speed)

            if full_audio is None:
                raise gr.Error("TTS generation failed to produce any audio.")
            download_path = None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            output_wav_path = temp_file.name
        sf.write(output_wav_path, full_audio, samplerate=24000)
        progress(1, desc="Speech Generated!")
        
        return output_wav_path, download_path

    except Exception as e:
        print(traceback.format_exc())
        raise gr.Error(f"An error occurred during speech synthesis: {e}")


# def update_tts_input_and_lang(choice, original_text, translated_text):
#     """Return text and language code based on user choice (unused)."""
#     if choice == "Use Original (German)":
#         return original_text, "de"
#     else:
#         return translated_text, "en"
