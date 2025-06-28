# translation_logic.py
import gradio as gr
import functools
import os
import re
from transformers import pipeline
from config import AppConfig, CUDA_AVAILABLE

@functools.lru_cache(maxsize=1)
def load_translator(device):
    """
    Loads the translation pipeline from the local, pre-downloaded model files.
    This ensures the function works completely offline.
    """
    local_model_path = AppConfig.LOCAL_TRANSLATION_MODEL_PATH

    # --- OFFLINE-READY CHECK ---
    # Check if the local model directory exists. If not, guide the user.
    if not os.path.isdir(local_model_path):
        raise FileNotFoundError(
            f"Translation model not found at '{local_model_path}'. "
            "Please run the 'download_model.py' script once with an internet connection to download the model."
        )

    print(f"Loading local translation model from '{local_model_path}'...")
    hf_device = -1 # Default to CPU
    if device == "cuda":
        hf_device = 0
    elif device == "mps":
        # Note: Transformers pipeline might not fully utilize MPS.
        # It's better than CPU but may not be as optimized as direct PyTorch.
        hf_device = "mps"

    # --- MODIFIED: Load from the local path ---
    return pipeline("translation_de_to_en", model=local_model_path, device=hf_device)


def _split_text(text, sentences_per_chunk=1):
    """Split text into chunks of a given number of sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def step5_translate_text(original_text, use_gpu, progress=gr.Progress()):
    """Translate text in smaller chunks and stream the result."""
    if not original_text.strip():
        raise gr.Error("No text to translate.")

    device = "cpu"
    if use_gpu:
        if CUDA_AVAILABLE:
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    progress(0, desc=f"Loading translator on {device}...")
    try:
        translator = load_translator(device)
        chunks = _split_text(original_text, sentences_per_chunk=1)
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            progress((i / max(len(chunks), 1)) * 0.9 + 0.05,
                     desc=f"Translating chunk {i + 1}/{len(chunks)}...")
            result = translator(chunk, max_length=512)
            translated_chunk = result[0]["translation_text"]
            translated_chunks.append(translated_chunk)
            yield " ".join(translated_chunks)
        progress(1, desc="Translation Complete!")
    except Exception as e:
        raise gr.Error(f"Translation failed: {e}")

# We need to import torch for the MPS check in step5_translate_text
import torch
