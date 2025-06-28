# translation_logic.py
import gradio as gr
import functools
import os
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


def step5_translate_text(original_text, use_gpu, progress=gr.Progress()):
    """
    Translates text using the locally-loaded model.
    """
    if not original_text.strip(): raise gr.Error("No text to translate.")

    # Determine device for the pipeline
    device = "cpu"
    if use_gpu:
        if CUDA_AVAILABLE:
            device = "cuda"
        elif torch.backends.mps.is_available(): # Check for MPS here as well
            device = "mps"

    progress(0, desc=f"Loading translator on {device}...")
    try:
        translator = load_translator(device)
        progress(0.5, desc="Translating DE -> EN...")
        # The pipeline handles batching, but for a single text box, this is fine.
        result = translator(original_text, max_length=512) # Added max_length for safety
        translated_text = result[0]['translation_text']
        progress(1, "Translation Complete!")
        return translated_text
    except Exception as e:
        raise gr.Error(f"Translation failed: {e}")

# We need to import torch for the MPS check in step5_translate_text
import torch