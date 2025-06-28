import gradio as gr
import whisper
import functools
import os
import tempfile
import shutil
# --- IMPORT AppConfig ---
from config import CUDA_AVAILABLE, AppConfig

@functools.lru_cache(maxsize=2)
def load_model(model_name, device):
    """
    Loads a Whisper model. If the models are present locally, it will use them.
    The `download_root` parameter points to our local models directory.
    """
    print(f"Loading Whisper model '{model_name}' on device '{device}'...")
    # --- MODIFY THIS LINE ---
    # This tells whisper to look for models in the specified local directory.
    # If a model is not found, it would attempt to download it to this location.
    # Since we pre-downloaded, it will just load the local file.
    return whisper.load_model(
        model_name,
        device=device,
        download_root=AppConfig.WHISPER_MODELS_PATH
    )


def step4_run_transcription(audio_files, model_size, language, use_gpu, progress=gr.Progress()):
    if not audio_files: raise gr.Error("No audio files available to transcribe.")
    device = "cuda" if use_gpu and CUDA_AVAILABLE else "cpu"
    progress(0, desc=f"Loading Whisper model on {device}...")
    try:
        model = load_model(model_size, device)
        use_fp16 = (device == "cuda")
        all_text, all_segments_text = [], []
        for i, audio_path in enumerate(audio_files):
            progress(i / len(audio_files), desc=f"Transcribing chunk {i + 1}/{len(audio_files)}...")
            result = model.transcribe(audio_path, fp16=use_fp16, language=language, temperature=0.0)
            all_text.append(result["text"].strip())
            if len(audio_files) > 1: all_segments_text.append(f"--- CHUNK {i + 1}/{len(audio_files)} ---\n")
            for segment in result["segments"]: all_segments_text.append(
                f"[{segment['start']:.2f} - {segment['end']:.2f}]: {segment['text'].strip()}\n")
            all_segments_text.append("\n")
        progress(1, desc="Transcription Complete!")
        final_full_text = " ".join(all_text)
        final_segments_text = "".join(all_segments_text)
        os.makedirs("transcripts", exist_ok=True)
        txt_path = os.path.join("transcripts", f"transcript_{os.path.basename(tempfile.mkstemp()[1])}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(
                f"=== Full Transcription ===\n\n{final_full_text}\n\n=== Segmented Transcription ===\n\n{final_segments_text}")
        chunk_dir = os.path.dirname(audio_files[0])
        if "chunk" in chunk_dir and os.path.exists(chunk_dir): shutil.rmtree(chunk_dir, ignore_errors=True)
        return final_full_text, final_segments_text, txt_path
    except Exception as e:
        raise gr.Error(f"Transcription failed: {e}")