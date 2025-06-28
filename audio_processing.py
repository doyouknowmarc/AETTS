# audio_processing.py
import gradio as gr
import subprocess
import os
import tempfile
import shutil
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# step1 and step2 functions are unchanged.
def step1_extract_audio(video_path, ffmpeg_path, progress=gr.Progress()):
    progress(0, desc="Starting...")
    if not video_path: raise gr.Error("Please upload a video file.")
    if not ffmpeg_path or not os.path.isfile(shutil.which(ffmpeg_path)): raise gr.Error(
        f"FFmpeg not found at '{ffmpeg_path}'.")
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    if ffmpeg_dir not in os.environ["PATH"]: os.environ["PATH"] += os.pathsep + ffmpeg_dir
    progress(0.3, desc="Extracting audio...")
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            output_path = temp_file.name
        cmd = [ffmpeg_path, "-i", video_path, "-vn", "-y", "-loglevel", "error", "-acodec", "pcm_s16le", "-ar", "16000",
               "-ac", "1", output_path]
        subprocess.run(cmd, check=True)
        progress(1, desc="Audio Extracted!")
        return output_path, output_path
    except Exception as e:
        raise gr.Error(f"Audio extraction failed: {e}")

def step2_remove_silence(original_audio_path, min_silence_len, silence_thresh, progress=gr.Progress()):
    progress(0, desc="Removing silence...")
    if not original_audio_path: raise gr.Error("No original audio file found to process.")
    input_file = original_audio_path
    try:
        audio = AudioSegment.from_file(input_file)
        nonsilent_parts = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        if not nonsilent_parts: raise gr.Error("No non-silent parts detected.")

        processed_audio = AudioSegment.empty()
        for start, end in nonsilent_parts:
            processed_audio += audio[start:end]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            processed_path = temp_file.name
        processed_audio.export(processed_path, format="wav")
        progress(1, desc="Silence Removed!")
        return processed_path, processed_path
    except Exception as e:
        raise gr.Error(f"Error during silence removal: {str(e)}")

# --- MODIFIED FUNCTION ---
def step3_chunk_audio(audio_to_chunk_path, do_chunking, chunk_duration, ffmpeg_path, progress=gr.Progress()):
    progress(0, desc="Preparing audio...")
    if not audio_to_chunk_path:
        raise gr.Error("No audio selected for chunking.")

    # If chunking is not enabled, return None for the download component
    if not do_chunking:
        progress(1, desc="Ready for Transcription (No Chunking)")
        # Return signature: [files_for_state], [files_for_download_component], [update_for_next_group]
        return [audio_to_chunk_path], None, gr.update(visible=True)

    progress(0.2, desc=f"Chunking into {chunk_duration}-second segments...")
    try:
        chunk_dir = tempfile.mkdtemp()
        output_pattern = os.path.join(chunk_dir, "chunk_%03d.wav")
        cmd = [ffmpeg_path, "-i", audio_to_chunk_path, "-f", "segment", "-segment_time", str(chunk_duration),
               "-c:a", "copy", output_pattern]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        chunk_files = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir)],
                             key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

        if not chunk_files:
            progress(1, desc="Audio is shorter than chunk duration, using single file.")
            shutil.rmtree(chunk_dir)
            # Return None for the download component if no chunks were made
            return [audio_to_chunk_path], None, gr.update(visible=True)

        progress(1, desc=f"Created {len(chunk_files)} chunks.")
        os.remove(audio_to_chunk_path)
        # --- KEY CHANGE: Return the list of chunks for both the state and the download component ---
        return chunk_files, chunk_files
    except Exception as e:
        if 'chunk_dir' in locals() and os.path.exists(chunk_dir): shutil.rmtree(chunk_dir)
        raise gr.Error(f"Error during chunking: {e}")