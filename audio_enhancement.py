import subprocess
import os
import tempfile
import zipfile
import gradio as gr

def enhance_audio(audio_files, ffmpeg_path, highpass_freq, lowpass_freq, use_compressor, use_noise_reduction, use_dialogue_enhance, progress=gr.Progress()):
    """
    Enhances audio files using ffmpeg filters.
    """
    if not audio_files:
        raise gr.Error("No audio files provided.")

    temp_dir = tempfile.mkdtemp()
    output_paths = []

    for i, audio_file in enumerate(audio_files):
        progress(i / len(audio_files), desc=f"Processing file {i+1}/{len(audio_files)}")
        base_name = os.path.basename(audio_file.name)
        output_filename = os.path.join(temp_dir, f"enhanced_{base_name}")

        filter_complex = []
        if highpass_freq > 0:
            filter_complex.append(f"highpass=f={highpass_freq}")
        if lowpass_freq > 0:
            filter_complex.append(f"lowpass=f={lowpass_freq}")
        if use_noise_reduction:
            filter_complex.append("afftdn")
        if use_dialogue_enhance:
            filter_complex.append("dialoguenhance")
        if use_compressor:
            filter_complex.append("acompressor=threshold=0.1:ratio=9:attack=200:release=1000")

        cmd = [
            ffmpeg_path,
            '-i', audio_file.name,
            '-af',
            ','.join(filter_complex),
            '-y', # Overwrite output file if it exists
            output_filename
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            output_paths.append(output_filename)
        except subprocess.CalledProcessError as e:
            raise gr.Error(f"ffmpeg error: {e.stderr}")

    progress(1, desc="Processing complete!")

    if len(output_paths) == 1:
        return output_paths[0], None
    else:
        zip_path = os.path.join(temp_dir, "enhanced_audio_files.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for f in output_paths:
                zf.write(f, os.path.basename(f))
        return None, zip_path