# app.py
import gradio as gr
import os
import shutil
import ssl

from config import AppConfig, CUDA_AVAILABLE, MPS_AVAILABLE
from audio_processing import step1_extract_audio, step2_remove_silence, step3_chunk_audio
from transcription_logic import step4_run_transcription
from translation_logic import step5_translate_text
# update_tts_input_and_lang is currently unused
from synthesis_logic import step6_synthesize_speech_kokoro
from audio_enhancement import enhance_audio

print(f"--- Startup Check ---")
print(f"CUDA Available: {CUDA_AVAILABLE}")
print(f"MPS (Apple Silicon GPU) Available: {MPS_AVAILABLE}")
if CUDA_AVAILABLE:
    import torch

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"---------------------")

ssl._create_default_https_context = ssl._create_unverified_context

def select_audio_for_chunking(choice, original_path, processed_path):
    return original_path if choice == "Use Original Audio" else processed_path

with gr.Blocks() as demo:
    gr.Markdown("# AETTS Workflow")
    gr.Markdown(
        "Follow the steps below to extract, transcribe, translate and synthesize audio. "
        "Each section can also be used on its own."
    )

    # --- STATE MANAGEMENT ---
    state_original_audio = gr.State(None)
    state_processed_audio = gr.State(None)
    state_audio_for_transcription = gr.State([])

    # --- UI DEFINITION ---

    with gr.Accordion("Step 1: Get Audio", open=False):
        gr.Markdown("#### Step 1: Provide Audio")
        gr.Markdown("Extract audio from a video file or upload an existing audio file.")
        with gr.Tabs():
            with gr.TabItem("From Video File"):
                video_input = gr.Video(label='Input Video')
                ffmpeg_path_input = gr.Textbox(label='Path to FFmpeg', value=AppConfig.FFMPEG_PATH)
                extract_button = gr.Button('1. Extract Audio', variant='primary')
                audio_output_s1 = gr.Audio(label='Extracted Audio', type='filepath')

            with gr.TabItem("From Audio File"):
                audio_upload_input = gr.Audio(label='Input Audio', type='filepath')

    with gr.Accordion("Step 2: Pre-process Audio", open=False):
        gr.Markdown("#### Step 2: Pre-process Audio")
        gr.Markdown(
            "Use the optional tools below to remove silence and split the audio into chunks for easier transcription."
        )
        
        with gr.Group():
            gr.Markdown("**2a. Remove Silence (Optional)**")
            min_silence_len_input = gr.Slider(minimum=100, maximum=2000, value=500, step=100, label="Min Silence (ms)")
            silence_thresh_input = gr.Slider(minimum=-70, maximum=-30, value=-50, step=5, label="Silence Thresh (dB)")
            process_silence_button = gr.Button("Remove Silence", variant="secondary")
            audio_output_s2 = gr.Audio(label="Silence-Removed Audio", type="filepath")
    
        with gr.Group():
            gr.Markdown("**2b. Split Audio into Chunks (for Transcription)**")
            audio_choice_radio = gr.Radio(
                ["Use Original Audio", "Use Silence-Removed Audio"],
                label="Select which audio to use for chunking",
                value="Use Silence-Removed Audio"
            )
            chunk_duration_slider = gr.Slider(minimum=30, maximum=600, value=180, step=30, label="Chunk Duration (seconds)")
    
            process_chunking_button = gr.Button("Prepare for Transcription", variant="primary")
            chunk_download_output = gr.File(label="Download Audio Chunks", interactive=False)

    with gr.Accordion("Step 3: Transcribe to Text", open=False):
        gr.Markdown("#### Step 3: Transcribe to Text")
        gr.Markdown("Convert the prepared audio into text using Whisper.")
        gr.Markdown("You can either use the audio prepared above or upload a new audio file directly.")
        with gr.Tabs():
            with gr.TabItem("Use Prepared Audio"):
                gr.Markdown("Click 'Run Transcription' to use the audio from Step 2.")
            with gr.TabItem("Upload New Audio"):
                direct_transcribe_audio = gr.Audio(label="Upload Audio File(s)", type="filepath")

        gpu_checkbox = gr.Checkbox(label="Use GPU (CUDA/MPS) if available", value=True, visible=(CUDA_AVAILABLE or MPS_AVAILABLE))
        model_size_input = gr.Dropdown(AppConfig.WHISPER_MODELS, value=AppConfig.DEFAULT_WHISPER_MODEL, label="Whisper Model")
        language_input = gr.Dropdown(AppConfig.LANGUAGES, value=AppConfig.DEFAULT_LANGUAGE, label="Audio Language")
        transcribe_button = gr.Button("3. Run Transcription", variant="primary")
        editable_transcription_output = gr.Textbox(label="Editable Transcription Result", lines=8, interactive=True)
        segmented_transcription_output = gr.Textbox(label="Segmented Transcription", lines=8, interactive=False)
        transcript_download_output = gr.File(label="Download Full Transcript (.txt)")

    with gr.Accordion("Step 4: Translate Text", open=False):
        gr.Markdown("#### Step 4: Translate Text")
        gr.Markdown("Translate the German transcript to English. You can also paste your own text below.")
        translate_button = gr.Button("4. Translate German to English", variant="primary")
        editable_translation_output = gr.Textbox(label="Editable Translation (English)", lines=8, interactive=True)

    with gr.Accordion("Step 5: Synthesize Speech (Kokoro TTS)", open=False):
        gr.Markdown("#### Step 5: Synthesize Speech")
        gr.Markdown(
            "Convert text back into speech using Kokoro TTS. You can start from the transcription or the translated text."
        )
        with gr.Row():
            take_from_transcription_button = gr.Button("Take from Transcription")
            take_from_translation_button = gr.Button("Take from Translation")
        tts_input_text = gr.Textbox(label="Text to Synthesize", interactive=True, lines=5)
        kokoro_voice_input = gr.Dropdown(choices=list(AppConfig.KOKORO_VOICES.keys()), value=AppConfig.DEFAULT_KOKORO_VOICE_LABEL, label="Kokoro TTS Voice")
        
        # This slider was added correctly
        tts_speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Voice Speed")

        with gr.Group():
            sentence_wise_checkbox = gr.Checkbox(label="Enable Sentence-wise Synthesis", value=False)
            pause_duration_slider = gr.Slider(minimum=0, maximum=2000, value=500, step=100, label="Pause between sentences (ms)", visible=False)

            def toggle_pause_slider(is_checked):
                return gr.update(visible=is_checked)

            sentence_wise_checkbox.change(toggle_pause_slider, sentence_wise_checkbox, pause_duration_slider)

        tts_button = gr.Button("5. Generate Speech", variant="primary")
        tts_audio_output = gr.Audio(label="Synthesized Speech", type="filepath")
        tts_sentence_download_output = gr.File(label="Download Individual Sentences (.zip)", interactive=False)

    with gr.Accordion("Bonus: Audio Enhancement Toolbox", open=False):
        gr.Markdown("#### Bonus: Audio Enhancement Toolbox")
        gr.Markdown(
            "Optional step to clean up audio with high/low pass filters, noise reduction and more."
        )
        gr.Markdown("Upload one or more audio files to apply enhancement filters.")
        with gr.Row():
            enhancement_audio_input = gr.File(label="Upload Audio File(s)", file_count="multiple")
        with gr.Row():
            highpass_slider = gr.Slider(minimum=0, maximum=500, value=100, step=10, label="High-pass Filter (Hz)")
            lowpass_slider = gr.Slider(minimum=2000, maximum=8000, value=4000, step=100, label="Low-pass Filter (Hz)")
        with gr.Row():
            noise_reduce_checkbox = gr.Checkbox(label="Enable Noise Reduction (afftdn)", value=False)
            dialogue_enhance_checkbox = gr.Checkbox(label="Enable Dialogue Enhancement", value=False)
        compressor_checkbox = gr.Checkbox(label="Enable Dynamic Range Compressor", value=True)
        enhance_button = gr.Button("Enhance Audio", variant="primary")
        with gr.Row():
            enhanced_audio_output = gr.Audio(label="Enhanced Audio Preview", type="filepath")
            enhanced_files_output = gr.File(label="Download Enhanced Files (.zip)")

    # --- EVENT HANDLERS ---
    # ... (no changes to other handlers)
    extract_button.click(step1_extract_audio, [video_input, ffmpeg_path_input], [audio_output_s1, state_original_audio])
    audio_upload_input.change(lambda x: x, inputs=[audio_upload_input], outputs=[state_original_audio])
    process_silence_button.click(step2_remove_silence, [state_original_audio, min_silence_len_input, silence_thresh_input], [audio_output_s2, state_processed_audio])
    process_chunking_button.click(
        lambda choice, orig, proc, ffmpeg, dur: step3_chunk_audio(select_audio_for_chunking(choice, orig, proc), True, dur, ffmpeg),
        inputs=[audio_choice_radio, state_original_audio, state_processed_audio, ffmpeg_path_input, chunk_duration_slider],
        outputs=[state_audio_for_transcription, chunk_download_output]
    )
    transcribe_button.click(
        lambda prep, direct, model, lang, gpu: step4_run_transcription([d for d in (direct or []) if d] or prep, model, lang, gpu),
        [state_audio_for_transcription, direct_transcribe_audio, model_size_input, language_input, gpu_checkbox],
        [editable_transcription_output, segmented_transcription_output, transcript_download_output]
    )
    translate_button.click(step5_translate_text, [editable_transcription_output, gpu_checkbox], [editable_translation_output])
    take_from_transcription_button.click(fn=lambda text: text, inputs=[editable_transcription_output], outputs=[tts_input_text])
    take_from_translation_button.click(fn=lambda text: text, inputs=[editable_translation_output], outputs=[tts_input_text])

    # --- THIS IS THE CORRECTED EVENT HANDLER ---
    @tts_button.click(
        inputs=[tts_input_text, kokoro_voice_input, tts_speed_slider, gpu_checkbox, sentence_wise_checkbox, pause_duration_slider], 
        outputs=[tts_audio_output, tts_sentence_download_output]
    )
    def tts_wrapper(text, voice_label, speed, use_gpu, sentence_wise, pause_duration, progress=gr.Progress()):
        voice_info = AppConfig.KOKORO_VOICES.get(voice_label)
        if not voice_info: raise gr.Error(f"Invalid voice selection: {voice_label}")
        
        if isinstance(voice_info, tuple):
            voice_id, lang = voice_info
        else:
            voice_id, lang = voice_info, "en"

        # THE FIX: The 'speed' parameter is now correctly passed to the backend function.
        return step6_synthesize_speech_kokoro(text, lang, voice_id, speed, use_gpu, sentence_wise, pause_duration, progress)

    # ... (no changes to enhancement handler)
    enhance_button.click(
        enhance_audio,
        inputs=[
            enhancement_audio_input, ffmpeg_path_input, 
            highpass_slider, lowpass_slider, compressor_checkbox,
            noise_reduce_checkbox, dialogue_enhance_checkbox
        ],
        outputs=[enhanced_audio_output, enhanced_files_output]
    )


if __name__ == "__main__":
    if os.path.exists("transcripts"): shutil.rmtree("transcripts")
    demo.launch()