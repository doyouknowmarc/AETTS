# AETTS

AETTS (Audio Extraction, Transcription, Translation and Synthesis) is a local pipeline for processing audio and video. It provides a Gradio interface that can extract audio from a video, pre‑process it, transcribe it with Whisper, translate it from German to English and synthesize new speech with Kokoro TTS. Optional ffmpeg based audio enhancement tools are also included.

## Features
- **Audio extraction** from video files via ffmpeg.
- **Silence removal** and **chunking** for easier transcription.
- **Offline transcription** using pre‑downloaded Whisper models.
- **Offline translation** (DE → EN) using a local HuggingFace model.
- **Speech synthesis** with Kokoro TTS and several voices.
- **Audio enhancement** filters (high/low pass, noise reduction, compressor, dialogue enhancement).

## Installation
1. Install Python 3.11 (as noted in `Notes`).
2. Create and activate a virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure `ffmpeg` is installed and update `FFMPEG_PATH` in `config.py` if needed.

## Downloading the models
The repository does not include the large model files. Run the following scripts once (with an internet connection) to populate the `models/` directory:
```bash
python download_whisper_model.py        # Whisper transcription models
python download_translation_model.py    # German→English translation model
python download_voices.py               # Kokoro model and voices
```
This will create the folders below and download the required files:
- `models/whisper/`
- `models/translation_model/`
- `models/kokoro_model/` (including `voices/`)

## Running the application
After downloading the models you can run the Gradio interface:
```bash
python app.py
```
The UI will be available at `http://127.0.0.1:7860`.

### Workflow steps
1. **Get & Prepare Audio** – extract audio from a video or upload audio, remove silence and optionally split into chunks.
2. **Transcribe to Text** – run Whisper to obtain a transcript of the audio.
3. **Translate Text** – translate the German transcript to English.
4. **Synthesize Speech** – generate speech from either text using Kokoro TTS. Voices are defined in `config.py` and detailed in `models/kokoro_model/VOICES.md`.
5. **Audio Enhancement Toolbox** (optional) – apply ffmpeg based filters and optionally download a zip of enhanced files.

Transcripts are saved in the `transcripts/` directory when created.

## Repository structure
- `app.py` – main Gradio interface.
- `audio_processing.py` – functions for extracting, cleaning and chunking audio.
- `transcription_logic.py` – Whisper transcription utilities.
- `translation_logic.py` – translation using HuggingFace transformers.
- `synthesis_logic.py` – Kokoro TTS synthesis.
- `audio_enhancement.py` – optional ffmpeg enhancement pipeline.
- `download_whisper_model.py`, `download_translation_model.py`, `download_voices.py` – scripts to download models.
- `config.py` – application settings and voice definitions.

## Notes
The `models/` directory in the repository only contains placeholder files. After running the download scripts the models will be stored locally so the application can run entirely offline.

