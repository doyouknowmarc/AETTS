import torch

class AppConfig:
    # --- Paths ---
    FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"
    LOCAL_TRANSLATION_MODEL_PATH = "./models/translation_model"
    LOCAL_KOKORO_MODEL_PATH = "./models/kokoro_model"
    # --- ADD THIS LINE ---
    WHISPER_MODELS_PATH = "./models/whisper"

    # --- Model & Language Defaults ---
    DEFAULT_WHISPER_MODEL = "tiny"
    DEFAULT_LANGUAGE = "de"
    DEFAULT_KOKORO_VOICE_LABEL = "EN: Heart" # Default voice for TTS UI

    # --- Available Options for UI ---
    WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
    LANGUAGES = ["de", "en", "fr", "es", "it"]

    # --- Kokoro TTS Configuration ---
    KOKORO_LANG_MAP = {
        "en": "a",
        "es": "e",
        "fr": "f",
        "it": "i",
    }
    # Structure: "UI Label": ("voice_id", "language_code")
    KOKORO_VOICES = {
        "EN: Heart": ("af_heart", "en"),
        "EN: Michael": ("am_michael", "en"),
        "EN: Adam": ("am_adam", "en"),
        "EN: Sky": ("af_sky", "en"),
        "EN: Santa": ("am_santa", "en"),
        "EN: Onyx": ("am_onyx", "en"),
        "EN: George": ("bm_george", "en"),
        "EN: Lewis": ("bm_lewis", "en"),
        "EN: Daniel": ("bm_daniel", "en"),
        "EN: OMEGA": ("hm_omega", "en"),
        "EN: Alpha": ("hf_alpha", "en"),
        "ES: Spanish Man": ("em_tales", "es"),
        "FR: French Lady": ("ff_jolanda", "fr"),
        "IT: Italian Man": ("im_tony", "it"),
        # Example for a German voice if one were added
        # "DE: German Voice": ("de_voice_id", "de"),
    }

# --- System Checks ---
CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()