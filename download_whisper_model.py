import whisper
import os
from config import AppConfig

# Define the target directory for Whisper models
# This path should match the one you will use in your config.
WHISPER_MODELS_DIR = "./models/whisper"

def download_all_models():
    """
    Downloads all Whisper models specified in the AppConfig to a local directory.
    This script needs to be run once with an internet connection.
    """
    print(f"Starting download of Whisper models to '{WHISPER_MODELS_DIR}'...")
    os.makedirs(WHISPER_MODELS_DIR, exist_ok=True)

    for model_name in AppConfig.WHISPER_MODELS:
        print(f"\nDownloading model: '{model_name}'...")
        try:
            # The whisper.load_model function will handle the download.
            # We specify `download_root` to control the location.
            whisper.load_model(model_name, download_root=WHISPER_MODELS_DIR)
            print(f"Successfully downloaded '{model_name}'.")
        except Exception as e:
            print(f"Failed to download model '{model_name}'. Error: {e}")
            print("Please check your internet connection and try again.")
            break # Stop if one model fails

    print("\nAll specified models have been downloaded.")
    print(f"You can now run the main application offline.")

if __name__ == "__main__":
    download_all_models()