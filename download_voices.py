import os
from huggingface_hub import hf_hub_download
from config import AppConfig # We'll reuse the voice list from our config

def download_kokoro_assets():
    """
    Downloads the main Kokoro model and all specified voice files to local directories.
    Run this script once while you have an internet connection.
    """
    # --- Define local paths ---
    KOKORO_MODEL_PATH = "./models/kokoro_model"
    VOICES_PATH = os.path.join(KOKORO_MODEL_PATH, "voices")

    # --- Check if already downloaded ---
    if os.path.exists(VOICES_PATH) and len(os.listdir(VOICES_PATH)) >= len(AppConfig.KOKORO_VOICES):
        print("Kokoro assets appear to be already downloaded. Skipping.")
        print(f"   Model path: {os.path.abspath(KOKORO_MODEL_PATH)}")
        print("If you need to re-download, please delete this directory first.")
        return

    print("Creating local directories for Kokoro...")
    os.makedirs(VOICES_PATH, exist_ok=True)

    # --- Download the main model files ---
    repo_id = "hexgrad/Kokoro-82M"
    files_to_download = [
        "config.json",
        "af_heart.pt",
        "vocab.json",
        "special_tokens_map.json",
        "tokenizer_config.json"
    ]
    print(f"\nDownloading main model files from {repo_id}...")
    try:
        for file in files_to_download:
            print(f"  - Downloading {file}...")
            hf_hub_download(repo_id=repo_id, filename=file, local_dir=KOKORO_MODEL_PATH, local_dir_use_symlinks=False)
    except Exception as e:
        print(f"\n❌ Failed to download main model files: {e}")
        return

    # --- Download all voice files ---
    print(f"\nDownloading voice files...")
    all_voices = AppConfig.KOKORO_VOICES.values()
    for voice_id in all_voices:
        try:
            print(f"  - Downloading voice: {voice_id}.pt")
            # This logic correctly places the voice file in the 'voices' subdirectory
            hf_hub_download(
                repo_id=repo_id,
                filename=f"voices/{voice_id}.pt",
                local_dir=VOICES_PATH,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download with this structure will save to .../voices/voices/voice.pt
            # We must move it up one level.
            downloaded_path = os.path.join(VOICES_PATH, "voices", f"{voice_id}.pt")
            target_path = os.path.join(VOICES_PATH, f"{voice_id}.pt")
            if os.path.exists(downloaded_path):
                os.rename(downloaded_path, target_path)

        except Exception as e:
            print(f"  - ❌ Failed to download voice {voice_id}: {e}")

    # Clean up the now-empty nested 'voices' directory
    nested_voices_dir = os.path.join(VOICES_PATH, "voices")
    if os.path.exists(nested_voices_dir) and not os.listdir(nested_voices_dir):
        os.rmdir(nested_voices_dir)


    print("\n" + "="*50)
    print("✅ Kokoro assets downloaded successfully!")
    print(f"   Saved to: {os.path.abspath(KOKORO_MODEL_PATH)}")
    print("You can now run the main application offline.")
    print("="*50)

if __name__ == "__main__":
    download_kokoro_assets()