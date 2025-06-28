import os
import shutil
from huggingface_hub import snapshot_download

def download_kokoro_assets():
    """Download the full Kokoro 82M snapshot with all voices.

    Run this script once while you have an internet connection.
    """
    # --- Define local path ---
    KOKORO_MODEL_PATH = "./models/kokoro_model"

    # --- Check for existing files ---
    if os.path.isdir(KOKORO_MODEL_PATH) and os.listdir(KOKORO_MODEL_PATH):
        response = input(
            f"Directory '{KOKORO_MODEL_PATH}' already exists. Overwrite it? [y/N]: "
        ).strip().lower()
        if response not in ("y", "yes"):
            print("Download cancelled.")
            return
        shutil.rmtree(KOKORO_MODEL_PATH)

    os.makedirs(KOKORO_MODEL_PATH, exist_ok=True)

    repo_id = "hexgrad/Kokoro-82M"
    print(f"\nDownloading Kokoro model snapshot from {repo_id}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=KOKORO_MODEL_PATH,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"\n❌ Failed to download Kokoro assets: {e}")
        return


    print("\n" + "="*50)
    print("✅ Kokoro assets downloaded successfully!")
    print(f"   Saved to: {os.path.abspath(KOKORO_MODEL_PATH)}")
    print("You can now run the main application offline.")
    print("="*50)

if __name__ == "__main__":
    download_kokoro_assets()