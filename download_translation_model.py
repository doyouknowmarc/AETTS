import os
import shutil
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration ---
MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
SAVE_DIRECTORY = "./models/translation_model"

def download_translation_model():
    """
    Downloads and saves the translation model and tokenizer to a local directory.
    If the target directory already exists and contains files, the user will be
    asked whether to overwrite it. Run this script once while you have an
    internet connection.
    """
    if os.path.exists(SAVE_DIRECTORY):
        if os.listdir(SAVE_DIRECTORY):
            response = input(
                f"Directory '{SAVE_DIRECTORY}' already exists and is not empty. Overwrite? [y/N]: "
            )
            if response.strip().lower() != "y":
                print("Skipping download.")
                return
            shutil.rmtree(SAVE_DIRECTORY)
        else:
            shutil.rmtree(SAVE_DIRECTORY)

    print(f"Creating directory: {SAVE_DIRECTORY}")
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

    print(f"Downloading model '{MODEL_NAME}' from Hugging Face Hub...")
    try:
        # Download and save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(SAVE_DIRECTORY)

        # Download and save the model
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.save_pretrained(SAVE_DIRECTORY)

        print("\n" + "="*50)
        print("✅ Model downloaded successfully!")
        print(f"   Saved to: {os.path.abspath(SAVE_DIRECTORY)}")
        print("You can now run the main application offline.")
        print("="*50)

    except Exception as e:
        print(f"\n❌ An error occurred during download: {e}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    download_translation_model()