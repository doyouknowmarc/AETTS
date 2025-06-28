import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration ---
MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
SAVE_DIRECTORY = "./models/translation_model"

def download_translation_model():
    """
    Downloads and saves the translation model and tokenizer to a local directory.
    Run this script once while you have an internet connection.
    """
    if os.path.exists(SAVE_DIRECTORY) and os.listdir(SAVE_DIRECTORY):
        print(f"Model directory '{SAVE_DIRECTORY}' already exists and is not empty. Skipping download.")
        print("If you need to re-download, please delete this directory first.")
        return

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