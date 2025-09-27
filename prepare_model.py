# prepare_model.py

import torch
from sentence_transformers.cross_encoder import CrossEncoder

def prepare_and_save_model():
    """
    Downloads, quantizes, and saves the Sentence-Transformer model.
    Run this script once locally to prepare the model for deployment.
    """
    model_name = 'cross-encoder/stsb-TinyBERT-L4'
    save_path = 'quantized_model.pth'

    # --- 1. Load the Original Model ---
    print(f"Loading original model: {model_name}...")
    model = CrossEncoder(model_name)
    print("Model loaded successfully.")

    # --- 2. Apply Dynamic Quantization ---
    print("Applying quantization to shrink the model...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print("Quantization complete.")

    # --- 3. Save the Quantized Model ---
    torch.save(quantized_model, save_path)
    print(f"\nQuantized model saved successfully to '{save_path}'")
    print("You can now use this file in your Flask application.")

if __name__ == '__main__':
    prepare_and_save_model()