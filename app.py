# app.py

import torch
from flask import Flask, request, render_template, jsonify
from sentence_transformers.cross_encoder import CrossEncoder

app = Flask(__name__)

# --- Load the pre-quantized small model ---
print("Loading the quantized model...")
model = torch.load('quantized_model.pth', weights_only=False)
model.eval()
print("Quantized model loaded successfully.")

# --- ROUTE 1: The Web Interface ---
@app.route("/", methods=["GET", "POST"])
def index():
    score = None
    text1 = ""
    text2 = ""
    
    if request.method == "POST":
        text1 = request.form.get("text1", "").strip()
        text2 = request.form.get("text2", "").strip()
        
        if text1 and text2:
            raw_score = model.predict((text1, text2), show_progress_bar=False)
            score = f"{float(raw_score):.4f}"
            
    return render_template("index.html", similarity_score=score, text1=text1, text2=text2)

# --- ROUTE 2: The API Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles API requests. Expects a JSON body and returns a JSON response.
    """
    data = request.get_json()

    if not data or "text1" not in data or "text2" not in data:
        return jsonify({"error": "Request body must contain 'text1' and 'text2' keys."}), 400

    text1 = data["text1"]
    text2 = data["text2"]

    raw_score = model.predict((text1, text2), show_progress_bar=False)
    
    response_data = {
        "similarity score": float(raw_score)
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)