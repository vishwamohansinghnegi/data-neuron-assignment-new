# app.py

import torch
from flask import Flask, request, render_template
# You need to import the class so PyTorch knows what it is when unpickling.
from sentence_transformers.cross_encoder import CrossEncoder


app = Flask(__name__)

# --- Load the pre-quantized small model ---
print("Loading the quantized model from 'quantized_cross_encoder.pth'...")
# --- KEY CHANGE: Add weights_only=False ---
model = torch.load('quantized_model.pth', weights_only=False)
model.eval()
print("Quantized model loaded successfully.")

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

if __name__ == '__main__':
    app.run(debug=True)