import torch
import pandas as pd

# --- 1. CONFIGURATION ---
MODEL_PATH = 'quantized_model.pth'
CSV_FILE = 'DataNeuron_Text_Similarity.csv'
NUMBER_OF_SAMPLES = 5

# --- 2. LOAD THE MODEL ---
print(f"Loading model from '{MODEL_PATH}'...")
model = torch.load(MODEL_PATH, weights_only=False)
model.eval()
print("Model loaded successfully.")

# --- 3. LOAD AND SAMPLE THE DATA ---
print(f"\nLoading data from '{CSV_FILE}'...")
df = pd.read_csv(CSV_FILE)
# Take a random sample of rows from the DataFrame
sample_df = df.sample(n=NUMBER_OF_SAMPLES)
print(f"Successfully loaded and sampled {NUMBER_OF_SAMPLES} random pairs.")

# --- 4. PROCESS THE RANDOM PAIRS ---
print("\n--- Calculating Scores for Random Pairs ---")

# Iterate through each row of the sampled DataFrame
for index, row in sample_df.iterrows():
    text1 = row['text1']
    text2 = row['text2']
    
    print(f"\n--- Pair {index+1} ---")
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    
    # Calculate the score for the pair
    score = model.predict((text1, text2), show_progress_bar=False)
    
    print(f"Similarity Score: {float(score):.4f}")