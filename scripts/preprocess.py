import pandas as pd
import re
import os
import nltk
from nltk.stem import WordNetLemmatizer

# 📦 Safe downloads (no punkt_tab needed)
nltk.download("wordnet")
nltk.download("omw-1.4")  # Lemma support

# 🧙‍♂️ Lemmatizer setup
lemmatizer = WordNetLemmatizer()

# 📁 File paths
input_path = "data/training_dataset.csv"
output_path = "data/processed_dataset.csv"

# 🧼 Clean + Lemmatize (safe version using .split())
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    tokens = text.split()  # 🔁 Using split to avoid punkt error
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)

# 🧠 Label encoding
label_map = {"pos": 1, "neg": 0, "neu": 2}

# 🔧 Process the file
if os.path.exists(input_path):
    df = pd.read_csv(input_path)
    df["clean_text"] = df["text"].apply(clean_text)
    df["label_encoded"] = df["label"].map(label_map)
    
    df_out = df[["clean_text", "language", "label_encoded"]]
    os.makedirs("data", exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"✅ Preprocessed dataset saved to: {output_path}")
else:
    print(f"⚠️ File not found: {input_path}")
