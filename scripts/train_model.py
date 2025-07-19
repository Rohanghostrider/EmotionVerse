import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# 📁 Paths
input_path = "data/processed_dataset.csv"
model_path = "models/emotion_model.pkl"

# 📖 Load data
if os.path.exists(input_path):
    df = pd.read_csv(input_path)
else:
    print(f"⚠️ File not found: {input_path}")
    exit()

# 💬 Features and labels
X = df["clean_text"]
y = df["label_encoded"]

# 🧪 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔮 Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# 📊 Report
y_pred = pipeline.predict(X_test)
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred))

# 💾 Save model
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, model_path)
print(f"✅ Model saved to: {model_path}")
