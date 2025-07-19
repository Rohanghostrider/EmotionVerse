import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# 📍 Paths
model_path = "models/emotion_model.pkl"
log_path = "data/dashboard_log.csv"

# 🔮 Load model
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model not found!")

# 🎯 Predict
def predict_sentiment(text):
    return model.predict([text])[0]

# 📝 Log
def log_prediction(text, language, label):
    entry = pd.DataFrame([[text, language, label]], columns=["text", "language", "predicted_label"])
    if os.path.exists(log_path):
        entry.to_csv(log_path, mode="a", header=False, index=False)
    else:
        entry.to_csv(log_path, index=False)

# 📊 Load logs
def load_log_data():
    return pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame(columns=["text", "language", "predicted_label"])

# 🎨 Pie chart
def show_pie_chart(data):
    label_counts = data["predicted_label"].value_counts()
    labels = label_counts.index
    sizes = label_counts.values
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

# 🎨 Emoji meter
def show_emoji_meter(data):
    emoji_map = {"pos": "🥰", "neg": "😠", "neu": "😐"}
    label_counts = data["predicted_label"].value_counts()
    st.subheader("Emoji Meter 💫")
    for label, count in label_counts.items():
        st.write(f"{emoji_map.get(label, '')} × {count}")

# 🚀 UI
st.title("🔮 EmotionVerse — Real-Time Multilingual Sentiment Tracker")
user_text = st.text_input("Enter your review:")
selected_language = st.selectbox("Select language:", ["en", "hi", "fr"])
if st.button("Predict"):
    prediction_label = predict_sentiment(user_text)
    log_prediction(user_text, selected_language, prediction_label)
    st.success(f"Predicted sentiment: **{prediction_label.upper()}**")

log_df = load_log_data()
if not log_df.empty:
    show_pie_chart(log_df)
    show_emoji_meter(log_df)
    st.subheader("📜 Recent Entries:")
    st.dataframe(log_df.tail(10))
