import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load text generation model
@st.cache_resource
def load_text_generator():
    model_name = "HuggingFaceH4/zephyr-7b-alpha"  # Open-access model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load emotion classification model
@st.cache_resource
def load_emotion_classifier():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Function to detect emotion from text
def detect_emotion(text):
    emotion = emotion_classifier(text)
    return emotion[0]['label']

# Load models
text_generator = load_text_generator()
emotion_classifier = load_emotion_classifier()

# Streamlit UI
st.title("🎭 AI-Powered Emotion-Based Story Generator")
st.subheader("📖 Generate AI-based Text Based on Emotion")
ip = st.text_area("✍ Enter a prompt:")

# Check if input is provided
if ip.strip():
    detected_emotion = detect_emotion(ip)
    st.write(f"🔍 Detected Emotion: **{detected_emotion.capitalize()}**")

    # Define prompt
    story_prompt = f"Once upon a time, a character experienced {detected_emotion}. One day, something unexpected happened that changed everything. What followed was an incredible journey of discovery and transformation..."

    # Generate text button
    if st.button("🚀 Generate Story"):
        with st.spinner("⏳ Generating your story..."):
            result = text_generator(
                story_prompt,
                min_length=200,
                max_length=500,
                temperature=0.6,
                top_p=0.95,
                num_return_sequences=1,
                repetition_penalty=1.2,
                do_sample=True,
                truncation=True
            )
        st.success("✅ Generated Story:")
        st.write(result[0]['generated_text'])
else:
    st.warning("⚠ Please enter a prompt to generate a story.")
