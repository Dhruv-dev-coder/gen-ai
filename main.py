import streamlit as st
from transformers import pipeline


# Load text generation model
@st.cache_resource
def load_text_generator():
    return pipeline("text-generation", model="gpt2-medium")  # Same model, just ensuring caching


# Load emotion classification model
@st.cache_resource
def load_emotion_classifier():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")


# Load models
text_generator = load_text_generator()
emotion_classifier = load_emotion_classifier()


# Function to detect emotion from text
def detect_emotion(text):
    emotion = emotion_classifier(text)
    return emotion[0]['label'].lower()  # Ensures consistency in case formatting


# Streamlit UI
st.title("🎭 AI-Powered Emotion-Based Story Generator")

st.subheader("📖 Generate AI-based Text Based on Emotion")
ip = st.text_area("✍ Enter a prompt:")

# Check if input is provided
if ip.strip():
    detected_emotion = detect_emotion(ip)
    st.write(f"🔍 Detected Emotion: {detected_emotion.capitalize()}")

    # Define prompts based on detected emotion
    emotion_prompts = {
        "happiness": "Generate a short, heartwarming story about people celebrating a joyful event together. Add an unexpected twist that makes the moment even more special.",
        "sadness": "Generate a short story about someone who is feeling sad but discovers an unexpected way to feel better. Include a surprising moment that changes their perspective.",
        "fear": "Generate a short story about a character facing their greatest fear. Add an unexpected twist that reshapes their understanding of courage.",
        "anger": "Generate a short story about someone struggling with anger but learning to control it. Include an unpredictable event that leads them to find inner peace.",
        "surprise": "Generate a short story about receiving an unexpected gift or news that changes everything. Include a twist that makes the surprise even more impactful.",
        "disgust": "Generate a short story about a character who initially finds something repulsive but learns to see it in a new light. Add an unexpected lesson that changes their perspective.",
        "anticipation": "Generate a short story about someone eagerly waiting for a life-changing event. Include a twist that turns their expectations upside down.",
        "joy": "Generate a short story about a moment of pure joy, where everything feels perfect. Add an unexpected element that makes the experience even more meaningful.",
        "guilt": "Generate a short story about a character dealing with guilt and trying to make things right. Include a twist that changes how they view their past actions.",
        "shame": "Generate a short story about someone overcoming shame and learning self-acceptance. Add a surprising event that helps them heal.",
        "regret": "Generate a short story about letting go of regret and moving forward. Include an unexpected realization that brings closure.",
        "embarrassment": "Generate a short story about someone facing an embarrassing moment but finding humor in the situation. Add a twist that turns their embarrassment into an opportunity.",
        "loneliness": "Generate a short story about a character feeling lonely but unexpectedly discovering connection and comfort. Include an event that changes their view on solitude.",
        "nostalgia": "Generate a short story about someone reflecting on fond memories and learning to appreciate the present. Add an unexpected reminder from the past.",
        "hope": "Generate a short story about finding hope after difficult times. Include an unexpected event that reignites the character’s optimism.",
        "excitement": "Generate a short story about a character eagerly anticipating an exciting event. Add a twist that makes the moment even more thrilling.",
        "contentment": "Generate a short story about someone finding peace in life's simple pleasures. Include an unexpected moment that deepens their appreciation.",
        "relief": "Generate a short story about a character overcoming a stressful situation and feeling immense relief. Add a twist that makes their victory even sweeter.",
        "pride": "Generate a short story about someone feeling proud of an accomplishment. Include an unexpected moment that makes their success even more meaningful.",
        "confusion": "Generate a short story about a character struggling with confusion but eventually finding clarity. Add a surprising revelation that changes their understanding.",
        "jealousy": "Generate a short story about a character dealing with jealousy but learning to focus on personal growth. Include an unexpected lesson that reshapes their feelings.",
        "boredom": "Generate a short story about someone stuck in boredom who unexpectedly discovers something exciting. Add a twist that changes their routine forever.",
        "affection": "Generate a short story about the power of affection in strengthening relationships. Include a surprising act of love that changes everything.",
        "love": "Generate a short story about experiencing deep, genuine love that overcomes obstacles. Add an unexpected challenge that makes their bond even stronger.",
        "compassion": "Generate a short story about a character showing compassion to someone in need. Include a twist where kindness leads to an unexpected reward.",
        "trust": "Generate a short story about building trust and deepening relationships. Add an unexpected test of trust that changes everything.",
        "gratitude": "Generate a short story about expressing gratitude for life’s simple joys. Include an unexpected realization that makes gratitude even more meaningful.",
        "sympathy": "Generate a short story about someone showing sympathy and making a positive impact. Add a twist that makes their gesture even more powerful.",
        "caring": "Generate a short story about the importance of caring for others. Include an unexpected moment that shows how kindness comes full circle.",
        "admiration": "Generate a short story about admiration that inspires positive change. Add an unexpected connection between the characters.",
        "frustration": "Generate a short story about overcoming frustration and finding success despite challenges. Include an unexpected breakthrough that changes the character’s mindset.",
        "anxiety": "Generate a short story about a character facing anxiety and learning to cope. Add an unexpected moment of support that helps them feel at ease.",
        "stress": "Generate a short story about managing stress and finding balance in life. Include a twist that changes how the character approaches pressure.",
        "hopelessness": "Generate a short story about a character feeling hopeless but discovering new possibilities. Add an unexpected source of inspiration that gives them strength.",
        "despair": "Generate a short story about finding light after a period of despair. Include a twist that changes their journey toward healing.",
        "helplessness": "Generate a short story about someone feeling helpless but eventually finding empowerment. Add an unexpected ally who helps them regain control."
    }

    # Get the corresponding prompt or use a default fallback
    prompt = emotion_prompts.get(detected_emotion, "Generate a short, meaningful story about overcoming challenges.")

    # Generate text button
    if st.button("🚀 Generate Story"):
        with st.spinner("⏳ Generating your story..."):
            result = text_generator(
                prompt,
                min_length=200,
                max_length=500,  # Reduced for faster output
                temperature=0.7,  # Adjusted for speed
                top_p=0.9,  # Balanced sampling
                num_return_sequences=1,
                pad_token_id=50256,  # Fixes token issue
                eos_token_id=50256,
                do_sample=True,
                truncation=True# Ensures proper stopping
            )
        st.success("✅ Generated Story:")
        st.write(result[0]['generated_text'])  # Ensures story is displayed properly
else:
    st.warning("⚠ Please enter a prompt to generate a story.")