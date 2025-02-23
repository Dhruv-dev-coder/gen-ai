import streamlit as st
from transformers import pipeline

@st.cache_resource()
def load_text_generator():
    return pipeline("text-generation", model="gpt2")


text_generator = load_text_generator()
st.title("AI-Powered Content Generator")


st.subheader("Generate AI-based Text")
prompt = st.text_area("Enter a prompt:")
if st.button("Generate Text"):
    with st.spinner("Generating..."):
        result = text_generator(prompt, max_length=100)[0]['generated_text']
    st.success("Generated Text:")
    st.write(result)


