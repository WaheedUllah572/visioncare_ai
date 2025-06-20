import streamlit as st
import os
from PIL import Image
from openai import OpenAI

from utils import (
    convert_image_to_base64,
    load_detection_model, detect_objects, draw_boxes
)

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
detection_model = load_detection_model()

# App layout
st.set_page_config(page_title="👁️ VisionCare AI", layout="centered", page_icon="🌟")
st.title("👁️ VisionCare AI: Vision Assistant for All")

st.markdown("""
Empowering Accessibility with AI Vision 💡  
**Features:**
- 🏞️ Scene Description  
- 🚧 Object Detection  
- 🧑‍🤝‍🧑 Personalized Assistance  
- 📝 OCR (via GPT-4o)
---
""")

# Upload section
uploaded_file = st.sidebar.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png", "webp"])
if uploaded_file:
    st.sidebar.image(uploaded_file, use_container_width=True)

# Buttons
btn1, btn2, btn3, btn4 = st.columns(4)
describe_btn = btn1.button("🏞️ Describe Scene")
object_btn = btn2.button("🚧 Detect Objects")
assist_btn = btn3.button("🤖 Assist")
ocr_btn = btn4.button("📝 Extract Text")

# Image processing logic
if uploaded_file:
    img = Image.open(uploaded_file)

    if describe_btn:
        with st.spinner("Analyzing scene..."):
            img_b64 = convert_image_to_base64(uploaded_file)
            prompt = "Describe the image simply for a blind person. Include objects, actions, people, and environment."

            try:
                res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }],
                    max_tokens=1024
                )
                st.subheader("🏞️ Scene Description")
                st.write(res.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {e}")

    if object_btn:
        with st.spinner("Detecting objects..."):
            preds = detect_objects(img, detection_model)
            boxed = draw_boxes(img.copy(), preds)
            st.subheader("🚧 Detected Objects")
            st.image(boxed)

    if assist_btn:
        with st.spinner("Providing assistance..."):
            img_b64 = convert_image_to_base64(uploaded_file)
            assist_prompt = "Analyze this image and describe any helpful context or tasks it relates to (e.g., reading a label, recognizing a product)."

            try:
                res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": assist_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }],
                    max_tokens=1024
                )
                st.subheader("🤖 Assistant Response")
                st.write(res.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {e}")

    if ocr_btn:
        with st.spinner("Extracting text..."):
            img_b64 = convert_image_to_base64(uploaded_file)
            ocr_prompt = "Extract all visible text from this image as cleanly as possible. Do not describe the image, only return raw readable text."

            try:
                res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": ocr_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }],
                    max_tokens=1024
                )
                st.subheader("📝 Extracted Text")
                st.write(res.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {e}")
