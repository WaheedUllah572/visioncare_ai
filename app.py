import streamlit as st
import os
from PIL import Image
from utils import (
    convert_image_to_base64, extract_text_from_image,
    load_detection_model, detect_objects, draw_boxes
)

# Load object detection model once
detection_model = load_detection_model()

st.set_page_config(page_title="👁️ VisionCare AI", layout="centered", page_icon="🌟")
st.title("👁️ VisionCare AI: Vision Assistant for All")

st.markdown("""
Empowering Accessibility with AI Vision 💡  
**Features:**
- 🏞️ Scene Description  
- 🚧 Object Detection  
- 🧑‍🤝‍🧑 Personalized Assistance  
- 📝 OCR (Text Extraction via GPT-4o)
---
""")

# ✅ Updated to support mobile (no strict file types)
uploaded_file = st.sidebar.file_uploader("📂 Upload Image")

# ✅ Display metadata to confirm upload works on mobile
if uploaded_file:
    st.sidebar.image(uploaded_file, use_container_width=True)
    st.success(f"Uploaded: {uploaded_file.name}")
    st.write(f"File type: {uploaded_file.type}")
    st.write(f"File size: {uploaded_file.size} bytes")

btn1, btn2, btn3, btn4 = st.columns(4)
describe_btn = btn1.button("🏞️ Describe Scene")
object_btn = btn2.button("🚧 Detect Objects")
assist_btn = btn3.button("🤖 Assist")
text_btn = btn4.button("📝 Extract Text")

# Core features
if uploaded_file:
    img = Image.open(uploaded_file)

    # 🏞️ Scene description
    if describe_btn:
        with st.spinner("Analyzing scene..."):
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
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

    # 🚧 Object Detection
    if object_btn:
        with st.spinner("Detecting objects..."):
            preds = detect_objects(img, detection_model)
            boxed = draw_boxes(img.copy(), preds)
            st.subheader("🚧 Detected Objects")
            st.image(boxed)

    # 🤖 Task Assistance
    if assist_btn:
        with st.spinner("Providing assistance..."):
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
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

    # 📝 OCR (via GPT-4o)
    if text_btn:
        with st.spinner("Extracting text..."):
            extracted_text = extract_text_from_image(uploaded_file)
            st.subheader("📝 Extracted Text")
            st.write(extracted_text)
