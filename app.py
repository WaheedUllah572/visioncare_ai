import streamlit as st
import os
from PIL import Image
from utils import (
    convert_image_to_base64, extract_text_from_image,
    load_detection_model, detect_objects, draw_boxes
)
from openai import OpenAI

# Load object detection model
detection_model = load_detection_model()

# Streamlit page setup
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

# ✅ File uploader with mobile support
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
    key="file_uploader",
    label_visibility="visible"
)

# ✅ Show file upload success/failure
if uploaded_file:
    st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
    st.sidebar.image(uploaded_file, use_container_width=True)
else:
    st.sidebar.warning("📷 Please upload an image file.")

# Buttons row
btn1, btn2, btn3, btn4 = st.columns(4)
describe_btn = btn1.button("🏞️ Describe Scene")
object_btn = btn2.button("🚧 Detect Objects")
assist_btn = btn3.button("🤖 Assist")
text_btn = btn4.button("📝 Extract Text")

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")  # Ensure format compatibility
    except Exception as e:
        st.error(f"❌ Error opening image: {e}")
        st.stop()

    # 🏞️ Scene Description
    if describe_btn:
        with st.spinner("Analyzing scene..."):
            img_b64 = convert_image_to_base64(uploaded_file)
            prompt = "Describe the image simply for a blind person. Include objects, actions, people, and environment."
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
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
                st.error(f"❌ Error: {e}")

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
            img_b64 = convert_image_to_base64(uploaded_file)
            assist_prompt = "Analyze this image and describe any helpful context or tasks it relates to (e.g., reading a label, recognizing a product)."
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
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
                st.error(f"❌ Error: {e}")

    # 📝 OCR Text Extraction
    if text_btn:
        with st.spinner("Extracting text..."):
            img_b64 = convert_image_to_base64(uploaded_file)
            ocr_prompt = "Extract all readable text from this image as accurately as possible. Return plain text only."
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
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
                st.error(f"❌ Error: {e}")
