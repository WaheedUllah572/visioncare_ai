from PIL import Image
from io import BytesIO
import base64
from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def extract_text_from_image(image_file):
    try:
        # Convert image to base64 string
        img = Image.open(image_file)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Send image to GPT-4o for OCR-style extraction
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please extract all visible text from this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]
                }
            ],
            max_tokens=1024
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ OCR via GPT-4o failed: {e}"
