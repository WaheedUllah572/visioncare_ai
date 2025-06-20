import base64
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision
import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def convert_image_to_base64(image_file):
    """Convert uploaded image to base64 format for OpenAI API."""
    image = Image.open(image_file).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_b64

def extract_text_from_image(image_file):
    """Use GPT-4o to extract text from an uploaded image."""
    img_b64 = convert_image_to_base64(image_file)
    prompt = "Extract all readable text from this image. Focus on documents, signs, or labels. Return only the plain text."

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
        return res.choices[0].message.content
    except Exception as e:
        return f"Error during OCR: {e}"

def load_detection_model():
    """Load a pre-trained Faster R-CNN object detection model from torchvision."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect_objects(image, model, threshold=0.6):
    """Run object detection on the input image and return labels/boxes above threshold."""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        preds = model(image_tensor)[0]

    result = []
    for i in range(len(preds["boxes"])):
        score = preds["scores"][i].item()
        if score >= threshold:
            label = preds["labels"][i].item()
            box = preds["boxes"][i].tolist()
            result.append({
                "label": label,
                "box": box,
                "score": score
            })
    return result

def draw_boxes(image, predictions):
    """Draw bounding boxes and labels on the image."""
    import matplotlib.pyplot as plt
    from torchvision.utils import draw_bounding_boxes
    import torchvision.transforms.functional as F
    import torchvision

    labels_map = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]

    boxes = torch.tensor([p["box"] for p in predictions])
    labels = [labels_map[p["label"]] for p in predictions]

    if boxes.nelement() == 0:
        return image

    img_tensor = transforms.ToTensor()(image)
    img_boxed = draw_bounding_boxes(img_tensor, boxes, labels=labels, colors="red", width=3, font_size=16)
    return F.to_pil_image(img_boxed)
