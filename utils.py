from PIL import Image
from io import BytesIO
import base64
from openai import OpenAI
import streamlit as st
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def convert_image_to_base64(image_file):
    img = Image.open(image_file)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode()

def extract_text_from_image(image_file):
    try:
        img = Image.open(image_file)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all visible text from this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]
                }
            ],
            max_tokens=1024
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ OCR via GPT-4o failed: {e}"

@st.cache_resource
def load_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect_objects(image, model, threshold=0.5):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)
    predictions = model([img_tensor])[0]
    keep = torch.ops.torchvision.nms(predictions["boxes"], predictions["scores"], 0.5)

    return {
        "boxes": predictions["boxes"][keep],
        "labels": predictions["labels"][keep],
        "scores": predictions["scores"][keep],
    }

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def draw_boxes(image, predictions, threshold=0.5):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if score >= threshold:
            x1, y1, x2, y2 = box
            class_name = COCO_CLASSES[label.item()]
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
            draw.text((x1, y1), f"{class_name} ({score:.2f})", fill="black")
    return image
