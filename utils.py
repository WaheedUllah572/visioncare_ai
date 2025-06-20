import base64
import io
from PIL import Image, ImageDraw
import pytesseract
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

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

def convert_image_to_base64(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_text_from_image(uploaded_file):
    img = Image.open(uploaded_file)
    text = pytesseract.image_to_string(img)
    return text if text.strip() else "No text found."

def load_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect_objects(image, model, threshold=0.5):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)
    with torch.no_grad():
        preds = model([img_tensor])[0]
    keep = torch.ops.torchvision.nms(preds["boxes"], preds["scores"], 0.5)
    return {
        "boxes": preds["boxes"][keep],
        "labels": preds["labels"][keep],
        "scores": preds["scores"][keep]
    }

def draw_boxes(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if score > threshold:
            x1, y1, x2, y2 = box
            label_name = COCO_CLASSES[label.item()]
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
            draw.text((x1, y1), f"{label_name} ({score:.2f})", fill="black")
    return image
