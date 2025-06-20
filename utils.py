from PIL import Image, ImageDraw
import base64
import io
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Convert image to base64 for OpenAI Vision API
def convert_image_to_base64(uploaded_file):
    img = Image.open(uploaded_file)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_b64

# Load object detection model
def load_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

# Detect objects in the image
def detect_objects(image, model, threshold=0.5):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)
    preds = model([img_tensor])[0]
    keep = preds['scores'] > threshold
    return {
        "boxes": preds["boxes"][keep],
        "labels": preds["labels"][keep],
        "scores": preds["scores"][keep]
    }

# Draw object boxes on the image
COCO_CLASSES = [
    "__background__","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","N/A","stop sign","parking meter","bench","bird","cat","dog","horse",
    "sheep","cow","elephant","bear","zebra","giraffe","N/A","backpack","umbrella","N/A","N/A","handbag",
    "tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","N/A","wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","N/A","dining table","N/A","N/A","toilet","N/A","tv","laptop",
    "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","N/A",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def draw_boxes(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    for label, box, score in zip(predictions["labels"], predictions["boxes"], predictions["scores"]):
        if score > threshold:
            x1, y1, x2, y2 = box
            class_name = COCO_CLASSES[label.item()] if label.item() < len(COCO_CLASSES) else "Unknown"
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
            draw.text((x1, y1), f"{class_name} ({score:.2f})", fill="black")
    return image
