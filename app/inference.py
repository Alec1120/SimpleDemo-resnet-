import onnxruntime
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

session = onnxruntime.InferenceSession("model/resnet18.onnx")
input_name = session.get_inputs()[0].name

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

with open("app/imagenet_classes.txt") as f:
    idx2label = [line.strip() for line in f.readlines()]

def predict(image: Image.Image):
    img_tensor = preprocess(image).unsqueeze(0).numpy()
    outputs = session.run(None, {input_name: img_tensor})
    pred_idx = int(np.argmax(outputs[0]))
    label = idx2label[pred_idx]
    return {"class_id": pred_idx, "label": label}
