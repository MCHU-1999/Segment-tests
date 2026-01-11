from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import torch

# 1. Load Model
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# 2. Load Image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 3. Run Inference
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

# 4. Post-processing (The part missing from your snippet)
# Rescale logits to original image size
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1], # (height, width)
    mode="bilinear",
    align_corners=False,
)

# Get the most likely class for each pixel
pred_seg = upsampled_logits.argmax(dim=1)[0]

# 5. Print what building elements were found
# ADE20K has 150 classes. We can access the names via the model config.
id2label = model.config.id2label

# Get unique class IDs found in this image
unique_classes = torch.unique(pred_seg).tolist()

print("Elements found in this image:")
for class_id in unique_classes:
    class_name = id2label[class_id]
    print(f"- {class_name}")

# If you specifically want to check for building parts:
building_parts = ['wall', 'building', 'sky', 'floor', 'ceiling', 'window', 'door']
found_parts = [id2label[c] for c in unique_classes if id2label[c] in building_parts]

print(f"\nSpecific Building Parts found: {found_parts}")
