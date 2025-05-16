from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

# Load and preprocess image
image_path = r"C:\Users\huyho\OneDrive\Desktop\stuff\p-projects\rag-multimodal\data\diagrams\arch_file.png"
image = Image.open(image_path).convert("RGB")

# Load processor and model with offloading support
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    device_map="auto",              # handles offloading automatically
    torch_dtype=torch.float16       # enables mixed precision
)

# Prompt
prompt = "Describe this image"

# Move inputs to correct device
device = model.device  # get the proper device from model
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

# Generate
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Output
print("Prompt:", prompt)
print("Response:", generated_text)
