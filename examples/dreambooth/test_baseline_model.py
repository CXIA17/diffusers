# test_baseline_model.py
from diffusers import StableDiffusionPipeline
import torch
import os

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

# Load your LoRA weights
pipe.load_lora_weights("output/baseline_lora", weight_name="pytorch_lora_weights.safetensors")

# Create output directory for test images
os.makedirs("output/test_images", exist_ok=True)

# Test with various prompts
test_prompts = [
    "a photo of sks dog",
    "a photo of sks dog in a bucket",
    "a photo of sks dog playing with a ball", 
    "a photo of sks dog on a beach",
    "a photo of sks dog wearing a hat",
    "a photo of sks dog sleeping"
]

for i, prompt in enumerate(test_prompts):
    print(f"Generating: {prompt}")
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    filename = f"output/test_images/baseline_{i}_{prompt.replace(' ', '_')[:30]}.png"
    image.save(filename)
    print(f"  Saved: {filename}")

print("\nAll test images generated in output/test_images/")