# test_rational_model.py
import os
from pathlib import Path
from diffusers import StableDiffusionPipeline
import torch

# ---------------------------
# Paths
# ---------------------------
pipeline_path = Path("./output/rational_stable").expanduser().resolve()
output_dir = Path("output/test_images_rational").expanduser().resolve()
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Load pipeline (handles meta tensors)
# ---------------------------
print(f"Loading pipeline from {pipeline_path}...")
pipe = StableDiffusionPipeline.from_pretrained(
    str(pipeline_path),
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
    local_files_only=True,
    low_cpu_mem_usage=True,   # must stay True
    device_map="cuda"         # load directly on GPU
)


# ---------------------------
# Prompts for testing
# ---------------------------
test_prompts = [
    "a photo of sks dog",
    "a photo of sks dog in a bucket",
    "a photo of sks dog playing with a ball",
    "a photo of sks dog on a beach",
    "a photo of sks dog wearing a hat",
    "a photo of sks dog sleeping"
]

# ---------------------------
# Generate images
# ---------------------------
for i, prompt in enumerate(test_prompts):
    print(f"Generating: {prompt}")
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    filename = output_dir / f"rational_{i}_{prompt.replace(' ', '_')[:30]}.png"
    image.save(filename)
    print(f"Saved: {filename}")

print("\nRational model test complete!")
