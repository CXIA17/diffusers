# create_comparison.py
from PIL import Image
import os

def create_comparison_grid():
    prompts = [
        "a_photo_of_sks_dog",
        "a_photo_of_sks_dog_in_a_bucket",
        "a_photo_of_sks_dog_playing_wit",
        "a_photo_of_sks_dog_on_a_beach"
    ]
    
    for idx, prompt_stub in enumerate(prompts):
        # Load baseline and rational images
        baseline_path = f"output/test_images/baseline_{idx}_{prompt_stub}.png"
        rational_path = f"output/test_images_rational/rational_{idx}_{prompt_stub}.png"
        
        if os.path.exists(baseline_path) and os.path.exists(rational_path):
            baseline_img = Image.open(baseline_path)
            rational_img = Image.open(rational_path)
            
            # Create side-by-side
            width = baseline_img.width + rational_img.width
            height = max(baseline_img.height, rational_img.height)
            
            comparison = Image.new('RGB', (width, height))
            comparison.paste(baseline_img, (0, 0))
            comparison.paste(rational_img, (baseline_img.width, 0))
            
            comparison.save(f"output/comparison_{idx}_{prompt_stub}.png")
            print(f"Created comparison_{idx}_{prompt_stub}.png")

create_comparison_grid()