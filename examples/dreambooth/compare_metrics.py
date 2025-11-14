# compare_metrics.py
import torch
import time
from diffusers import StableDiffusionPipeline

def benchmark_model(model_type, model_path):
    print(f"\nBenchmarking {model_type}...")
    
    # Load model
    if model_type == "baseline":
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16
        ).to("cuda")
        pipe.load_lora_weights(model_path)
    else:  # rational
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to("cuda")
    
    # Count parameters
    total_params = sum(p.numel() for p in pipe.unet.parameters())
    trainable_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
    
    # Warm up
    _ = pipe("test", num_inference_steps=10).images[0]
    
    # Benchmark inference
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(3):
        _ = pipe("a photo of sks dog", num_inference_steps=50).images[0]
    
    torch.cuda.synchronize()
    avg_time = (time.time() - start) / 3
    
    # Memory usage
    memory = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'inference_time': avg_time,
        'memory_gb': memory
    }

# Run benchmarks
results = {}
results['baseline'] = benchmark_model("baseline", "output/baseline_lora")
results['rational'] = benchmark_model("rational", "output/rational_full")

print("\n=== Performance Comparison ===")
for name, metrics in results.items():
    print(f"\n{name.upper()}:")
    print(f"  Total params: {metrics['total_params']:,}")
    print(f"  Trainable params: {metrics['trainable_params']:,}")
    print(f"  Avg inference: {metrics['inference_time']:.2f}s")
    print(f"  Memory: {metrics['memory_gb']:.2f} GB")