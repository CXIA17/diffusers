# debug_rational_init.py
import torch
import sys
sys.path.append('.')
from modules.custom import RationalWrapper

# Test rational function in isolation
print("Testing rational function initialization...")

# Create a simple test
linear = torch.nn.Linear(768, 768)
rational = RationalWrapper(
    pretrained_layer=linear,
    numerator_degree=5,
    denominator_degree=4,
    init_method='gelu',
    freeze_linear=True
)

# Test forward pass
x = torch.randn(1, 768)
try:
    y = rational(x)
    print(f"Output shape: {y.shape}")
    print(f"Output has NaN: {torch.isnan(y).any()}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")
except Exception as e:
    print(f"Error in forward pass: {e}")

# Check parameters
for name, param in rational.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN in {name}")
    print(f"{name}: shape={param.shape}, mean={param.mean():.4f}")