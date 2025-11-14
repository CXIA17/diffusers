# ldm/modules/rational_activation.py
"""
Improved Trainable Rational Activation Functions
Combines ideas from KAT with simplified Y = f(Wx) implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np
import json
import os
from pathlib import Path


class ImprovedRationalFunction(nn.Module):
    """
    Improved rational activation function inspired by KAT
    f(x) = P(x) / Q(x) with stability improvements
    """
    def __init__(
        self, 
        numerator_degree=5, 
        denominator_degree=4,
        num_groups=1,
        init_method='gelu',
        shared=True,
        num_features=None,
        use_abs_denominator=True
    ):
        """
        Args:
            numerator_degree: Degree of numerator polynomial
            denominator_degree: Degree of denominator polynomial  
            num_groups: Number of groups for group-wise rational (like KAT)
            init_method: Initialization ('gelu', 'relu', 'tanh', 'identity', etc.)
            shared: If True and num_groups=1, use same function for all features
            num_features: Required if not shared
            use_abs_denominator: Use absolute values in denominator for stability
        """
        super().__init__()
        self.numerator_degree = numerator_degree
        self.denominator_degree = denominator_degree
        self.num_groups = num_groups
        self.use_abs_denominator = use_abs_denominator
        
        # Determine parameter shape based on groups and sharing
        if num_groups > 1:
            # Group-wise rational function
            self.weight_numerator = nn.Parameter(
                torch.zeros(num_groups, numerator_degree + 1)
            )
            self.weight_denominator = nn.Parameter(
                torch.zeros(num_groups, denominator_degree)
            )
        elif shared:
            # Single shared rational function
            self.weight_numerator = nn.Parameter(
                torch.zeros(1, numerator_degree + 1)
            )
            self.weight_denominator = nn.Parameter(
                torch.zeros(1, denominator_degree)
            )
        else:
            # Per-feature rational function
            if num_features is None:
                raise ValueError("num_features required when not shared")
            self.weight_numerator = nn.Parameter(
                torch.zeros(num_features, numerator_degree + 1)
            )
            self.weight_denominator = nn.Parameter(
                torch.zeros(num_features, denominator_degree)
            )
        
        self.num_features = num_features
        self.shared = shared
        
        # Initialize weights
        self._initialize_weights(init_method)
    
    def _initialize_weights(self, method):
        """Initialize weights based on method or from preset configurations"""
        
        # Try to load from JSON file first (like KAT)
        json_path = Path(__file__).parent / 'rational_init.json'
        if json_path.exists() and method in ['gelu', 'relu', 'tanh', 'swish']:
            try:
                with open(json_path) as f:
                    init_data = json.load(f)
                if method in init_data:
                    num_weights = torch.tensor(init_data[method]["numerator"])
                    den_weights = torch.tensor(init_data[method]["denominator"])
                    
                    # Broadcast to correct shape
                    if self.num_groups > 1:
                        self.weight_numerator.data = num_weights.expand(self.num_groups, -1)
                        self.weight_denominator.data = den_weights.expand(self.num_groups, -1)
                    else:
                        self.weight_numerator.data = num_weights.unsqueeze(0)
                        self.weight_denominator.data = den_weights.unsqueeze(0)
                    return
            except:
                pass
        
        # Fallback to programmatic initialization
        with torch.no_grad():
            if method == 'identity':
                # f(x) = x
                self.weight_numerator.data.zero_()
                self.weight_numerator.data[:, 0] = 0  # a_0 = 0
                self.weight_numerator.data[:, 1] = 1  # a_1 = 1
                self.weight_denominator.data.zero_()
                
            elif method == 'gelu':
                # Approximate GELU
                # Based on the rational approximation from KAT paper
                self.weight_numerator.data[:, 0] = 0.0
                self.weight_numerator.data[:, 1] = 0.5
                if self.numerator_degree >= 2:
                    self.weight_numerator.data[:, 2] = 0.3989
                if self.numerator_degree >= 3:
                    self.weight_numerator.data[:, 3] = 0.0
                if self.numerator_degree >= 4:
                    self.weight_numerator.data[:, 4] = 0.0535
                    
                self.weight_denominator.data[:, 0] = 0.1353  # b_1
                if self.denominator_degree >= 2:
                    self.weight_denominator.data[:, 1] = 0.0  # b_2
                if self.denominator_degree >= 3:
                    self.weight_denominator.data[:, 2] = 0.0311  # b_3
                    
            elif method == 'relu':
                # Smooth ReLU approximation
                self.weight_numerator.data[:, 0] = 0.0
                self.weight_numerator.data[:, 1] = 0.5
                if self.numerator_degree >= 2:
                    self.weight_numerator.data[:, 2] = 0.5
                    
                self.weight_denominator.data[:, 0] = 1.0
                
            elif method == 'tanh':
                # Padé approximation of tanh
                self.weight_numerator.data[:, 0] = 0.0
                self.weight_numerator.data[:, 1] = 1.0
                if self.numerator_degree >= 3:
                    self.weight_numerator.data[:, 3] = -0.333
                    
                if self.denominator_degree >= 2:
                    self.weight_denominator.data[:, 1] = 0.333
                    
            else:
                # Random small initialization
                self.weight_numerator.data.normal_(0, 0.01)
                self.weight_numerator.data[:, 1] += 1.0  # Add linear component
                self.weight_denominator.data.normal_(0, 0.01)
    
    def _compute_powers(self, x, max_degree):
        """Efficiently compute powers of x up to max_degree"""
        device = x.device
        dtype = x.dtype
        
        # More efficient computation that ensures device consistency
        powers = []
        x_power = torch.ones_like(x, device=device, dtype=dtype)
        powers.append(x_power)
        
        if max_degree >= 1:
            x_power = x
            powers.append(x_power)
            
        for i in range(2, max_degree + 1):
            x_power = x_power * x  # Reuse previous power
            powers.append(x_power)
        
        return torch.stack(powers, dim=-1)
    
    def forward(self, x):
        """
        Forward pass computing P(x) / Q(x)
        Supports both 2D (batch, features) and 3D (batch, seq_len, features) inputs
        """
        input_shape = x.shape
        device = x.device
        dtype = x.dtype
        
        if self.num_groups > 1:
            # Group-wise rational (like KAT)
            if x.dim() == 2:
                B, D = x.shape
                D_per_group = D // self.num_groups
                x = x.view(B, self.num_groups, D_per_group)
                x = x.transpose(0, 1)  # (groups, batch, features_per_group)
            elif x.dim() == 3:
                B, L, D = x.shape
                D_per_group = D // self.num_groups
                x = x.view(B, L, self.num_groups, D_per_group)
                x = x.permute(2, 0, 1, 3)  # (groups, batch, seq_len, features_per_group)
            
            # Compute powers
            max_degree = max(self.numerator_degree, self.denominator_degree)
            x_flat = x.reshape(self.num_groups, -1)
            powers = self._compute_powers(x_flat, max_degree)
            
            # Compute numerator: P(x) = sum(a_i * x^i)
            num_powers = powers[..., :self.numerator_degree + 1]
            numerator = (self.weight_numerator.unsqueeze(1) * num_powers).sum(dim=-1)
            
            # Compute denominator: Q(x) = 1 + sum(|b_i| * x^i) or 1 + sum(b_i * x^i)
            denom = torch.ones_like(numerator)
            if self.denominator_degree > 0:
                den_powers = powers[..., 1:self.denominator_degree + 1]
                if self.use_abs_denominator:
                    denom = denom + (self.weight_denominator.abs().unsqueeze(1) * den_powers).sum(dim=-1)
                else:
                    denom = denom + (self.weight_denominator.unsqueeze(1) * den_powers).sum(dim=-1)
            
            # Compute rational function
            denom = denom.clamp_min(1e-6)
            output = numerator / denom
            
            # Reshape back
            output = output.reshape(self.num_groups, *input_shape[:-1], D_per_group)
            if x.dim() == 4:  # was 3D input
                output = output.permute(1, 2, 0, 3).reshape(*input_shape)
            else:  # was 2D input
                output = output.transpose(0, 1).reshape(*input_shape)
                
        else:
            # Standard rational function
            powers = self._compute_powers(x, max(self.numerator_degree, self.denominator_degree))
            
            # Numerator - parameters should already be on the right device
            num_powers = powers[..., :self.numerator_degree + 1]
            if self.shared:
                # Ensure weight_numerator is on the same device as num_powers
                numerator = (self.weight_numerator.to(device) * num_powers).sum(dim=-1)
            else:
                numerator = (self.weight_numerator.to(device).unsqueeze(0) * num_powers).sum(dim=-1)
            
            # Denominator
            denom = torch.ones_like(x)
            if self.denominator_degree > 0:
                den_powers = powers[..., 1:self.denominator_degree + 1]
                if self.use_abs_denominator:
                    if self.shared:
                        denom = denom + (self.weight_denominator.to(device).abs() * den_powers).sum(dim=-1)
                    else:
                        denom = denom + (self.weight_denominator.to(device).abs().unsqueeze(0) * den_powers).sum(dim=-1)
                else:
                    if self.shared:
                        denom = denom + (self.weight_denominator.to(device) * den_powers).sum(dim=-1)
                    else:
                        denom = denom + (self.weight_denominator.to(device).unsqueeze(0) * den_powers).sum(dim=-1)
            
            output = numerator / (denom + 1e-8)
        
        return output
    
    def to(self, *args, **kwargs):
        """Override to() to ensure all components move together"""
        self = super().to(*args, **kwargs)
        # Parameters should move automatically since they're registered
        # But we can ensure it here if needed
        return self
    
    def extra_repr(self):
        return f'num_groups={self.num_groups}, numerator_degree={self.numerator_degree}, denominator_degree={self.denominator_degree}'


class RationalWrapper(nn.Module):
    def __init__(
        self,
        pretrained_layer: Optional[nn.Linear] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        numerator_degree: int = 5,
        denominator_degree: int = 4,
        num_groups: int = 1,
        init_method: str = 'gelu',
        freeze_linear: bool = True,
        use_abs_denominator: bool = True,
        use_bias: bool = True
    ):
        super().__init__()
        
        # Setup linear layer
        if pretrained_layer is not None:
            self.linear = pretrained_layer
            out_features = pretrained_layer.out_features
        else:
            if in_features is None or out_features is None:
                raise ValueError("Must provide either pretrained_layer or both in_features and out_features")
            self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        
        # Don't freeze here - will be handled by optimizer selection
        self.freeze_linear = freeze_linear
        
        # Setup rational activation
        self.rational = ImprovedRationalFunction(
            numerator_degree=numerator_degree,
            denominator_degree=denominator_degree,
            num_groups=min(num_groups, out_features),
            init_method=init_method,
            shared=(num_groups == 1),
            num_features=out_features if num_groups == 1 else None,
            use_abs_denominator=use_abs_denominator
        )
    
    def forward(self, x):
        """Y = f(Wx + b)"""
        linear_out = self.linear(x)
        return self.rational(linear_out)
    
    def parameter_count(self):
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        rational_params = sum(p.numel() for p in self.rational.parameters())
        linear_params = sum(p.numel() for p in self.linear.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'rational': rational_params,
            'linear': linear_params,
            'frozen': total - trainable
        }


def create_rational_init_json():
    """Create initialization JSON file with optimal presets"""
    init_data = {
        "gelu": {
            "numerator": [0.0, 0.5, 0.3989, 0.0, 0.0535, 0.0],
            "denominator": [0.1353, 0.0, 0.0311, 0.0]
        },
        "relu": {
            "numerator": [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
            "denominator": [1.0, 0.0, 0.0, 0.0]
        },
        "tanh": {
            "numerator": [0.0, 1.0, 0.0, -0.333, 0.0, 0.0],
            "denominator": [0.0, 0.333, 0.0, 0.0]
        },
        "swish": {
            "numerator": [0.0, 0.5, 0.25, 0.0, 0.0, 0.0],
            "denominator": [0.5, 0.0, 0.0, 0.0]
        }
    }
    
    json_path = Path(__file__).parent / 'rational_init.json'
    with open(json_path, 'w') as f:
        json.dump(init_data, f, indent=2)
    print(f"Created initialization file at {json_path}")


def replace_layers_with_rational(
    model, 
    layer_names=None,
    freeze_linear=True,  # This now means "don't optimize" rather than "don't compute gradients"
    numerator_degree=5,
    denominator_degree=4,
    num_groups=1,
    init_method='gelu',
    use_abs_denominator=True,
    **kwargs
):
    """Replace linear layers with rational wrapped versions"""
    replaced_count = 0
    
    def replace_module(module, name, parent):
        nonlocal replaced_count
        if isinstance(module, nn.Linear):
            if layer_names is None or any(name.endswith(ln) for ln in layer_names):
                rational_wrapper = RationalWrapper(
                    pretrained_layer=module,
                    freeze_linear=False,  # Don't set requires_grad=False
                    numerator_degree=numerator_degree,
                    denominator_degree=denominator_degree,
                    num_groups=num_groups,
                    init_method=init_method,
                    use_abs_denominator=use_abs_denominator,
                    **kwargs
                )
                setattr(parent, name.split('.')[-1], rational_wrapper)
                replaced_count += 1
                return True
        return False
    
    def recursive_replace(parent, prefix=''):
        for name, module in parent.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if not replace_module(module, full_name, parent):
                recursive_replace(module, full_name)
    
    recursive_replace(model)
    
    if replaced_count > 0:
        print(f"Replaced {replaced_count} layers with rational wrappers")
        # Count only rational parameters as trainable
        rational_params = 0
        for module in model.modules():
            if isinstance(module, ImprovedRationalFunction):
                rational_params += sum(p.numel() for p in module.parameters())
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable rational parameters: {rational_params:,} ({100*rational_params/total_params:.3f}%)")

    model = freeze_base_model(model)

    return model

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer: Y = Wx + ABx = (W + AB)x
    """
    def __init__(self, in_features, out_features, rank=16, alpha=1.0, dropout=0.0):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank for low-rank matrices A and B
            alpha: Scaling factor for LoRA
            dropout: Dropout probability for LoRA path
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Frozen pretrained weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False  # Freeze pretrained weights
        
        # LoRA decomposition matrices
        # A: maps from in_features to rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        # B: maps from rank to out_features  
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # Standard linear transformation (frozen)
        result = F.linear(x, self.weight)
        
        # Add LoRA adaptation
        # x: (batch, in_features)
        # lora_A: (rank, in_features) -> F.linear gives (batch, rank)
        # lora_B: (out_features, rank) -> F.linear gives (batch, out_features)
        x_dropped = self.dropout(x)
        lora_output = F.linear(F.linear(x_dropped, self.lora_A), self.lora_B)
        result = result + lora_output * self.scaling
        
        return result
    
    def merge_weights(self):
        """Merge LoRA weights into the main weights for inference."""
        with torch.no_grad():
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            # Reset LoRA parameters after merging
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))


class LoRARationalWrapper(nn.Module):
    """
    Combined LoRA + Rational function: Y = f(Wx + ABx)
    Integrates LoRA adaptation with trainable rational activation
    """
    def __init__(
        self,
        pretrained_layer: Optional[nn.Linear] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        # LoRA parameters
        lora_rank: int = 16,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        # Rational parameters
        numerator_degree: int = 5,
        denominator_degree: int = 4,
        num_groups: int = 1,
        init_method: str = 'gelu',
        use_abs_denominator: bool = True,
        # Training options
        train_mode: str = 'both',  # 'lora_only', 'rational_only', 'both'
        freeze_base: bool = True,
        use_bias: bool = True
    ):
        """
        Args:
            pretrained_layer: Existing linear layer to enhance
            in_features: Input dimension (if creating new)
            out_features: Output dimension (if creating new)
            lora_rank: LoRA decomposition rank
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout for LoRA path
            numerator_degree: Degree of P(x) in rational
            denominator_degree: Degree of Q(x) in rational
            num_groups: Number of rational function groups
            init_method: Rational initialization method
            use_abs_denominator: Use |b_i| in denominator
            train_mode: What to train ('lora_only', 'rational_only', 'both')
            freeze_base: Freeze base linear weights
            use_bias: Include bias term
        """
        super().__init__()
        
        # Setup dimensions
        if pretrained_layer is not None:
            in_features = pretrained_layer.in_features
            out_features = pretrained_layer.out_features
            # Copy pretrained weights
            weight = pretrained_layer.weight.data.clone()
            bias = pretrained_layer.bias.data.clone() if pretrained_layer.bias is not None else None
        else:
            if in_features is None or out_features is None:
                raise ValueError("Must provide either pretrained_layer or both in_features and out_features")
            weight = torch.randn(out_features, in_features) * 0.02
            bias = torch.zeros(out_features) if use_bias else None
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Base linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.linear.weight.data = weight
        if bias is not None and use_bias:
            self.linear.bias.data = bias
        
        # Freeze base weights if requested
        if freeze_base:
            for param in self.linear.parameters():
                param.requires_grad = False
        
        # LoRA components
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / lora_rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(lora_rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Initialize LoRA
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Rational activation function
        self.rational = ImprovedRationalFunction(
            numerator_degree=numerator_degree,
            denominator_degree=denominator_degree,
            num_groups=min(num_groups, out_features),
            init_method=init_method,
            shared=(num_groups == 1),
            num_features=out_features if num_groups == 1 else None,
            use_abs_denominator=use_abs_denominator
        )
        
        # Set initial training mode
        self.set_training_mode(train_mode)
    
    def forward(self, x):
        """
        Compute Y = f(Wx + ABx)
        Modes:
            - 'rational_only': train only rational function
            - 'lora_only': train only LoRA
            - 'both': train both LoRA and rational
        """
        # Base linear output (frozen)
        base_output = self.linear(x)

        # LoRA adaptation (may be frozen depending on mode)
        x_dropped = self.lora_dropout(x)
        lora_output = F.linear(F.linear(x_dropped, self.lora_A), self.lora_B)

        # Combine base + LoRA
        combined = base_output + lora_output * self.lora_scaling

        if self.train_mode == 'rational_only':
            # Detach frozen parts and allow gradients only for rational
            combined = combined.detach().requires_grad_(True)
            return self.rational(combined)

        elif self.train_mode == 'lora_only':
            # Skip rational, just return LoRA-adapted linear output
            return combined

        elif self.train_mode == 'both':
            # Train both: rational receives gradients from combined output
            return self.rational(combined)

        else:
            raise ValueError(f"Unknown train_mode '{self.train_mode}'")




    
    def set_training_mode(self, mode: str):
        """
        Configure which components are trainable.
        mode: 'lora_only', 'rational_only', 'both', 'none'
        """
        self.train_mode = mode

        # LoRA parameters
        lora_grad = mode in ['lora_only', 'both']
        self.lora_A.requires_grad = lora_grad
        self.lora_B.requires_grad = lora_grad

        # Rational parameters
        rational_grad = mode in ['rational_only', 'both']
        for param in self.rational.parameters():
            param.requires_grad = rational_grad

    
    def merge_lora(self):
        """Merge LoRA weights into base for efficient inference"""
        with torch.no_grad():
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.lora_scaling
            # Reset LoRA to identity
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def parameter_count(self):
        """Detailed parameter count"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base_params = sum(p.numel() for p in self.linear.parameters())
        lora_params = self.lora_A.numel() + self.lora_B.numel()
        rational_params = sum(p.numel() for p in self.rational.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'base': base_params,
            'lora': lora_params,
            'rational': rational_params,
            'frozen': total - trainable,
            'efficiency': f"{100 * trainable / total:.2f}%"
        }


def replace_layers_with_lora_rational(
    model,
    layer_names=None,
    lora_rank=16,
    lora_alpha=1.0,
    numerator_degree=5,
    denominator_degree=4,
    num_groups=1,
    init_method='gelu',
    train_mode='both',
    **kwargs
):
    """
    Replace linear layers with LoRA + Rational wrappers
    
    Args:
        model: PyTorch model to modify
        layer_names: Specific layers to replace (None = all)
        lora_rank: LoRA rank
        lora_alpha: LoRA scaling
        numerator_degree: Rational numerator degree
        denominator_degree: Rational denominator degree
        num_groups: Number of rational groups
        init_method: Rational initialization
        train_mode: What to train ('both', 'lora_only', 'rational_only')
        **kwargs: Additional arguments
    
    Returns:
        Modified model with LoRA+Rational layers
    """
    replaced_count = 0
    
    def replace_module(module, name, parent):
        nonlocal replaced_count
        if isinstance(module, nn.Linear):
            if layer_names is None or any(name.endswith(ln) for ln in layer_names):
                lora_rational = LoRARationalWrapper(
                    pretrained_layer=module,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    numerator_degree=numerator_degree,
                    denominator_degree=denominator_degree,
                    num_groups=num_groups,
                    init_method=init_method,
                    train_mode=train_mode,
                    freeze_base=True,
                    **kwargs
                )
                setattr(parent, name.split('.')[-1], lora_rational)
                replaced_count += 1
                return True
        return False
    
    def recursive_replace(parent, prefix=''):
        for name, module in parent.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if not replace_module(module, full_name, parent):
                recursive_replace(module, full_name)
    
    recursive_replace(model)
    
    if replaced_count > 0:
        print(f"\nReplaced {replaced_count} layers with LoRA+Rational wrappers")
        print(f"Configuration:")
        print(f"  LoRA rank: {lora_rank}")
        print(f"  Rational: P_{numerator_degree}(x) / Q_{denominator_degree}(x)")
        print(f"  Groups: {num_groups}")
        print(f"  Training mode: {train_mode}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.3f}%)")
    
    model = freeze_base_model(model)
    
    return model


class ProgressiveLoRARationalTrainer:
    """
    Progressive training strategy for LoRA + Rational models
    """
    def __init__(self, model):
        self.model = model
        self.lora_rational_blocks = [
            m for m in model.modules() 
            if isinstance(m, LoRARationalWrapper)
        ]
        print(f"Found {len(self.lora_rational_blocks)} LoRA+Rational blocks")
    
    def stage1_rational_only(self, lr=1e-3):
        """Stage 1: Train only rational functions"""
        print("\n=== Stage 1: Rational Functions Only ===")
        for block in self.lora_rational_blocks:
            block.set_training_mode('rational_only')
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")
        
        return torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr
        )
    
    def stage2_lora_only(self, lr=1e-3):
        """Stage 2: Train only LoRA weights"""
        print("\n=== Stage 2: LoRA Weights Only ===")
        for block in self.lora_rational_blocks:
            block.set_training_mode('lora_only')
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")
        
        return torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr
        )
    
    def stage3_both(self, lr=1e-4):
        """Stage 3: Fine-tune both together"""
        print("\n=== Stage 3: Both LoRA + Rational ===")
        for block in self.lora_rational_blocks:
            block.set_training_mode('both')
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")
        
        return torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr
        )
    
    def merge_for_inference(self):
        """Merge LoRA weights for deployment"""
        print("\nMerging LoRA weights for inference...")
        for block in self.lora_rational_blocks:
            block.merge_lora()
        print("Merge complete - model ready for deployment")

def freeze_base_model(model):
    """
    Freeze all base model parameters, leaving only LoRA and Rational trainable.
    Call this after replacing layers with LoRA, Rational, or LoRA+Rational.
    """
    # Freeze everything first
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Re-enable LoRA and Rational parameters
    for name, param in model.named_parameters():
        if "lora" in name.lower() or "rational" in name.lower():
            param.requires_grad = True
    
    # Compute stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora" in n.lower())
    rational_params = sum(p.numel() for n, p in model.named_parameters() if "rational" in n.lower())
    
    # Print summary safely
    if total_params > 0:
        print(
            f"Froze base model parameters.\n"
            f"  Total params: {total_params:,}\n"
            f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.3f}%)\n"
            f"    LoRA: {lora_params:,}\n"
            f"    Rational: {rational_params:,}"
        )
    else:
        print("Froze base model parameters. No parameters found in this module.")
    
    return model



# Test the combined implementation
if __name__ == "__main__":
    print("Testing LoRA + Rational Combined Implementation")
    print("=" * 60)
    
    # Test 1: Single layer
    print("\n1. Single LoRA+Rational layer:")
    layer = nn.Linear(512, 256)
    lora_rational = LoRARationalWrapper(
        pretrained_layer=layer,
        lora_rank=8,
        lora_alpha=16,
        numerator_degree=5,
        denominator_degree=4,
        num_groups=4,
        init_method='gelu',
        train_mode='both'
    )
    
    x = torch.randn(32, 512)
    y = lora_rational(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Parameters: {lora_rational.parameter_count()}")
    
    # Test 2: Model replacement
    print("\n2. Full model replacement:")
    model = nn.Sequential(
        LoRARationalWrapper(in_features=784, out_features=256, lora_rank=4),
        LoRARationalWrapper(in_features=256, out_features=128, lora_rank=4),
        LoRARationalWrapper(in_features=128, out_features=10, lora_rank=4)
    )
    
    # Test 3: Progressive training
    print("\n3. Progressive training simulation:")
    trainer = ProgressiveLoRARationalTrainer(model)
    
    # Simulate each stage
    opt1 = trainer.stage1_rational_only()
    opt2 = trainer.stage2_lora_only()
    opt3 = trainer.stage3_both()
    
    print("\n" + "=" * 60)
    print("LoRA + Rational integration complete!")