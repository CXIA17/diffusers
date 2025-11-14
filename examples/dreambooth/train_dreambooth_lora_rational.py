# In train_dreambooth_custom_lora.py, etc.
from modules.custom import (
    LoRALayer,
    RationalWrapper,
    LoRARationalWrapper,
    replace_layers_with_lora_rational
)

def inject_lora_rational_into_unet(unet, config):
    """Inject combined LoRA+Rational into UNet"""
    
    # Target different layers for different effects
    attention_targets = ["to_k", "to_q", "to_v", "to_out.0"]
    feedforward_targets = ["ff.net.0", "ff.net.2"]
    
    replaced_layers = {}
    
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            # Determine configuration based on layer type
            if any(target in name for target in attention_targets):
                # Attention layers: smaller rational, larger LoRA
                wrapper = LoRARationalWrapper(
                    pretrained_layer=module,
                    lora_rank=config.get('attn_lora_rank', 8),
                    lora_alpha=config.get('attn_lora_alpha', 16),
                    numerator_degree=3,
                    denominator_degree=2,
                    num_groups=1,
                    init_method='identity',  # Minimal change for attention
                    train_mode='both'
                )
            elif any(target in name for target in feedforward_targets):
                # Feedforward layers: more complex rational
                wrapper = LoRARationalWrapper(
                    pretrained_layer=module,
                    lora_rank=config.get('ff_lora_rank', 4),
                    lora_alpha=config.get('ff_lora_alpha', 8),
                    numerator_degree=5,
                    denominator_degree=4,
                    num_groups=4,
                    init_method='gelu',
                    train_mode='both'
                )
            else:
                continue
            
            # Replace the module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent_module = unet
            for part in parent_name.split('.'):
                if part:
                    parent_module = getattr(parent_module, part)
            setattr(parent_module, child_name, wrapper)
            replaced_layers[name] = wrapper
    
    print(f"Replaced {len(replaced_layers)} layers with LoRA+Rational")
    return unet, replaced_layers