import copy
import os
import json
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from modeling.low_rank_layer import LowRankLinear

def replace_linear_with_low_rank(module, rank, target_modules=None):
    """
    Recursively replaces nn.Linear layers with LowRankLinear layers.
    
    Args:
        module: The module to modify.
        rank: The rank for the low-rank decomposition. Can be an int (global rank) 
              or a dict {layer_name_substring: rank}.
        target_modules: A list of string names of modules to replace (e.g., ['q_proj', 'v_proj']).
                        If None, replaces all nn.Linear layers.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Check if we should replace this specific module
            if target_modules is None or any(t in name for t in target_modules):
                # Determine rank for this layer
                layer_rank = None
                if isinstance(rank, int):
                    layer_rank = rank
                elif isinstance(rank, dict):
                    # Look for matching key in rank config
                    for key, val in rank.items():
                        if key in name:
                            layer_rank = val
                            break
                    # If no specific match, check for 'default' key
                    if layer_rank is None and 'default' in rank:
                        layer_rank = rank['default']
                
                if layer_rank is not None:
                    # We found a target linear layer and a valid rank
                    new_layer = LowRankLinear.from_linear(child, layer_rank)
                    setattr(module, name, new_layer)
        else:
            # Recursively check children
            replace_linear_with_low_rank(child, rank, target_modules)

def create_student_model(teacher_model, rank, target_modules=None):
    """
    Creates a student model by copying the teacher and replacing linear layers.
    
    Args:
        teacher_model: The pre-trained teacher model.
        rank: The target rank for LowRankLinear layers. Int or Dict.
        target_modules: List of module names to replace.
        
    Returns:
        student_model: The modified model with low-rank layers.
    """
    # Create a deep copy of the teacher model structure
    # Note: This copies weights too, which is inefficient if we are going to overwrite them immediately
    # with SVD, but ensures we keep embeddings, layernorms, etc. correct.
    student_model = copy.deepcopy(teacher_model)
    
    # Replace layers
    replace_linear_with_low_rank(student_model, rank, target_modules)
    
    return student_model

def load_student_model(checkpoint_dir, device="cpu", dtype=torch.float32):
    """
    Loads a distilled student model from a checkpoint directory.
    
    Args:
        checkpoint_dir: Path to the directory containing model weights and distillation_config.json.
        device: Device to load the model onto.
        dtype: Data type for the model.
        
    Returns:
        student_model: The loaded student model.
    """
    # 1. Load distillation config
    config_path = os.path.join(checkpoint_dir, "distillation_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find distillation_config.json in {checkpoint_dir}. "
                              "Make sure the model was trained with the updated run_distillation.py.")
    
    with open(config_path, 'r') as f:
        dist_config = json.load(f)
    
    rank = dist_config["rank"]
    target_modules = dist_config["target_modules"]
    
    # 2. Re-create student structure
    # Load base config
    try:
        config = AutoConfig.from_pretrained(checkpoint_dir)
    except OSError:
        # Fallback: try loading from teacher model if config.json not found (rare if saved with Trainer)
        # But Trainer saves config.json
        raise
        
    # Initialize empty teacher model structure
    # Use meta device to avoid allocating memory for full teacher if possible, 
    # but create_student_model deepcopies, so we need a real model or handle meta.
    # We will use CPU for safety and then move.
    with torch.device("cpu"):
        base_model = AutoModelForCausalLM.from_config(config)
        
    # Create student structure
    # This replaces linear layers with LowRankLinear
    student_model = create_student_model(base_model, rank, target_modules)
    
    # 3. Load weights
    from transformers.modeling_utils import load_state_dict
    
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    
    state_dict = None
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Colud not find model checkpoints in {checkpoint_dir}")
        
    # Load state dict strictly
    keys = student_model.load_state_dict(state_dict, strict=True)
    
    # Move to device and dtype
    student_model.to(device=device, dtype=dtype)
    
    return student_model
