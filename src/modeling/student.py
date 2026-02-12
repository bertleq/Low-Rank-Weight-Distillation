import copy
import torch.nn as nn
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
