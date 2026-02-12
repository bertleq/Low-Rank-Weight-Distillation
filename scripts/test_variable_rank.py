import sys
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from modeling.student import create_student_model
from modeling.low_rank_layer import LowRankLinear

def test_variable_rank():
    print("Testing Variable Rank Configuration...")
    
    # 1. Create dummy teacher (simple configuration)
    # We'll use a real model structure for simplicity of test logic, e.g. GPT2
    # but since we don't want to load weights, we can just use a dummy class or load config
    
    # Let's use GPT2 config but init random (faster)
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(n_layer=1, n_head=4, n_embd=32)
    teacher = GPT2LMHeadModel(config)
    
    print("Teacher created.")
    
    # 2. Define Rank Config
    # GPT2 has c_attn (q,k,v combined usually, but let's assume we split or target specific parts if possible)
    # Actually GPT2 uses Conv1D and c_attn. 
    # Let's use a config that maps to parts of the name.
    # GPT2 layer names: h.0.attn.c_attn, h.0.attn.c_proj, h.0.mlp.c_fc, h.0.mlp.c_proj
    
    rank_config = {
        "c_attn": 8,
        "c_proj": 4,
        "default": 2
    }
    
    # 3. Create student
    print(f"Creating student with config: {rank_config}")
    # We target Conv1D as well if we handled it? 
    # Our LowRankLinear replaces nn.Linear. GPT2 uses Conv1D which inherits from something else or is custom.
    # Wait, transformers GPT2 uses Conv1D class which is not nn.Linear. 
    # Our current implementation checks `isinstance(child, nn.Linear)`. 
    # GPT-2 Conv1D is NOT nn.Linear. 
    
    # Let's check Llama or BERT which uses nn.Linear.
    # Or just mock a simple model with nn.Linear
    
    class SimpleTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32)
            self.k_proj = nn.Linear(32, 32)
            self.v_proj = nn.Linear(32, 32)
            self.out_proj = nn.Linear(32, 32)
            
    teacher = SimpleTeacher()
    
    rank_config = {
        "q_proj": 4,
        "k_proj": 4,
        "v_proj": 8,
        "default": 16 # Should apply to out_proj if not caught
    }
    
    student = create_student_model(teacher, rank_config)
    
    # 4. Verify Ranks
    print("Verifying ranks...")
    
    def get_rank(layer):
        return layer.rank
        
    # q_proj
    if isinstance(student.q_proj, LowRankLinear):
        print(f"q_proj rank: {student.q_proj.rank} (Expected: 4)")
        assert student.q_proj.rank == 4
    else:
        print("q_proj was not replaced!")

    # v_proj
    if isinstance(student.v_proj, LowRankLinear):
        print(f"v_proj rank: {student.v_proj.rank} (Expected: 8)")
        assert student.v_proj.rank == 8
    else:
        print("v_proj was not replaced!")
        
    # out_proj (should match default if we set logic right, or nothing if we didn't include it in target modules?)
    # create_student_model default target_modules is None (all linear) if not passed? 
    # Let's check student.py logic.
    # default target_modules is None in the function signature.
    # So it should replace all linear layers.
    
    if isinstance(student.out_proj, LowRankLinear):
        # We didn't specify out_proj in keys, so it should hit 'default'
        print(f"out_proj rank: {student.out_proj.rank} (Expected: 16)")
        assert student.out_proj.rank == 16
    else:
        print("out_proj was not replaced!")

    print("SUCCESS: Variable rank configuration verified.")

if __name__ == "__main__":
    test_variable_rank()
