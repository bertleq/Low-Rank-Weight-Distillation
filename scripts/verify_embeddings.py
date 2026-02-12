import sys
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from modeling.student import create_student_model

def verify_embeddings():
    print("Verifying Embedding Initialization...")
    
    # Use a small real model for verification to ensure structure matches
    model_name = "gpt2" # Small and fast
    print(f"Loading teacher: {model_name}")
    try:
        teacher = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load {model_name}: {e}")
        return

    # Create student
    print("Creating student...")
    student = create_student_model(teacher, rank=16, target_modules=['c_attn'])

    # Check embeddings
    # GPT-2 embeddings are usually in transformer.wte
    
    print("\nChecking Embeddings:")
    
    # 1. Input Embeddings
    teacher_emb = teacher.get_input_embeddings()
    student_emb = student.get_input_embeddings()
    
    if teacher_emb is not None and student_emb is not None:
        diff = (teacher_emb.weight - student_emb.weight).abs().sum()
        print(f"Input Embedding Difference: {diff.item()}")
        if diff == 0:
            print("SUCCESS: Input embeddings are identical.")
        else:
            print("FAILURE: Input embeddings differ!")
    else:
        print("Could not find input embeddings.")

    # 2. Output Embeddings (Head)
    teacher_head = teacher.get_output_embeddings()
    student_head = student.get_output_embeddings()
    
    if teacher_head is not None and student_head is not None:
        diff = (teacher_head.weight - student_head.weight).abs().sum()
        print(f"Output Embedding Difference: {diff.item()}")
        if diff == 0:
            print("SUCCESS: Output embeddings are identical.")
        else:
            print("FAILURE: Output embeddings differ!")
    else:
        print("Could not find output embeddings.")

if __name__ == "__main__":
    verify_embeddings()
