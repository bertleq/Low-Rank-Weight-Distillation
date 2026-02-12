import sys
import os
import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from modeling.student import load_student_model

def generate_text(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: python use_student.py <checkpoint_dir> [prompt]")
        sys.exit(1)
        
    checkpoint_dir = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"
    
    print(f"Loading model from {checkpoint_dir}...")
    
    # Load model
    # Note: You might need to adjust device="mps" or "cuda" depending on your hardware
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        
    model = load_student_model(checkpoint_dir, device=device)
    model.eval()
    
    # Load tokenizer from the same directory (Trainer saves it)
    # If not found, fall back to what was likely the teacher (user needs to know)
    # Here we assume tokenizer is saved in output_dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    except:
        print("Tokenizer not found in checkpoint. Please specify teacher name or ensure tokenizer is saved.")
        return

    print(f"Generating text for prompt: '{prompt}'")
    output = generate_text(model, tokenizer, prompt)
    print("-" * 40)
    print(output)
    print("-" * 40)

if __name__ == "__main__":
    main()
