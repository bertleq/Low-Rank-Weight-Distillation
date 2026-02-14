import sys
import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from modeling.student import create_student_model
from distillation.svd_init import initialize_student_from_teacher
from distillation.trainer import DistillationTrainer

import json

def parse_args():
    parser = argparse.ArgumentParser(description="Low-Rank Knowledge Distillation")
    parser.add_argument("--teacher_model", type=str, required=True, help="Path or name of teacher model")
    parser.add_argument("--mode", type=str, default="train", help="Mode (train or init)")
    parser.add_argument("--rank", type=int, default=32, help="Rank for low-rank approximation (global default)")
    parser.add_argument("--rank_config", type=str, default='{\"q_proj\": 256, \"v_proj\": 256, \"k_proj\": 256, \"gate_proj\": 128, \"up_proj\": 256, \"down_proj\": 256, \"lm_head\": 256}', help="JSON string for variable rank config (e.g. '{\"q_proj\": 16, \"v_proj\": 32}')")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.8, help="Distillation loss weight")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--target_modules", nargs='+', default=['o_proj', 'gate_proj', 'up_proj', 'down_proj'], help="Modules to replace with LowRankLinear") 
    return parser.parse_args()

def generate_text(model, tokenizer, prompt, max_new_tokens=512):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def main():
    args = parse_args()
    
    print(f"Loading teacher model: {args.teacher_model}")
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    rank_arg = args.rank
    if args.rank_config:
        try:
            rank_arg = json.loads(args.rank_config)
            print(f"Using variable rank config: {rank_arg}")
        except json.JSONDecodeError as e:
            print(f"Error parsing rank_config: {e}")
            sys.exit(1)
    
    print(f"Creating student model with rank {rank_arg}...")
    student_model = create_student_model(teacher_model, rank_arg, target_modules=args.target_modules)
    
    # Cast student to float32 to ensure stable training with AMP (avoids "Attempting to unscale FP16 gradients" error)
    # The teacher is FP16, so copied layers (embeddings, norms) are FP16. New layers are FP32.
    # We want consistent FP32 master weights for the optimizer.
    student_model = student_model.to(torch.float32)
    
    # Move student to the same device as teacher (likely 'mps' or 'cuda')
    student_model = student_model.to(teacher_model.device)
    
    print("Initializing student with SVD from teacher...")
    initialize_student_from_teacher(student_model, teacher_model)

    # Freeze all parameters except LowRankLinear layers
    # This focuses training on the factorized weights only
    frozen_count = 0
    trainable_count = 0
    for name, param in student_model.named_parameters():
        if "project_in" in name or "project_out" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    
    print(f"Frozen parameters: {frozen_count:,}")
    print(f"Trainable parameters (LowRankLinear only): {trainable_count:,}")

    # Calculate parameter count
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Reduction: {100 * (1 - student_params/teacher_params):.2f}%")
    
    print("Who are you?")
    print(generate_text(student_model, tokenizer, "Who are you?"))

    # Prepare Dataset
    print(f"Loading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=500,
        fp16=True, # Use mixed precision
        remove_unused_columns=False, # Important for custom loss where we need input for teacher
    )
    
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None,
        data_collator=data_collator,
        processing_class=tokenizer,
        teacher_model=teacher_model,
        alpha=args.alpha,
        temperature=args.temperature,
    )

    print("Saving non-trained student model...")
    trainer.save_model(args.output_dir)
    
    # Save distillation config for loading
    distillation_config = {
        "rank": rank_arg,
        "target_modules": args.target_modules
    }
    with open(os.path.join(args.output_dir, "distillation_config.json"), "w") as f:
        json.dump(distillation_config, f, indent=2)
    print(f"Saved distillation config to {args.output_dir}/distillation_config.json")

    if args.mode == "init":
        return

    print("Starting training...")
    trainer.train()
    
    print("Saving student model...")
    trainer.save_model(args.output_dir)
    
    # Save distillation config for loading
    distillation_config = {
        "rank": rank_arg,
        "target_modules": args.target_modules
    }
    with open(os.path.join(args.output_dir, "distillation_config.json"), "w") as f:
        json.dump(distillation_config, f, indent=2)
    print(f"Saved distillation config to {args.output_dir}/distillation_config.json")

if __name__ == "__main__":
    main()
