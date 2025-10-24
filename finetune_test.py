#!/usr/bin/env python3
"""
Multi-GPU LLM Fine-tuning Script

Usage:
    accelerate launch train_llm.py --model_name gpt2 --train_file data.pkl --output_dir ./output

Requirements:
    pip install transformers datasets accelerate torch
"""

import argparse
import os
import pickle
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an LLM on a pickle file containing list of strings using multiple GPUs")
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')"
    )
    
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to pickle file containing list of strings"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the fine-tuned model"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps"
    )
    
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every N steps"
    )
    
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use mixed precision (FP16) training"
    )
    
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Use bfloat16 precision training (better than FP16 if supported)"
    )
    
    return parser.parse_args()


def load_and_prepare_data(train_file, tokenizer, max_length):
    """Load pickle file containing list of strings and prepare dataset"""
    print(f"Loading data from {train_file}...")
    
    # Load the pickle file
    with open(train_file, 'rb') as f:
        text_list = pickle.load(f)
    
    # Verify it's a list
    if not isinstance(text_list, list):
        raise TypeError(f"Expected pickle file to contain a list, but got {type(text_list)}")
    
    # Verify all elements are strings
    if not all(isinstance(item, str) for item in text_list):
        raise TypeError("All elements in the list must be strings")
    
    print(f"Loaded {len(text_list)} text examples")
    
    # Create dataset from list
    dataset = Dataset.from_dict({"text": text_list})
    
    # Tokenize the dataset
    def tokenize_function(examples):
        # Tokenize and chunk the text
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # Copy input_ids to labels for causal language modeling
        result["labels"] = result["input_ids"].copy()
        return result
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    return tokenized_dataset


def main():
    args = parse_args()
    
    # Verify file exists
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    
    print(f"🚀 Starting multi-GPU training")
    print(f"Model: {args.model_name}")
    print(f"Training file: {args.train_file}")
    print(f"Output directory: {args.output_dir}")
    
    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Set pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.use_fp16 or args.use_bf16 else torch.float32,
        cache_dir='/scratch-shared/lfletcher',
        device_map='auto'
    )
    
    # Prepare dataset
    train_dataset = load_and_prepare_data(
        args.train_file,
        tokenizer,
        args.max_length
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        logging_dir=f"{args.output_dir}/logs",
        fp16=args.use_fp16,
        bf16=args.use_bf16,
        # Multi-GPU settings
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        # Optimization
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Reporting
        report_to=["tensorboard"],
        load_best_model_at_end=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n🏋️ Starting training...")
    print(f"Total batch size: {args.per_device_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    trainer.train()
    
    # Save the final model
    print("\n💾 Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n✅ Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()