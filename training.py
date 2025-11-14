#!/usr/bin/env python3
"""
Multi-GPU LLM Fine-tuning Script (supports text files and pickle)

Usage with text files:
    accelerate launch train_llm.py --model_name gpt2 --train_files data.txt --output_dir ./output
    
Usage with pickle (one or more categories):
    accelerate launch train_llm.py --model_name gpt2 --pickle_file data.pkl --categories 4chan wikipedia --output_dir ./output

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
    parser = argparse.ArgumentParser(description="Fine-tune an LLM on text files or pickle data using multiple GPUs")
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')"
    )
    
    # Either text files OR pickle file
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--train_files",
        type=str,
        nargs="+",
        help="Path(s) to training text file(s)"
    )
    input_group.add_argument(
        "--pickle_file",
        type=str,
        help="Path to pickle file containing categorized data"
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="Category names to train on from pickle (e.g., '4chan' 'wikipedia'). If not specified, uses all categories."
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
    
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory (slower but uses less VRAM)"
    )
    
    return parser.parse_args()


def load_pickle_data(pickle_file, categories=None):
    """Load data from pickle file and extract specified categories"""
    print(f"Loading pickle file: {pickle_file}")
    
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected pickle to contain a dictionary, got {type(data)}")
    
    print(f"Available categories in pickle: {list(data.keys())}")
    
    # If no categories specified, use all
    if categories is None:
        categories = list(data.keys())
        print(f"No categories specified, using all: {categories}")
    else:
        # Validate requested categories exist
        missing = [cat for cat in categories if cat not in data]
        if missing:
            raise ValueError(f"Categories not found in pickle: {missing}")
        print(f"Using specified categories: {categories}")
    
    # Collect all text from selected categories
    texts = []
    for category in categories:
        category_data = data[category]
        if isinstance(category_data, list):
            texts.extend(category_data)
        else:
            texts.append(str(category_data))
        print(f"  {category}: {len(category_data) if isinstance(category_data, list) else 1} item(s)")
    
    print(f"Total examples loaded: {len(texts)}")
    return texts


def load_text_files(train_files):
    """Load data from text files"""
    from datasets import load_dataset
    
    print(f"Loading data from {len(train_files)} file(s)...")
    dataset = load_dataset(
        "text",
        data_files={"train": train_files},
        split="train"
    )
    print(f"Loaded {len(dataset)} examples")
    return dataset


def prepare_dataset(texts, tokenizer, max_length):
    """Convert texts to tokenized dataset"""
    # Create dataset from texts
    dataset = Dataset.from_dict({"text": texts})
    
    print(f"Preparing dataset with {len(dataset)} examples...")
    
    # Tokenize
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
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
    
    print(f"🚀 Starting multi-GPU training")
    print(f"Model: {args.model_name}")
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
    )
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled (saves VRAM)")
    
    # Load data based on input type
    if args.pickle_file:
        if not os.path.exists(args.pickle_file):
            raise FileNotFoundError(f"Pickle file not found: {args.pickle_file}")
        
        print(f"\n📦 Loading from pickle file")
        texts = load_pickle_data(args.pickle_file, args.categories)
        train_dataset = prepare_dataset(texts, tokenizer, args.max_length)
    
    else:  # text files
        for file_path in args.train_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Training file not found: {file_path}")
        
        print(f"\n📄 Loading from text files")
        print(f"Training files: {args.train_files}")
        dataset = load_text_files(args.train_files)
        
        # Tokenize
        def tokenize_function(examples):
            result = tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        print("Tokenizing dataset...")
        train_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
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
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
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