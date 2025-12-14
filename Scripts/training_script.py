import json
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

IGNORE_INDEX = -100

# ---------------------------
# Arguments
# ---------------------------
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-3B"
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to JSON list of conversations"})
    train_ratio: float = field(default=0.9, metadata={"help": "Train split fraction"})

@dataclass
class FinetuneArguments(TrainingArguments):
    model_max_length: int = field(default=2048)


# ---------------------------
# Helpers: build examples (source/target)
# ---------------------------
def safe(x):
    return "" if x is None else str(x)

def format_scenario_meta(sample: dict) -> str:
    s = sample.get("scenario", {})
    cat = safe(s.get("category"))
    items = s.get("items", [])
    item_lines = []
    for it in items:
        item_lines.append(f"- Name: {safe(it.get('name'))} | List price: ${safe(it.get('list_price'))}")
        if it.get("description"):
            item_lines.append(f"  Description: {safe(it.get('description'))}")
    buyer_target = safe(s.get("buyer_target_price"))
    seller_target = safe(s.get("seller_target_price"))
    buyer_bottom = safe(s.get("buyer_bottomline"))
    seller_bottom = safe(s.get("seller_bottomline"))
    meta = [
        f"Category: {cat}",
        "Items:",
        *item_lines,
        f"Buyer target price: {buyer_target}",
        f"Seller target price: {seller_target}",
        f"Buyer bottomline: {buyer_bottom}",
        f"Seller bottomline: {seller_bottom}",
    ]
    return "\n".join(meta)

def build_examples_from_dialog(sample: dict):
    examples = []
    turns = sample.get("turns", [])
    history = []
    for i, t in enumerate(turns):
        role = t.get("role", "").lower()
        text = t.get("text", None)
        if isinstance(text, str):
            text = text.replace("\n", " ").strip()
        else:
            text = None

        if role == "buyer" and text:
            history.append(f"Buyer: {text}")
            next_seller = None
            for j in range(i + 1, len(turns)):
                if turns[j].get("role", "").lower() == "seller":
                    next_seller = turns[j]
                    break
            if next_seller and isinstance(next_seller.get("text"), str) and next_seller.get("text").strip() != "":
                seller_text = next_seller["text"].strip()
                seller_intent = next_seller.get("intent", "unknown")
                scenario_block = format_scenario_meta(sample)
                source = (
                    "SCENARIO:\n"
                    f"{scenario_block}\n\n"
                    "NEGOTIATION HISTORY:\n"
                    + "\n".join(history)
                    + "\nSeller:"
                )
                target = f"Seller: {seller_text}\nIntent: {safe(seller_intent)}"
                examples.append({"source": source, "target": target})
                history.append(f"Seller: {seller_text}")
        elif role == "seller" and text:
            history.append(f"Seller: {text}")
    return examples


# ---------------------------
# Dataset
# ---------------------------
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_max_length=2048, train_ratio=0.9):
        logging.warning("Loading JSON dataset...")
        raw = json.load(open(data_path, "r", encoding="utf-8"))
        logging.warning(f"Loaded {len(raw)} top-level conversations.")

        self.examples = []
        for conv in raw:
            self.examples.extend(build_examples_from_dialog(conv))

        logging.warning(f"Built {len(self.examples)} supervised examples.")

        random.shuffle(self.examples)
        split = int(len(self.examples) * train_ratio)
        self.train_examples = self.examples[:split]
        self.eval_examples = self.examples[split:]

        self.tokenizer = tokenizer
        self.max_len = model_max_length

    def get_split(self, split="train"):
        exs = self.train_examples if split == "train" else self.eval_examples
        input_ids = []
        labels = []
        for ex in exs:
            source = ex["source"] + " "
            target = ex["target"] + self.tokenizer.eos_token
            
            full_text = source + target
            tokenized = self.tokenizer(full_text, truncation=True, max_length=self.max_len)
            
            ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
            lbl = ids.clone()
            
            # Mask source
            source_tok = self.tokenizer(source, truncation=True, max_length=self.max_len, add_special_tokens=False)
            src_len = len(source_tok["input_ids"])
            
            if src_len < len(ids):
                lbl[:src_len] = IGNORE_INDEX

            input_ids.append(ids)
            labels.append(lbl)
        return input_ids, labels

    def build_tensors_for(self, split="train"):
        return self.get_split(split)

# ---------------------------
# Data collator
# ---------------------------
class DataCollator:
    def __init__(self, tok): self.tok = tok
    def __call__(self, batch):
        ids = [b["input_ids"] for b in batch]
        labs = [b["labels"] for b in batch]
        ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=self.tok.pad_token_id)
        labs = torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=IGNORE_INDEX)
        return {"input_ids": ids, "labels": labs, "attention_mask": ids.ne(self.tok.pad_token_id)}

# ---------------------------
# Training entrypoint
# ---------------------------
def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, FinetuneArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. BitsAndBytes Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # 3. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True, 
    )

    # 4. Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA Config 
    lora_config = LoraConfig(
        r=64,                
        lora_alpha=128,      
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],                    
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Data
    dataset_obj = SFTDataset(data_args.data_path, tokenizer, model_max_length=train_args.model_max_length, train_ratio=data_args.train_ratio)
    train_input_ids, train_labels = dataset_obj.build_tensors_for("train")
    eval_input_ids, eval_labels = dataset_obj.build_tensors_for("eval")

    train_dataset = [{"input_ids": x, "labels": y} for x, y in zip(train_input_ids, train_labels)]
    eval_dataset = [{"input_ids": x, "labels": y} for x, y in zip(eval_input_ids, eval_labels)]

    collator = DataCollator(tokenizer)

    # --- FIX: Use processing_class instead of tokenizer ---
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,  # <--- RENAMED ARGUMENT
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=collator,
    )
    # ----------------------------------------------------

    trainer.train()
    trainer.save_model(train_args.output_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
    
    
# run this code using the following line on your terminal:
# !python myscript5.py --model_name_or_path Qwen/Qwen2.5-3B --data_path craiglist_converted.json --output_dir new_qwen_intent --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --num_train_epochs 1 --learning_rate 2e-4 --bf16 True --logging_steps 2