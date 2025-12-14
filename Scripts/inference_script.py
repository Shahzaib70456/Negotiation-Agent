import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
MODEL_DIR = "./qwen_intent"  # Make sure this matches your training output_dir
BASE_MODEL = "Qwen/Qwen2.5-3B"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def load_model():
    print("Loading tokenizer...")
    # Added trust_remote_code=True for Qwen
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # Ensure padding side matches training
    tokenizer.padding_side = "right"
    
    # Qwen-specific fix: Use EOS token if PAD is missing (matches your training script)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (4-bit)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True, # Added for Qwen
    )

    print("Loading LoRA adapter...")
    # We load the adapter from the trained folder
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()
    return model, tokenizer

# ------------------------------------------------------------------
# 1. HELPER: Safe String (From Training Script)
# ------------------------------------------------------------------
def safe(x):
    return "" if x is None else str(x)

# ------------------------------------------------------------------
# 2. HELPER: Scenario Formatting (MUST MATCH TRAINING EXACTLY)
# ------------------------------------------------------------------
def format_scenario_meta_inference(scenario: dict) -> str:
    # Adapt simple JSON input to match training structure
    cat = safe(scenario.get("category"))
    
    item_name = scenario.get("item_name")
    item_desc = scenario.get("item_description")
    list_price = scenario.get("list_price", "N/A") 
    
    item_lines = []
    item_lines.append(f"- Name: {safe(item_name)} | List price: ${safe(list_price)}")
    if item_desc:
        item_lines.append(f"  Description: {safe(item_desc)}")

    buyer_target = safe(scenario.get("buyer_target_price"))
    seller_target = safe(scenario.get("seller_target_price"))
    buyer_bottom = safe(scenario.get("buyer_bottomline"))
    seller_bottom = safe(scenario.get("seller_bottomline"))

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

# ------------------------------------------------------------------
# 3. PROMPT BUILDER
# ------------------------------------------------------------------
def build_inference_prompt(scenario, history, new_buyer_msg):
    scenario_block = format_scenario_meta_inference(scenario)

    hist_lines = []
    for role, text in history:
        clean_text = text.replace("\n", " ").strip()
        hist_lines.append(f"{role}: {clean_text}")

    hist_lines.append(f"Buyer: {new_buyer_msg.replace('\n', ' ').strip()}")
    
    # Handle the "Double Seller" artifact from training
    history_block = "\n".join(hist_lines) + "\nSeller:"

    prompt = (
        "SCENARIO:\n"
        f"{scenario_block}\n\n"
        "NEGOTIATION HISTORY:\n"
        f"{history_block}"
    )
    return prompt

def parse_output(decoded_text):
    if "Intent:" not in decoded_text:
        return decoded_text, "unknown"

    try:
        parts = decoded_text.split("Intent:")
        intent = parts[-1].strip()
        
        pre_intent = parts[-2] 
        if "Seller:" in pre_intent:
            message = pre_intent.split("Seller:")[-1].strip()
        else:
            message = pre_intent.strip()
            
        return message, intent
    except:
        return decoded_text, "parsing_error"

def generate_reply(model, tokenizer, scenario, history, buyer_msg):
    prompt = build_inference_prompt(scenario, history, buyer_msg)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Silence the warning about pad_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100, 
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id, 
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # The model will output the full prompt + generation. 
    # We only want the new part.
    # Qwen sometimes repeats the prompt in decode, so we slice it if needed.
    # Simple way: split by the prompt end if present, or just parse the end.
    # Our parse_output is robust enough to handle the full text usually.
    
    seller_msg, intent = parse_output(decoded)
    return seller_msg, intent

# ------------------------------------------------------------------
# 4. LOGIC GUARDRAILS (The "Solution B")
# ------------------------------------------------------------------

def extract_price(text):
    """
    Finds the last number in the text (e.g. "350", "$350", "350.00").
    Returns float or None.
    """
    # Regex for money/numbers
    matches = re.findall(r'\$?\s?(\d+(?:\,\d{3})*(?:\.\d{2})?)', text)
    if not matches:
        return None
    
    try:
        # cleanup commas/dollars and take the last one found
        val = matches[-1].replace(',', '').replace('$', '')
        return float(val)
    except:
        return None

def run_guardrail(buyer_msg, seller_text, intent, scenario):
    """
    If the model accepts a price lower than seller_bottomline, override it.
    """
    seller_bottom = float(scenario.get("seller_bottomline", 0))
    offer = extract_price(buyer_msg)
    
    # If no price detected in buyer msg, nothing to check
    if offer is None:
        return seller_text, intent
        
    # Check if model Accepted
    is_deal = (
        intent in ["accept", "agree", "deal"] or 
        "deal" in seller_text.lower() or 
        "sounds good" in seller_text.lower() or
        "works for me" in seller_text.lower()
    )
    
    # Check Constraint
    if is_deal and offer < seller_bottom:
        print(f"\n[GUARDRAIL TRIGGERED]: Model tried to accept ${offer} (Limit: ${seller_bottom})")
        return f"I appreciate the offer of ${offer}, but I really can't go lower than ${seller_bottom}.", "reject"

    return seller_text, intent

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    model, tokenizer = load_model()

    print("\nPaste scenario JSON OR press Enter for default.")
    raw = input("Scenario JSON: ").strip()

    if raw:
        scenario = json.loads(raw)
    else:
        scenario = {
            "category": "electronics",
            "item_name": "iPhone 12 (128GB)",
            "item_description": "Used, good battery",
            "list_price": 500,
            "seller_target_price": 420,
            "seller_bottomline": 380,
            "buyer_target_price": 350,
            "buyer_bottomline": 400
        }

    history = [] 

    print(f"\nNegotiation started. (Seller Limit hidden: ${scenario.get('seller_bottomline')})")
    print("Type 'exit' to quit.\n")

    while True:
        buyer_msg = input("Buyer: ").strip()
        if buyer_msg.lower() == "exit":
            break

        # 1. Generate Raw Model Response
        seller, intent = generate_reply(model, tokenizer, scenario, history, buyer_msg)

        # 2. Run the Guardrail
        final_seller, final_intent = run_guardrail(buyer_msg, seller, intent, scenario)

        print("\n--- MODEL REPLY ---")
        print(f"Seller: {final_seller}")
        print(f"Intent: {final_intent}")
        print("-------------------\n")

        # Append final (possibly overridden) text to history
        history.append(("Buyer", buyer_msg))
        history.append(("Seller", final_seller))

if __name__ == "__main__":
    main()