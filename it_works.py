import sounddevice as sd
import numpy as np
import torch
import subprocess
import tempfile
import os
import wave
import json

import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------
# Load Whisper STT
# ---------------------------
print("Loading Whisper STT...")
whisper_model = whisper.load_model("base")


# ---------------------------------------------------------
#  USE YOUR FINETUNED QWEN-INFERENCE CODE (MINIMAL CHANGES)
# ---------------------------------------------------------
MODEL_DIR = r"C:\Users\redia\LLMs_Project\qwen_intent"
BASE_MODEL = "Qwen/Qwen2.5-3B"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def load_finetuned_qwen():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (4-bit)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()
    return model, tokenizer

model, tokenizer = load_finetuned_qwen()


# ---------------------------------------------------------
# YOUR PARSER / PROMPT FUNCTIONS
# ---------------------------------------------------------
import re

def safe(x):
    return "" if x is None else str(x)

def format_scenario_meta_inference(scenario: dict) -> str:
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

def build_inference_prompt(scenario, history, new_buyer_msg):
    scenario_block = format_scenario_meta_inference(scenario)

    hist_lines = []
    for role, text in history:
        clean = text.replace("\n", " ").strip()
        hist_lines.append(f"{role}: {clean}")

    hist_lines.append(f"Buyer: {new_buyer_msg.strip()}")
    history_block = "\n".join(hist_lines) + "\nSeller:"

    return (
        "SCENARIO:\n"
        f"{scenario_block}\n\n"
        "NEGOTIATION HISTORY:\n"
        f"{history_block}"
    )

def parse_output(decoded_text):
    if "Intent:" not in decoded_text:
        return decoded_text, "unknown"

    try:
        parts = decoded_text.split("Intent:")
        intent = parts[-1].strip()
        pre = parts[-2]

        if "Seller:" in pre:
            msg = pre.split("Seller:")[-1].strip()
        else:
            msg = pre.strip()

        return msg, intent
    except:
        return decoded_text, "parsing_error"

def generate_reply(model, tokenizer, scenario, history, buyer_msg):
    prompt = build_inference_prompt(scenario, history, buyer_msg)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return parse_output(decoded)

def extract_price(text):
    matches = re.findall(r'\$?\s?(\d+(?:\,\d{3})*(?:\.\d{2})?)', text)
    if not matches:
        return None
    return float(matches[-1].replace(",", "").replace("$", ""))

def run_guardrail(buyer_msg, seller_text, intent, scenario):
    seller_bottom = float(scenario.get("seller_bottomline", 0))
    offer = extract_price(buyer_msg)
    if offer is None:
        return seller_text, intent

    is_deal = (
        intent in ["accept", "agree", "deal"] or
        "deal" in seller_text.lower() or
        "sounds good" in seller_text.lower() or
        "works for me" in seller_text.lower()
    )

    if is_deal and offer < seller_bottom:
        print(f"\n[GUARDRAIL TRIGGERED]: Tried to accept ${offer} (Limit: ${seller_bottom})")
        return f"I appreciate the offer of ${offer}, but I really can't go lower than ${seller_bottom}.", "reject"

    return seller_text, intent


# ---------------------------
# Piper TTS (unchanged)
# ---------------------------
from piper import PiperVoice, SynthesisConfig

PIPER_MODEL = "./en_US-ryan-medium.onnx"

print("Loading Piper...")
voice = PiperVoice.load(PIPER_MODEL, use_cuda=False)

tts_config = SynthesisConfig(
    volume=1.0,
    length_scale=1.0,
    noise_scale=0.667,
    noise_w_scale=0.8,
    normalize_audio=True
)

def speak(text):
    print(f"[AI SAYS] {text}")

    # Streaming synthesis generator
    audio_stream = voice.synthesize(text, syn_config=tts_config)

    stream = None

    for chunk in audio_stream:

        # Lazy-init audio output on first chunk
        if stream is None:
            stream = sd.OutputStream(
                samplerate=chunk.sample_rate,
                channels=chunk.sample_channels,
                dtype="int16"
            )
            stream.start()
        pcm = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
        # Write raw PCM chunk
        stream.write(pcm)

    # Cleanup
    if stream is not None:
        stream.stop()
        stream.close()


# ---------------------------
# Recording
# ---------------------------
def record_audio(duration=5, fs=16000):
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def speech_to_text(audio):
    result = whisper_model.transcribe(audio)
    return result["text"]


# ---------------------------------------------------------
# RUN NEGOTIATION WITH VOICE
# ---------------------------------------------------------
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

print("Ready! You are the BUYER. Speak your offer.\nSay 'quit' to exit.\n")

while True:
    audio = record_audio(duration=5)
    buyer_msg = speech_to_text(audio).strip()
    print(f"[YOU SAID] {buyer_msg}")

    if buyer_msg.lower() in ["quit", "exit"]:
        print("Goodbye.")
        break

    seller_raw, intent = generate_reply(model, tokenizer, scenario, history, buyer_msg)
    seller_final, intent_final = run_guardrail(buyer_msg, seller_raw, intent, scenario)

    print(f"[SELLER] {seller_final}   (intent: {intent_final})")

    speak(seller_final)

    history.append(("Buyer", buyer_msg))
    history.append(("Seller", seller_final))
