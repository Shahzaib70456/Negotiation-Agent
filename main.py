import sounddevice as sd
import numpy as np
import torch
import subprocess
import tempfile
import os
import wave
import json

whisper = None
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Globals for heavy models (initialized via initialize_models)
whisper_model = None
model = None
tokenizer = None
voice = None
tts_config = None

# Initialization status tracking
_init_status = {
    'step': 'not_started',
    'message': 'Not initialized',
    'error': None,
    'ready': False
}

def get_init_status():
    return _init_status.copy()

def is_initialized():
    return _init_status.get('ready', False)


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


def initialize_models(use_cuda=False):
    """Load Whisper, the base model + LoRA adapter, and Piper voice.
    Updates `_init_status` as it progresses.
    """
    global whisper_model, model, tokenizer, voice, _init_status
    try:
        _init_status.update({'step': 'whisper', 'message': 'Loading Whisper STT...', 'error': None})
        import whisper as _whisper
        whisper_model = _whisper.load_model("base")

        _init_status.update({'step': 'tokenizer', 'message': 'Loading tokenizer...', 'error': None})
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer_local = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer_local.padding_side = "right"
        if tokenizer_local.pad_token is None:
            tokenizer_local.pad_token = tokenizer_local.eos_token

        _init_status.update({'step': 'base_model', 'message': 'Loading base model (4-bit)...', 'error': None})
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=BNB_CONFIG,
            device_map="auto",
            trust_remote_code=True
        )

        _init_status.update({'step': 'lora', 'message': 'Loading LoRA adapter...', 'error': None})
        from peft import PeftModel
        model_local = PeftModel.from_pretrained(base, MODEL_DIR)
        model_local.eval()

        _init_status.update({'step': 'piper', 'message': 'Loading Piper TTS...', 'error': None})
        from piper import PiperVoice, SynthesisConfig
        voice_local = PiperVoice.load(PIPER_MODEL, use_cuda=use_cuda)
        tts_config_local = SynthesisConfig(
            volume=1.0,
            length_scale=1.0,
            noise_scale=0.667,
            noise_w_scale=0.8,
            normalize_audio=True
        )

        # Assign to globals only after successful load
        model = model_local
        tokenizer = tokenizer_local
        voice = voice_local
        tts_config = tts_config_local
        # whisper_model already assigned above

        _init_status.update({'step': 'done', 'message': 'All models loaded', 'error': None, 'ready': True})
        return True
    except Exception as e:
        _init_status.update({'step': 'error', 'message': 'Initialization failed', 'error': str(e), 'ready': False})
        raise


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

def generate_reply(model=None, tokenizer=None, scenario=None, history=None, buyer_msg=None):
    """Generate a reply using provided model/tokenizer or the initialized globals.
    Keeps the original argument order but accepts None for model/tokenizer.
    """
    if model is None or tokenizer is None:
        # Use globals
        model_local = globals().get('model')
        tokenizer_local = globals().get('tokenizer')
    else:
        model_local = model
        tokenizer_local = tokenizer

    if model_local is None or tokenizer_local is None:
        raise RuntimeError('Model/tokenizer not initialized. Call initialize_models() first.')

    prompt = build_inference_prompt(scenario, history, buyer_msg)
    inputs = tokenizer_local(prompt, return_tensors="pt").to(model_local.device)
    tokenizer_local.pad_token_id = tokenizer_local.eos_token_id

    with torch.no_grad():
        out = model_local.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer_local.eos_token_id
        )

    decoded = tokenizer_local.decode(out[0], skip_special_tokens=True)
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


PIPER_MODEL = "./en_US-ryan-medium.onnx"

def speak(text):
    global voice, tts_config
    if voice is None or tts_config is None:
        raise RuntimeError('TTS voice not initialized. Call initialize_models() first.')

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
def default_scenario():
    return {
        "category": "electronics",
        "item_name": "iPhone 12 (128GB)",
        "item_description": "Used, good battery",
        "list_price": 500,
        "seller_target_price": 420,
        "seller_bottomline": 380,
        "buyer_target_price": 350,
        "buyer_bottomline": 400
    }


if __name__ == "__main__":
    scenario = default_scenario()
    history = []

    # Initialize models when running standalone
    try:
        print('Initializing models...')
        initialize_models(use_cuda=False)
    except Exception as e:
        print(f'Initialization failed: {e}')
        raise

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
