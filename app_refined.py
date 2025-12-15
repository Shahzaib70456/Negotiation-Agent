from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from uuid import uuid4
import re
import os
import io
import base64
import wave
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from piper import PiperVoice, SynthesisConfig
import google.generativeai as genai

GENI_MODEL = "gemini-2.5-flash" 
genai_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=genai_api_key)

app = Flask(__name__, static_folder='static')
CORS(app)

conversations = {}

# ---------------------------
# Config
# ---------------------------
MODEL_DIR = # finetuned model directory
BASE_MODEL = "Qwen/Qwen2.5-3B"
PIPER_MODEL = "./en_US-ryan-medium.onnx"

model = None
tokenizer = None
voice = None
tts_config = None

# ---------------------------
# Helpers
# ---------------------------
def safe(x): return "" if x is None else str(x)

def format_scenario_meta_inference(scenario: dict) -> str:
    """
    Formats the scenario for the Seller Model.
    IMPORTANT: We DO NOT include Buyer Target/Bottomline here.
    """
    cat = safe(scenario.get("category"))
    item_name = scenario.get("item_name")
    item_desc = scenario.get("item_description")
    list_price = scenario.get("list_price", "N/A")
    
    item_lines = [f"- Name: {safe(item_name)} | List price: ${safe(list_price)}"]
    if item_desc: item_lines.append(f"  Description: {safe(item_desc)}")
    
    # Only Seller info
    seller_target = safe(scenario.get("seller_target_price"))
    seller_bottom = safe(scenario.get("seller_bottomline"))
    
    meta = [
        f"Category: {cat}",
        "Items:",
        *item_lines,
        f"Seller target price: {seller_target}",
        f"Seller bottomline: {seller_bottom}",
    ]
    return "\n".join(meta)

def extract_price(text):
    """
    Finds the last valid price. Ignores small numbers unless $ is present.
    """
    matches = re.findall(r'(\$)?\s?(\d+(?:\,\d{3})*(?:\.\d{2})?)', text)
    if not matches: return None
    
    valid_prices = []
    for has_dollar, number_str in matches:
        try:
            val = float(number_str.replace(',', ''))
            # Filter out ages/versions (e.g. 2, 12) unless they have '$'
            if has_dollar or val > 25: 
                valid_prices.append(val)
        except: continue
            
    return valid_prices[-1] if valid_prices else None

# ---------------------------
# Load Models
# ---------------------------
def initialize_models():
    global model, tokenizer, voice, tts_config

    if model is None or tokenizer is None:
        print("Loading Qwen model...")
        BNB_CONFIG = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=BNB_CONFIG,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base, MODEL_DIR)
        model.eval()
        print("✓ Qwen ready")

    if voice is None:
        print("Loading Piper TTS...")
        try:
            voice = PiperVoice.load(PIPER_MODEL, use_cuda=False)
            tts_config = SynthesisConfig(
                volume=1.0,
                length_scale=1.0,
                noise_scale=0.667,
                noise_w_scale=0.8,
                normalize_audio=True
            )
            print("✓ Piper ready")
        except Exception as e:
            print(f"X Piper Failed: {e}")

# ---------------------------
# LLM Logic
# ---------------------------
def build_inference_prompt(scenario, history, buyer_msg):
    scenario_block = format_scenario_meta_inference(scenario)
    
    hist_lines = []
    for role, text in history:
        clean_text = text.replace("\n", " ").strip()
        hist_lines.append(f"{role}: {clean_text}")
    
    # FIXED: Clean text OUTSIDE the f-string to avoid backslash error
    clean_buyer_msg = buyer_msg.replace("\n", " ").strip()
    hist_lines.append(f"Buyer: {clean_buyer_msg}")
    
    history_block = "\n".join(hist_lines) + "\nSeller:"
    
    # Inject Strategy
    STRATEGY_PROMPT = """
INSTRUCTIONS:
You are a skilled negotiator. Your goal is to maximize the selling price.
1. Do NOT lower your price immediately.
2. If the buyer offers a low price, persist on your target price.
3. You must reject low offers at least 1-2 times before making a significant drop.
"""
    return f"{STRATEGY_PROMPT}\n\nSCENARIO:\n{scenario_block}\n\nNEGOTIATION HISTORY:\n{history_block}"

def parse_output(decoded_text):
    if "Intent:" not in decoded_text: return decoded_text, "unknown"
    try:
        parts = decoded_text.split("Intent:")
        intent = parts[-1].strip()
        pre = parts[-2]
        if "Seller:" in pre: msg = pre.split("Seller:")[-1].strip()
        else: msg = pre.strip()
        return msg, intent
    except:
        return decoded_text, "parsing_error"

def generate_reply_with_model(scenario, history, buyer_msg):
    prompt = build_inference_prompt(scenario, history, buyer_msg)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return parse_output(decoded)

# ---------------------------
# TTS Logic
# ---------------------------
def synthesize_piper_wav_bytes(text):
    if not voice: return None
    audio_chunks = []
    sample_rate = None
    channels = None
    for chunk in voice.synthesize(text, syn_config=tts_config):
        if sample_rate is None:
            sample_rate = chunk.sample_rate
            channels = chunk.sample_channels
        audio_chunks.append(chunk.audio_int16_bytes)
    pcm = b"".join(audio_chunks)
    bio = io.BytesIO()
    with wave.open(bio,'wb') as wf:
        wf.setnchannels(channels or 1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate or 22050)
        wf.writeframes(pcm)
    return bio.getvalue()

# ---------------------------
# Grading Logic
# ---------------------------
# ---------------------------
# Grading Logic (RESTORED VERSION)
# ---------------------------
def grade_negotiation_with_gemini(conversation_history, scenario):
    """
    conversation_history = [('Buyer', 'text'), ('Seller', 'text'), ...]
    Returns a Python dict parsed from the model's JSON output.
    """
    import json, re

    transcript = "\n".join([f"{r}: {t}" for r, t in conversation_history])
    scenario_text = format_scenario_meta_inference(scenario)

    prompt = (
        "You are a negotiation performance grader.\n\n"
        "You must:\n"
        "1. Read the negotiation transcript.\n"
        "2. Evaluate the user's negotiation performance using the rubric.\n"
        "3. Think step-by-step internally to reach your answer, but DO NOT reveal the reasoning.\n"
        "4. Output ONLY the final evaluation in valid JSON.\n"
        "5. ALWAYS address the user directly as “you”. Never say “the user” or “they”.\n"
        "6. The response MUST start with '{' and end with '}' with no extra text.\n\n"
        "Rubric:\n"
        "- Emotional control (0–20)\n"
        "- Strategic clarity (0–20)\n"
        "- Quality of offers/anchors (0–20)\n"
        "- Persuasiveness (0–20)\n"
        "- Outcome quality (0–20)\n\n"
        "Return the following JSON fields exactly (no extra fields):\n"
        "{\n"
        '  "score": number,\n'
        '  "summary": string,\n'
        '  "improvement": string\n'
        "}\n\n"
        "IMPORTANT: Output ONLY valid JSON. No markdown, no code fences, no explanation, nothing else.\n\n"
        "Transcript:\n"
        f"{transcript}\n\n"
        "Scenario:\n"
        f"{scenario_text}\n"
    )

    # Call Gemini
    model = genai.GenerativeModel(GENI_MODEL)
    response = model.generate_content(prompt)

    raw = ""
    try:
        raw = response.text if hasattr(response, "text") else str(response)
    except:
        raw = str(response)

    # Direct JSON parse
    try:
        return json.loads(raw.strip())
    except:
        # fallback — extract first { ... }
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception as e:
                return {
                    "score": 0,
                    "summary": f"Gemini JSON parse failed: {e} | Raw: {raw[:500]}",
                    "improvement": "N/A"
                }
        else:
            return {
                "score": 0,
                "summary": f"Gemini returned no JSON object. Raw: {raw[:500]}",
                "improvement": "N/A"
            }


# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    return send_from_directory('static','index.html')

@app.route('/api/start', methods=['POST'])
def api_start():
    data = request.json or {}
    scenario = data.get('scenario',{})
    conv_id = str(uuid4())
    conversations[conv_id] = {'scenario': scenario,'history':[]}
    return jsonify({'conv_id':conv_id,'scenario_meta':format_scenario_meta_inference(scenario)})

@app.route('/api/message', methods=['POST'])
def api_message():
    data = request.json or {}
    conv_id = data.get('conv_id')
    buyer_msg = data.get('message','')

    if conv_id not in conversations:
        return jsonify({'error':'conversation not found'}),404

    conv = conversations[conv_id]
    scenario = conv['scenario']
    history = conv['history']

    history.append(('Buyer',buyer_msg))

    # 1. LLM Reply
    try:
        seller_text, intent = generate_reply_with_model(scenario, history, buyer_msg)
    except Exception as e:
        return jsonify({'error':f'Model generation failed: {e}'}),500

    # 2. ROBUST GUARDRAIL (Logic B)
    seller_bottom = float(scenario.get('seller_bottomline',0) or 0)
    seller_target = float(scenario.get('seller_target_price', seller_bottom * 1.1))

    buyer_offer = extract_price(buyer_msg)
    seller_offer = extract_price(seller_text)

    # 2a. Strategic Accept (Target met)
    # Check for complaints ("Is $500 too much?") vs Offers
    is_complaint = any(x in buyer_msg.lower() for x in ["too much", "expensive", "high"])
    
    if buyer_offer and buyer_offer >= seller_target and not is_complaint:
        if intent not in ['accept', 'deal']:
             seller_text = f"That is a great price. I accept ${buyer_offer}."
             intent = "accept"

    # 2b. Prevent Lowball Accept
    is_deal = intent in ['accept', 'agree', 'deal'] or "deal" in seller_text.lower()
    if buyer_offer and is_deal and buyer_offer < seller_bottom:
        seller_text = f"I appreciate the offer of ${int(buyer_offer)}, but I really can't go lower than ${int(seller_bottom)}."
        intent = 'reject'

    # 2c. Prevent Self-Sabotage (Seller offering too low)
    if seller_offer and seller_offer < seller_bottom and intent != 'reject':
        seller_text = f"Actually, looking at my costs, the lowest I can go is ${int(seller_bottom)}."
        intent = 'counter-price'

    history.append(('Seller', seller_text))

    # 3. TTS
    audio_b64 = None
    if voice:
        try:
            wav_bytes = synthesize_piper_wav_bytes(seller_text)
            audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        except Exception as e:
            print(f"TTS Error: {e}")

    return jsonify({'seller_text':seller_text,'intent':intent,'history':history,'audio_b64':audio_b64})

@app.route("/api/grade", methods=["POST"])
def api_grade():
    data = request.json or {}
    conv_id = data.get("conv_id")

    if conv_id not in conversations:
        return jsonify({"error": "conversation not found"}), 404

    conv = conversations[conv_id]
    scenario = conv["scenario"]
    history = conv["history"]

    try:
        result = grade_negotiation_with_gemini(history, scenario)
        return jsonify({
            "score": result.get("score", 0),
            "summary": result.get("summary", "No summary produced."),
            "improvement": result.get("improvement", "No improvement suggestions.")
        })
    except Exception as e:
        return jsonify({"error": f"grading failed: {e}"}), 500


if __name__ == "__main__":
    initialize_models()
    app.run(host='0.0.0.0',port=5000,debug=True)
