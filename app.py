from flask import Flask, request, jsonify, send_from_directory
from uuid import uuid4
import re
import os
import io
import base64
import tempfile

# --- heavy ML imports (may be large) ---
try:
    import torch
    import whisper
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from piper import PiperVoice, SynthesisConfig
except Exception as e:
    # Defer import errors until runtime endpoints are hit
    torch = None
    whisper = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None
    PeftModel = None
    PiperVoice = None
    SynthesisConfig = None
    _import_error = e
else:
    _import_error = None

app = Flask(__name__, static_folder='static')

conversations = {}

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


def extract_price(text):
    matches = re.findall(r'\$?\s?(\d+(?:\,\d{3})*(?:\.\d{1,2})?)', text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", "").replace("$", ""))
    except:
        return None


# ---------------------------
# Load models (Whisper, Qwen + LoRA, Piper)
# ---------------------------
MODEL_DIR = r"C:\Users\redia\LLMs_Project\qwen_intent"
BASE_MODEL = "Qwen/Qwen2.5-3B"
PIPER_MODEL = "./en_US-ryan-medium.onnx"

whisper_model = None
model = None
tokenizer = None
voice = None
tts_config = None

def initialize_models():
    global whisper_model, model, tokenizer, voice, tts_config
    if _import_error:
        raise RuntimeError(f"Missing ML dependencies: {_import_error}")

    if whisper_model is None:
        print("Loading Whisper STT...")
        whisper_model = whisper.load_model("base")

    if model is None or tokenizer is None:
        print("Loading Qwen tokenizer and model (this may take a while)...")
        BNB_CONFIG = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=BNB_CONFIG,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, MODEL_DIR)
        model.eval()

    if voice is None:
        print("Loading Piper TTS...")
        voice = PiperVoice.load(PIPER_MODEL, use_cuda=False)
        tts_config = SynthesisConfig(
            volume=1.0,
            length_scale=1.0,
            noise_scale=0.667,
            noise_w_scale=0.8,
            normalize_audio=True
        )


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


def generate_reply_with_model(scenario, history, buyer_msg):
    # Ensure models are initialized
    initialize_models()
    prompt = build_inference_prompt(scenario, history, buyer_msg)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return parse_output(decoded)


def synthesize_piper_wav_bytes(text):
    # Ensure Piper loaded
    initialize_models()
    audio_chunks = []
    sample_rate = None
    channels = None
    for chunk in voice.synthesize(text, syn_config=tts_config):
        if sample_rate is None:
            sample_rate = chunk.sample_rate
            channels = chunk.sample_channels
        audio_chunks.append(chunk.audio_int16_bytes)

    # concatenate bytes
    pcm = b"".join(audio_chunks)

    # write WAV to bytes
    bio = io.BytesIO()
    import wave
    with wave.open(bio, 'wb') as wf:
        wf.setnchannels(channels or 1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate or 22050)
        wf.writeframes(pcm)

    wav_bytes = bio.getvalue()
    return wav_bytes


# Initialize heavy models on server start so we enforce using them only
try:
    initialize_models()
    print("All models initialized and ready.")
except Exception as e:
    print(f"Failed to initialize models at startup: {e}")
    # Re-raise so the server fails fast — user requested strict usage of these models
    raise


def mock_generate_reply(scenario, history, buyer_msg):
    # Simple rule-based negotiation behavior to mimic the original agent
    seller_target = float(scenario.get("seller_target_price", 0) or 0)
    seller_bottom = float(scenario.get("seller_bottomline", 0) or 0)
    list_price = float(scenario.get("list_price", 0) or 0)

    offer = extract_price(buyer_msg)

    # If no numeric offer, interpret intent from text
    lower = buyer_msg.lower()
    if any(w in lower for w in ["quit", "exit", "bye", "thanks"]):
        return "Thanks — if you'd like to resume, provide another offer.", "end"

    if offer is None:
        # If buyer asks question or says a phrase, give a short probing response
        return "Can you share a price you're comfortable with?", "probe"

    # If buyer offers at or above seller target -> accept
    if offer >= seller_target and seller_target > 0:
        return f"I can accept ${int(offer)}. That works for me.", "accept"

    # If offer is between bottom and target -> counter with midpoint
    if seller_bottom <= offer < seller_target:
        counter = round((offer + seller_target) / 2)
        if counter <= offer:
            counter = int(seller_target)
        return f"I appreciate ${int(offer)}. Could you do ${int(counter)}?", "counter"

    # Offer below bottom -> politely reject and restate bottom
    if offer < seller_bottom:
        return f"I appreciate the offer of ${int(offer)}, but I can't go below ${int(seller_bottom)}.", "reject"

    # Fallback
    return "Hmm — I'm not sure. Can you make a clearer offer?", "unknown"


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/start', methods=['POST'])
def api_start():
    data = request.json or {}
    scenario = data.get('scenario', {})
    conv_id = str(uuid4())
    conversations[conv_id] = {
        'scenario': scenario,
        'history': []
    }
    return jsonify({'conv_id': conv_id, 'scenario_meta': format_scenario_meta_inference(scenario)})


@app.route('/api/message', methods=['POST'])
def api_message():
    # Accept either JSON with `message` or multipart/form-data with `audio` file
    buyer_msg = ''
    conv_id = None
    if request.content_type and request.content_type.startswith('multipart/form-data'):
        conv_id = request.form.get('conv_id')
        audio = request.files.get('audio')
        if audio:
            # save to temp file
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1] or '.wav')
            audio.save(tf.name)
            tf.close()
            try:
                initialize_models()
                # whisper can accept file path
                res = whisper_model.transcribe(tf.name)
                buyer_msg = res.get('text', '').strip()
            except Exception as e:
                return jsonify({'error': f'Transcription error: {e}'}), 500
            finally:
                try:
                    os.unlink(tf.name)
                except:
                    pass
    else:
        data = request.json or {}
        conv_id = data.get('conv_id')
        buyer_msg = data.get('message', '')

    if conv_id not in conversations:
        return jsonify({'error': 'conversation not found'}), 404

    conv = conversations[conv_id]
    scenario = conv['scenario']
    history = conv['history']

    # Append buyer message
    history.append(('Buyer', buyer_msg))

    # Generate reply using the fine-tuned Qwen model (no fallback)
    try:
        seller_text, intent = generate_reply_with_model(scenario, history, buyer_msg)
    except Exception as e:
        return jsonify({'error': f'Model generation error: {e}'}), 500

    # Run a minimal guardrail similar to original app
    seller_bottom = float(scenario.get('seller_bottomline', 0) or 0)
    offer = extract_price(buyer_msg)
    if intent in ['accept', 'agree', 'deal'] and offer is not None and offer < seller_bottom:
        seller_text = f"I appreciate the offer of ${int(offer)}, but I really can't go lower than ${int(seller_bottom)}."
        intent = 'reject'

    history.append(('Seller', seller_text))

    # Synthesize seller speech with Piper and return base64 WAV so client can play
    try:
        wav = synthesize_piper_wav_bytes(seller_text)
        audio_b64 = base64.b64encode(wav).decode('utf-8')
    except Exception as e:
        return jsonify({'error': f'TTS synthesis error: {e}'}), 500

    return jsonify({'seller_text': seller_text, 'intent': intent, 'history': history, 'audio_b64': audio_b64})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
