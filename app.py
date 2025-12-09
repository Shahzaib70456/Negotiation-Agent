from flask import Flask, request, jsonify, send_from_directory
from uuid import uuid4
import re
import os

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

    seller_text, intent = mock_generate_reply(scenario, history, buyer_msg)

    # Run a minimal guardrail similar to original app
    seller_bottom = float(scenario.get('seller_bottomline', 0) or 0)
    offer = extract_price(buyer_msg)
    if intent in ['accept', 'agree', 'deal'] and offer is not None and offer < seller_bottom:
        seller_text = f"I appreciate the offer of ${int(offer)}, but I really can't go lower than ${int(seller_bottom)}."
        intent = 'reject'

    history.append(('Seller', seller_text))

    return jsonify({'seller_text': seller_text, 'intent': intent, 'history': history})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
