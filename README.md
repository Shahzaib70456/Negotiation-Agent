# Negotiation Agent Demo (Frontend)

This is a minimal web frontend and backend to demo a negotiation AI agent. It uses a lightweight, rule-based mock agent for the demo so you don't need heavy ML dependencies.

Run locally:

1. Create a virtual environment (optional)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install requirements

```powershell
pip install -r requirements.txt
```

3. Run the server

```powershell
python app.py
```

4. Open http://localhost:5000 in your browser.

Notes:
- The original `it_works.py` in the workspace used heavyweight models (Whisper, Qwen, Piper). This demo uses a simplified rule-based agent that mimics negotiation behavior and returns an `intent` for each seller message.
- If you want the web UI to call your original model code, we can wire `app.py` to call functions from `it_works.py`, but note that those models require GPUs and additional libraries.
