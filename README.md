# LLM-Based Negotiation Roleplay Coach

An end-to-end, locally deployable negotiation training system that simulates realistic buyer–seller bargaining using a fine-tuned large language model, symbolic logic guardrails, and real-time speech interaction. This project was developed as part of **DS-UA 301** and demonstrates how neuro-symbolic systems can overcome numerical reasoning limitations in LLMs while maintaining natural dialogue.

---

## Project Overview

The Negotiation Roleplay Coach allows users to practice negotiations in an immersive, conversational environment. Users can **speak or type offers**, receive **spoken responses from a seller agent**, and get **post-negotiation coaching feedback** that evaluates strategy, emotional control, and outcome quality.

Key goals of the project:

* Train an LLM to behave like a strategic seller
* Enforce strict numerical constraints (e.g., bottom-line prices)
* Provide real-time speech interaction without cloud dependencies
* Deliver actionable, second-person negotiation feedback

---

## Running the Web App

From the root of the repository, run (after setting the finetuned model directory in app_refined.py):

```bash
python app_refined.py
```
---

## Repository Structure

```text
.
├── app_refined.py              # Flask backend + orchestration + scoring agent logic
├── Dataset/                    # Unified negotiation datasets (Craigslist + CaSiNo)
├── Scripts/                    # Training and inference scripts
├── tests/                      # Comparative inference and evaluation scripts
├── en_US-ryan-medium.onnx.json # Piper model for Text-To-Speech
├── static/                     # Frontend assets (CSS/JS)
├── requirements.txt            # Packages to install
└── README.md

```


---

## Core Features

* **Fine-Tuned Negotiation LLM**
  Qwen 2.5-3B fine-tuned with QLoRA for strategic bargaining, **explicit intent prediction**, and persona consistency.
  
  **Fine-tuned Model:** https://huggingface.co/rediah321/Qwen2.5-3B-Negotiation/tree/main

* **Neuro-Symbolic Guardrails**
  Python-based logic enforces pricing constraints and overrides invalid model decisions ("math blindness" prevention).

* **Real-Time Speech Interface**

  * Browser-based Speech-to-Text (Web Speech API)
  * Offline Text-to-Speech using Piper

* **Hybrid Strategy Engine**
  Dynamic strategy injection based on price gaps (lowball, negotiation, closing phases).

* **Automated Negotiation Grader**
  Gemini-powered evaluation agent that scores and critiques the user’s negotiation performance.

---

## Datasets

### Primary Dataset: Craigslist Bargain Dataset

The **Craigslist Bargain** dataset served as the **main training dataset** for this project. It provides structured buyer–seller negotiation dialogues focused on price bargaining and was the original backbone for our negotiation schema and training pipeline.

Key advantages:

* Explicit buyer–seller roles
* Clear price offers and counteroffers
* Well-suited for supervised fine-tuning on negotiation structure

### Supplementary Dataset: CaSiNo (Campsite Negotiation Corpus)

To enrich linguistic diversity and strategic depth, we additionally incorporated the **CaSiNo (Campsite Negotiation) Corpus**.

**Why CaSiNo?**

* More natural, persuasive, multi-turn dialogue
* Annotated negotiation strategies
* Participant metadata and satisfaction ratings

### Schema-First Dataset Unification

Before training, we **designed a unified negotiation schema** and converted *both* datasets into this format:

* Craigslist Bargain → mapped directly into schema
* CaSiNo → converted from task-specific priorities into numeric values

The unified schema standardizes:

* Roles (buyer / seller)
* Turn structure (agent, role, action)
* Numerical values and Intent
* Final deal outcomes

### Intent Annotation

Each model turn explicitly includes an **Intent label** (e.g., `init-price`, `counter-price`, `accept`, `reject`, `inquiry`).

* Existing CaSiNo annotations were reused where available
* Missing intents were classified using the **Gemini 2.5 Flash API** with rate-limit-safe retries

This intent supervision was critical for teaching the model **strategic control and consistency** during negotiation.

---

## Speech Pipeline (STT → LLM → TTS)

1. User speaks → Web Speech API transcribes locally
2. Transcript sent to backend
3. Fine-tuned Qwen model generates seller response
4. Logic layer validates offers and injects strategy
5. Piper synthesizes speech → Base64 WAV
6. Browser plays audio response automatically

This design enables **low-latency, fully offline negotiation simulation**.

---

##  Model & Training Details

* **Base Model:** Qwen 2.5-3B
* **Fine-Tuning Method:** QLoRA (4-bit NF4)
* **LoRA Rank:** 64
* **Learning Rate:** 2 × 10⁻⁴
* **Epochs:** 1
* **Batch Size:** 4 (Gradient Accumulation: 4)
* **Precision:** bf16

The configuration balances memory efficiency with strict output formatting and strategic behavior.

---

## System Architecture

### Backend (Flask)

Responsible for orchestration and state management:

* Model and TTS initialization
* Session-based negotiation tracking
* Logic guardrails and strategy engine

**API Endpoints:**

* `/api/start` – Initialize a negotiation scenario
* `/api/message` – Process user input and return seller response + audio
* `/api/grade` – Generate post-negotiation evaluation

### Frontend (HTML/CSS/JS)

* Scenario selection and customization
* Voice or text input
* Chat-style conversation UI
* Automatic audio playback
* Post-negotiation grading display

---

## Results

* **100% adherence** to seller-only output format with explicit **Intent tagging**
* Strong persona consistency after fine-tuning
* Improved strategic anchoring vs. base model
* Stable, coherent grading feedback across scenarios
* Significantly enhanced realism with voice interaction

Comparative tests showed the fine-tuned model consistently outperformed the raw base model in both strategy and constraint adherence.

---

## Limitations

* Occasional persona drift or minor hallucinations
* Context window limitations in very long negotiations
* Regex-based price extraction can fail on ambiguous phrasing ("fifty bucks")

---

## Future Work

* Reinforcement Learning (RLHF) to maximize sale outcomes
* Robust structured parsing for offer extraction
* Emotion detection from speech for adaptive strategy
* Longer-term memory and state tracking

---

## Team Contributions

* **Muhammad Haider Asif**
  Real-time speech pipeline (STT + Piper TTS), documentation, repository management

* **Muhammad Shahzaib Hassan**
  Dataset conversion, frontend development, logic guardrails

* **Muhammad Talal Naveed**
  Model training, Qwen migration, experimentation, evaluation

---

## Final Note

This project demonstrates that **small, efficient LLMs combined with symbolic reasoning** can outperform larger models in constrained, real-world tasks like negotiation. By enforcing logic externally and letting the model focus on language and strategy, we achieve both correctness and realism.







