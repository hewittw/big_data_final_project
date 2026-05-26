import os, json, math, webbrowser, threading
import torch
from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

BASE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, 'booth_results')
MODEL_DIR   = os.path.join(RESULTS_DIR, 'models', 'trump_gpt2')
CUTOFFS_PATH = os.path.join(RESULTS_DIR, 'results', 'bucket_cutoffs.json')

# ── Load model ──────────────────────────────────────────────────────────────
print("Loading tokenizer and model from booth_results/models/trump_gpt2/ ...")
# Load base GPT-2 tokenizer and re-add special tokens (avoids tokenizers version mismatch)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
with open(os.path.join(MODEL_DIR, 'tokenizer_config.json')) as f:
    _tcfg = json.load(f)
_special = _tcfg.get('extra_special_tokens', [])
tokenizer.add_special_tokens({'additional_special_tokens': _special})
tokenizer.pad_token = tokenizer.eos_token
model     = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device(
    'cuda'  if torch.cuda.is_available() else
    'mps'   if torch.backends.mps.is_available() else
    'cpu'
)
model.to(device)
print(f"Model loaded on {device}")

# ── Load bucket cutoffs ──────────────────────────────────────────────────────
with open(CUTOFFS_PATH) as f:
    cutoffs = json.load(f)

# ── Detect topics from tokenizer special tokens ──────────────────────────────
topics = sorted([
    int(t.replace('[TOPIC_', '').replace(']', ''))
    for t in tokenizer.additional_special_tokens
    if t.startswith('[TOPIC_')
])
print(f"Topics: {topics}")

# ── Bucket helpers ───────────────────────────────────────────────────────────
def bucket_vix(v):
    if v <= cutoffs['vix'][0]: return 'VIX_LOW'
    if v <= cutoffs['vix'][1]: return 'VIX_MED'
    return 'VIX_HIGH'

def bucket_epu(e):
    if e <= cutoffs['epu'][0]: return 'EPU_LOW'
    if e <= cutoffs['epu'][1]: return 'EPU_MED'
    return 'EPU_HIGH'

# ── Generation ───────────────────────────────────────────────────────────────
def generate_for_topic(vix_label, epu_label, topic_id,
                       max_new_tokens=90, temperature=0.85, top_p=0.92):
    prompt     = f"[{vix_label}] [{epu_label}] [TOPIC_{topic_id}]"
    input_ids  = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cutoffs')
def get_cutoffs():
    return jsonify({
        'vix': cutoffs['vix'],
        'epu': cutoffs['epu'],
        'vix_labels': ['VIX_LOW', 'VIX_MED', 'VIX_HIGH'],
        'epu_labels': ['EPU_LOW', 'EPU_MED', 'EPU_HIGH'],
    })

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    try:
        vix_val = float(data['vix'])
        epu_val = float(data['epu'])
    except (KeyError, ValueError, TypeError):
        return jsonify({'error': 'Invalid input. Provide numeric vix and epu values.'})

    vix_label = bucket_vix(vix_val)
    epu_label = bucket_epu(epu_val)

    tweets = []
    for topic_id in topics:
        text = generate_for_topic(vix_label, epu_label, topic_id)
        tweets.append({'topic': topic_id, 'text': text})

    return jsonify({
        'vix_label': vix_label,
        'epu_label': epu_label,
        'vix_value': vix_val,
        'epu_value': epu_val,
        'tweets':    tweets,
    })

# ── Launch ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    def open_browser():
        webbrowser.open('http://localhost:5001')
    threading.Timer(1.5, open_browser).start()
    print("Starting server at http://localhost:5001")
    app.run(debug=False, port=5001)
