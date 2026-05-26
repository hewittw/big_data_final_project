import os, json, math
import torch
from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from huggingface_hub import hf_hub_download

app = Flask(__name__)

HF_MODEL_ID = "hewittw/trump-gpt2"

# ── Load tokenizer from base GPT-2 + special tokens from HF Hub ─────────────
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tcfg_path = hf_hub_download(repo_id=HF_MODEL_ID, filename="tokenizer_config.json")
with open(tcfg_path) as f:
    tcfg = json.load(f)
special_tokens = tcfg.get('extra_special_tokens', [])
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
tokenizer.pad_token = tokenizer.eos_token

# ── Load model from HF Hub ───────────────────────────────────────────────────
print("Loading model...")
model = GPT2LMHeadModel.from_pretrained(HF_MODEL_ID)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Model loaded on {device}")

# ── Load bucket cutoffs from HF Hub ─────────────────────────────────────────
cutoffs_path = hf_hub_download(repo_id=HF_MODEL_ID, filename="bucket_cutoffs.json")
with open(cutoffs_path) as f:
    cutoffs = json.load(f)

# ── Detect topics from special tokens ───────────────────────────────────────
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
    prompt    = f"[{vix_label}] [{epu_label}] [TOPIC_{topic_id}]"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
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

# ── Launch (HF Spaces uses port 7860) ───────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
