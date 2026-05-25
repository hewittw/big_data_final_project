# Tweeting Fear
### Analyzing Bidirectional Dynamics Between Trump's Posts and U.S. Market Volatility to Create Synthetic Tweets
**BUSN 20800 Big Data · University of Chicago Booth · 2026**  
Hewitt Watkins · Erik Lopez · Mateo Fretes · Vedant Dangayach

---

## Project Overview

This project investigates the relationship between Trump's social media posts and U.S. market volatility (VIX) and economic policy uncertainty (EPU). The hallmark deliverable is a fine-tuned GPT-2 model that generates synthetic Trump-style tweets conditioned on real-time market conditions.

---

## Repository Structure

```
├── trump_vix_proposal (1).ipynb   # Main analysis: supervised learning (OLS, Ridge, Lasso, RF, XGBoost)
├── unsupervised_learning.ipynb    # Unsupervised v1: LDA + K-means (4 topics, 3-bucket VIX/EPU)
├── unsupervised_v2.ipynb          # Unsupervised v2: clean text, 6 topics, 5-bucket VIX/EPU
├── gpt2_finetuning_colab.ipynb    # GPT-2 training v1 (run in Google Colab)
├── gpt2_finetuning_v2_colab.ipynb # GPT-2 training v2 (run in Google Colab)
├── model_demo.ipynb               # Test both models locally — all VIX×EPU combinations
├── app.py                         # Flask web app v1 (4 topics, 3-bucket)
├── app_v2.py                      # Flask web app v2 (6 topics, 5-bucket)
├── app_space.py                   # HuggingFace Spaces deployment version
├── templates/index.html           # Frontend v1
├── templates_v2/index_v2.html     # Frontend v2
├── Dockerfile                     # For HuggingFace Spaces Docker deployment
├── requirements.txt               # Python dependencies
└── booth_results/                 # v1 trained model (download from Drive, not in git)
└── booth_results_v2/              # v2 trained model (download from Drive, not in git)
```

---

## Running the Web Apps Locally

### Prerequisites

```bash
pip install flask transformers torch
```

You need the trained model folders downloaded from Google Drive and placed in this directory:
- `booth_results/` — v1 model (4 topics, 3-bucket VIX/EPU)
- `booth_results_v2/` — v2 model (6 topics, 5-bucket VIX/EPU)

### Web App v1

```bash
python app.py
```

Opens automatically at `http://localhost:5001`

- Enter a **VIX value** (e.g. 22.5) and **EPU value** (e.g. 165.0)
- Click **Generate Tweets**
- Returns 4 AI-generated Trump-style tweets, one per topic:
  - Topic 0, 1, 2, 3
- VIX/EPU bucketed into 3 levels: `LOW`, `MED`, `HIGH`

### Web App v2

```bash
python app_v2.py
```

Opens automatically at `http://localhost:5002`

- Same interface with finer conditioning and named topics
- Returns 6 AI-generated tweets, one per named topic:
  - **Fake News & Witch Hunt**
  - **Crooked Democrats**
  - **America First**
  - **MAGA Endorsements**
  - **Make America Great Again**
  - **Rallies & Events**
- VIX/EPU bucketed into 5 levels: `VERY_LOW`, `LOW`, `MED`, `HIGH`, `VERY_HIGH`

### Running Both Simultaneously

Both apps run on different ports so you can compare outputs side-by-side:

```bash
# Terminal 1
python app.py       # localhost:5001

# Terminal 2
python app_v2.py    # localhost:5002
```

### Typical VIX/EPU Reference Values

| Market Condition | VIX | EPU |
|---|---|---|
| Calm markets | 12 | 100 |
| Normal | 18 | 150 |
| Elevated anxiety | 25 | 200 |
| Crisis (e.g. COVID) | 65 | 350 |

---

## Testing the Models — Demo Notebook

`model_demo.ipynb` lets you test both models locally without running the web apps.

### V1 section
- Loads `booth_results/models/trump_gpt2/`
- Generates all 36 combinations (3 VIX × 3 EPU × 4 topics)

### V2 section
- Loads `booth_results_v2/models/trump_gpt2_v2/`
- Generates all 150 combinations (5 VIX × 5 EPU × 6 topics) with named topics

### Side-by-side comparison
Set `VIX_VALUE` and `EPU_VALUE` in the last cell to compare v1 vs v2 output for a specific market condition.

> The V1 and V2 sections are independent — skip the V1 cells if you only have `booth_results_v2/` downloaded.

---

## Reproducing the Full Pipeline

### 1. Data & Supervised Learning
Run `trump_vix_proposal (1).ipynb` locally. Requires `trump_tweets_dataset.csv` (not in repo — large file).

### 2. Unsupervised Learning (v1)
Run `unsupervised_learning.ipynb` locally. Outputs `tweet_labels.csv`.

### 2b. Unsupervised Learning (v2 — recommended)
Run `unsupervised_v2.ipynb` locally. Outputs `tweet_labels_v2.csv` and `bucket_cutoffs_v2.json`.

### 3. GPT-2 Fine-Tuning
Upload outputs to Google Drive folder `booth_final_data/`, then run the Colab notebook on a T4 GPU:
- **v1:** `gpt2_finetuning_colab.ipynb` → saves to `booth_results/` in Drive
- **v2:** `gpt2_finetuning_v2_colab.ipynb` → saves to `booth_results_v2/` in Drive

### 4. Download & Run
Download `booth_results/` or `booth_results_v2/` from Drive into this directory, then run `app.py` or `app_v2.py`.

---

## Key Results

| Model | Target | R² |
|---|---|---|
| OLS (VADER features) | Same-week VIX | 0.332 |
| ΔVIX same-week | — | 0.027 (n.s.) |
| GPT-2 v1 perplexity | — | 11.41 |

Tweet count: tweet_count, mean_retweets, mean_tweet_len significant predictors of contemporaneous VIX level (p<0.05). No forward-looking predictive power found.
