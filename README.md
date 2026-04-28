# Stock Prediction from Financial News
**Group 6 Market prediction from financial news - group 1 — CS 1671, University of Pittsburgh**  
Irisin Yu · Astalaxmi Dhanaseelan · Ciara Dwyer · Yifei Tian · Sanjana Pejathaya

Predicting S&P 500 intraday direction (close ≥ open) from same-day financial news headlines.  
We compare FinBERT-based supervised models against Gemma3-4B zero-shot reasoning.

---

## Quick Results

| System | Accuracy |
|--------|----------|
| Majority baseline (always up) | 53.5% |
| FinBERT → LinearSVC (baseline) | 64.1% |
| FinBERT + Price → LinearSVC | 64.2% |
| FinBERT + Price → XGBoost | 62.4% |
| FinBERT + Price → FT-Transformer | 62.2% |
| Multi-Headline FinBERT → FT-Transformer | **65.0%** |
| **Gemma3-4B Zero-Shot** | **73.8%** |

---

## Key Files

### Models (run on Google Colab)

| File | Description |
|------|-------------|
| `StockPredict.ipynb` | Baseline: FinBERT sentiment → LinearSVC (64.1%) |
| `combination.ipynb` | FinBERT + 4 price features → LinearSVC / XGBoost |
| `combination_transformer.ipynb` | FinBERT + 4 price features → Feature Tokenizer Transformer |
| `combination_multisent.ipynb` | **Best supervised model**: per-headline FinBERT aggregation (11 features) → FT-Transformer (65.0%) |
| `combination_multisent_LinearSVC.ipynb` | Same 11 features → LinearSVC (ablation) |
| `combination_tryGemma.ipynb` | **Best overall**: Gemma3-4B zero-shot via Pitt API (73.8%) |
| `LLM_Prompt_Construction_LLama.py` | LLaMA 3.1 zero-shot and few-shot via Pitt API |

### Data

| File | Description |
|------|-------------|
| `sp500_headlines_with_prices.csv` | 3,507 trading days with concatenated headlines + OHLCV prices |
| `sp500_headlines_with_prices sep ~.csv` | Same data, headlines separated by ` ~ ` for per-headline processing |
| `sp500_headlines_2008_2024.csv` | Raw individual headlines (19,127 rows) |

### Report

```
Association_for_Computational_Linguistics__ACL__conference/latex/
├── acl_latex.tex    ← main report source
├── custom.bib       ← references
└── acl_latex.pdf    ← compiled PDF
```

---

## How to Run

### Supervised models (Colab)
1. Open any `combination_*.ipynb` in Google Colab
2. Upload the data file when prompted (`sp500_headlines_with_prices sep ~.csv` or `sp500_headlines_with_prices.csv`)
3. Run all cells — FinBERT inference is cached automatically to avoid re-running

### Gemma3 zero-shot (Colab)
1. Open `combination_tryGemma.ipynb` in Google Colab
2. Upload `config.py` (contains API key — **never committed to GitHub**)
3. Upload `sp500_headlines_with_prices sep ~.csv`
4. Run all cells — results cached to `gemma_cache.json`

### Compile the report (local)
```bash
cd Association_for_Computational_Linguistics__ACL__conference/latex
pdflatex acl_latex.tex
bibtex acl_latex
pdflatex acl_latex.tex
pdflatex acl_latex.tex
```

---

## Notes
- `config.py` is in `.gitignore` and is never pushed to GitHub
- All train/test splits are **chronological** (no shuffling) to prevent data leakage
- Models marked with "Colab" require GPU; T4 is sufficient for all experiments
