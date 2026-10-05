# NLP Disaster Tweet Classification Project

## 1. Project Overview

### 1.1 Objective

This project aims to develop a natural language processing (NLP)-based classification system to automatically identify whether a tweet is related to a real disaster event.  
It is a binary classification task, where tweets must be labeled as:

- `1`: Disaster-related  
- `0`: Not disaster-related

### 1.2 Technical Challenges

- **Linguistic Ambiguity**: Distinguishing literal disaster descriptions from metaphorical expressions (e.g., “on fire” can mean excitement rather than fire).
- **Context Understanding**: Identical words may convey different meanings in different contexts.
- **Data Imbalance**: The number of non-disaster tweets is significantly higher than disaster-related ones.

### 1.3 Example Illustrations

| Tweet | Label | Description |
|-------|-------|-------------|
| "I'm on fire tonight! Winning all my poker games!" | 0 | Metaphorical expression, not related to a disaster |
| "Forest fire near La Ronge Sask. Canada. Residents asked to evacuate immediately." | 1 | Describes a real disaster event |

---

## 2. Dataset Description

### 2.1 Dataset Files

| Filename | Description | Records |
|----------|-------------|---------|
| `train.csv` | Training data with tweet content and labels | 7,613 |
| `test.csv` | Testing data without labels | 3,263 |
| `sample_submission.csv` | Sample submission format for Kaggle | - |

### 2.2 Column Description

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Unique tweet identifier |
| `text` | String | Tweet content (main analysis target) |
| `location` | String | Tweet location (can be null) |
| `keyword` | String | Associated keyword (can be null) |
| `target` | Integer | Disaster label (`0` or `1`) — only in training data |

---

## 3. Data Analysis

### 3.1 Class Distribution Analysis

- **Imbalance Observed**:
  - Disaster tweets: ~3,271 (43%)
  - Non-disaster tweets: ~4,342 (57%)

- **Potential Impact**: Class imbalance may cause the model to bias toward the majority class.

- **Solutions**:
  - Use `class_weight='balanced'` in the model
  - Apply oversampling techniques like SMOTE
  - Prefer **F1-score** over accuracy for evaluation

### 3.2 Tweet Length Analysis

- **Length Concentration**: Most tweets fall within **135–140 characters**
- **Class Differences**:
  - Disaster tweets are generally shorter and more concise
  - Non-disaster tweets tend to be longer with more modifiers

- **Solutions**:
  - Normalize sequence lengths via `padding` and `truncation`
  - Consider tweet length as an additional feature
  - Set `max_length ≈ 140` to match tweet characteristics

### 3.3 Word Cloud & Keyword Analysis

- **Common Words**: “one”, “people”, “got”, “New”, “going”, etc.
- **Disaster Keywords**: “fire”, “emergency”, “dead” are strong indicators
- **Noise Removal**: Terms like “amp”, “https” often appear in URLs and should be cleaned

- **Solutions**:
  - Use **TF-IDF** or **BERT tokenizer** for enhanced text representation
  - Create a custom disaster keyword dictionary
  - Clean URLs and special characters during preprocessing

### 3.4 Location Analysis

- **Missing Location Issue**:
  - Around 2,500 tweets have unknown locations (majority of the data)

- **Geographic Patterns**:
  - Tweets with known locations show higher disaster tweet ratios in India, Mumbai, USA

- **Solutions**:
  - Lower the weight of location features in the model
  - Consider indirect geographic indicators if possible

---

## 4. Planned Modeling Approaches

### 4.1 Traditional Sequence Model: **BiLSTM**

- Bidirectional LSTM model captures sequential dependencies and context.
- Serves as a strong baseline for handling tweet-level textual data.

### 4.2 Contextual Embedding Model: **BERT**

- BERT (Bidirectional Encoder Representations from Transformers) excels at contextual understanding.
- Especially effective on informal language patterns commonly seen in tweets.
- Widely adopted in Kaggle solutions.

### 4.3 Large Language Model (LLM): **DeepSeek**

- To further improve comprehension and generalization, DeepSeek LLM is considered.
- Provides human-like understanding with simple implementation and integration.

### 4.4 Model Ensemble: **DeBERTa + RoBERTa**

- Combine DeBERTa and RoBERTa for ensemble learning:
  - **DeBERTa** improves attention and position encoding
  - **RoBERTa** is an optimized BERT variant with better training strategy
- **Goal**: Leverage complementary strengths for more stable and accurate predictions, especially beneficial in competition settings.

---

## 5. Repository Structure

```
.
├── data/                          # Kaggle dataset
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── eda/
│   └── EDA.ipynb                  # Class distribution, text length, word cloud
├── models/
│   ├── bilstm/                    # BiLSTM baseline (notebook)
│   ├── distilbert/                # DistilBERT fine-tuning (notebook)
│   ├── roberta/                   # RoBERTa fine-tuning (notebook)
│   ├── deberta/                   # DeBERTa-v3 + focal loss (notebook)
│   ├── ensemble_roberta_deberta/  # RoBERTa (LoRA) + DeBERTa logit ensemble (script)
│   └── deepseek_lora/             # DeepSeek-LLM-7B 4-bit + LoRA instruction tuning (script)
└── docs/
    └── DL_team7_checkpoint4.pdf   # Project checkpoint report
```

Each folder under `models/` contains the code, a `README.md` and a `requirements.txt`.

- **Notebooks** (`bilstm`, `distilbert`, `roberta`, `deberta`) were written for Google Colab and read data from `/content/`. Upload the files in `data/` to the Colab session before running them.
- **Scripts** (`ensemble_roberta_deberta`, `deepseek_lora`) read `train.csv` / `test.csv` from the current working directory. Copy the files in `data/` there, or run the scripts from inside `data/`.

---

## 6. Evaluation Protocol

To make the models comparable, every model holds out the **same validation set**:

```python
train_test_split(..., test_size=0.2, random_state=42, stratify=target)   # 1,523 tweets
```

- Each model keeps the checkpoint with the best **validation F1** and predicts the test set with it.
- The ensemble tunes its RoBERTa/DeBERTa weight on this validation set, so its reported ensemble F1 is slightly optimistic.
- Known data caveat: `train.csv` contains 69 tweet texts that appear more than once (179 rows), 18 of them with conflicting labels, and 78 test tweets also appear verbatim in `train.csv`. Duplicates can fall on both sides of the split and inflate validation scores a little.

> The outputs saved inside the notebooks come from runs made **before** these evaluation fixes (different splits, and for BiLSTM the submission was produced by the last epoch rather than the best one). Re-run the notebooks to get numbers that follow this protocol.

---

##  Notes

This project is based on the Kaggle challenge: [Real or Not? NLP with Disaster Tweets].  
It focuses on text mining, model experimentation, and multi-model integration for better performance.

---

