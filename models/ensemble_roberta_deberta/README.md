

# Disaster Tweet Classification with RoBERTa and DeBERTa Ensemble

This project performs disaster tweet classification using transformer-based models: **RoBERTa** and **DeBERTa**. It uses Hugging Face's `transformers` and `Trainer` API along with **LoRA** fine-tuning on RoBERTa and model ensembling for better accuracy.

## 🧠 Features

* Fine-tunes `roberta-base` with LoRA (parameter-efficient fine-tuning)
* Fine-tunes `microsoft/deberta-v3-base` directly
* Uses Hugging Face `Trainer` API for training
* Implements logit-level weighted ensembling (RoBERTa + DeBERTa)
* Outputs final predictions to a CSV for Kaggle submission
* Metrics: Accuracy and F1-score

## 📁 File Structure

```
.
├── ensemble.py                  # Main training and inference script
├── train.csv                    # Training dataset
├── test.csv                     # Test dataset
├── sample_submission.csv        # Sample format for submission
├── roberta_final_model/         # Fine-tuned RoBERTa model (saved)
├── deberta_final_model/         # Fine-tuned DeBERTa model (saved)
└── essemble_submission.csv      # Final output file for submission
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers peft scikit-learn pandas
```

### 2. Prepare Data

Ensure the following files are in your working directory:

* `train.csv`: Contains `text` and `target` columns.
* `test.csv`: Contains `id` and `text` columns.
* `sample_submission.csv`: Sample submission format with `id` and `target`.

### 3. Run the Script

```bash
python ensemble.py
```

The script will:

1. Load and preprocess data
2. Fine-tune RoBERTa using LoRA
3. Fine-tune DeBERTa
4. Make predictions on test data
5. Perform logit-level weighted ensembling (default weights: 0.3 RoBERTa, 0.7 DeBERTa)
6. Export predictions to `essemble_submission.csv`

## ⚙️ Configuration

You can modify:

* Model names (RoBERTa, DeBERTa)
* `TrainingArguments` for batch size, epochs, learning rate, etc.
* LoRA parameters (`r`, `alpha`, `dropout`, target modules)
* Ensemble weights (default: 0.3 for RoBERTa, 0.7 for DeBERTa)

## 📊 Evaluation

The model is evaluated using:

* **Accuracy**
* **F1 Score** (used to select the best model)

Metrics are computed during training using a custom `compute_metrics` function.

## 🛠 Dependencies

* Python 3.7+
* PyTorch
* Hugging Face Transformers
* `peft` (for LoRA)
* scikit-learn
* pandas

## 📌 Notes

* RoBERTa uses LoRA for efficient fine-tuning. DeBERTa is fine-tuned directly.
* If GPU memory issues arise, the script includes basic cache-clearing techniques.
* The validation set is not separated in this demo version; for real applications, split the training data accordingly.

## 📤 Output

The final prediction file is saved as:

```
essemble_submission.csv
```

Format:

| id  | target |
| --- | ------ |
| 123 | 1      |
| 456 | 0      |

---

