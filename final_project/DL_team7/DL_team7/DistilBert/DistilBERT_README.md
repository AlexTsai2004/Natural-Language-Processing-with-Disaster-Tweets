
#DistilBERT 推文分類器

本專案使用 Hugging Face 的 `DistilBERT` 模型進行推文分類任務。資料來自於訓練資料 (`train.csv`) 和測試資料 (`test.csv`)，目的是判斷一則推文是否與災難相關。

---

##安裝必要套件

```bash
!pip install transformers datasets scikit-learn tensorboard --quiet
```

---

##模型架構與功能

- 使用 `DistilBERT` 作為語言模型
- 整合欄位（keyword, location, text）進行預處理
- 支援 Early Stopping、Learning Rate Scheduler
- 訓練過程使用 TensorBoard 進行即時監控
- 評估指標包含：Accuracy、F1 Score、Confusion Matrix
- 匯出驗證預測與提交檔案（submission）

---

## 專案結構

```
.
├── train.csv
├── test.csv
├── best_model.pt
├── val_predictions.csv
├── submission.csv
├── requirements.txt
└── runs/
```

---

##資料預處理

- `keyword` 和 `location` 欄位的空值會被填入空字串 `''`
- 三欄合併為 `input` 欄，作為 BERT 模型輸入

---

##訓練模型

### Step 1: 初始化模型與資料集

```python
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
```

### Step 2: 建立自訂 Dataset

使用 `TweetDataset` 將文字資料轉換為 BERT 格式的輸入格式。

### Step 3: 訓練並評估模型

```python
train_model(model, train_loader, val_loader)
```

支援 EarlyStopping，避免過度擬合，並自動儲存最佳模型為 `best_model.pt`。

---

##模型評估

- 訓練與驗證 Loss 曲線圖
- Confusion Matrix 熱圖（使用 Seaborn）
- 輸出驗證集預測結果：`val_predictions.csv`

---

##測試集預測與提交

將最終測試預測結果儲存為 `submission.csv`。

---

##使用 TensorBoard 查看訓練過程

```python
%load_ext tensorboard
%tensorboard --logdir ./runs
```

---

##匯出依賴套件

```bash
!pip list > requirements.txt
```

可下載 `requirements.txt` 並在其他環境安裝相同套件：

```python
from google.colab import files
files.download('requirements.txt')
```

---

##輸出結果說明

- `val_predictions.csv`：驗證集預測與真實標籤對照
- `submission.csv`：測試集預測結果，用於比賽提交

---

##使用環境建議

- Python 3.8+
- 建議使用 GPU 加速（Colab、Kaggle、雲端 GPU 等）

---
