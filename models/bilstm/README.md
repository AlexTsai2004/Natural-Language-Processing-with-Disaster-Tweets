# 📘 BiLSTM 災難推文分類（Baseline）

本資料夾使用 **雙向 LSTM（BiLSTM）** 判斷推文是否與真實災難有關（二元分類），作為後續 Transformer 模型的 baseline。以 PyTorch 實作，流程寫在 Google Colab Notebook 中。

---

## 📁 專案檔案

- `BiLSTM.ipynb`：主程式 Notebook，內含資料處理、模型建立、訓練、評估與測試集預測全流程。
- `requirements.txt`：所需套件與版本。

---

## 📌 流程說明

### 🔹 1. 文字前處理

- 轉小寫，移除網址與 `@使用者`。
- hashtag 只移除 `#` 符號、保留字本身（例如 `#earthquake` → `earthquake`，這類字是很強的災難訊號）。
- 移除 emoji 與標點，以 NLTK 斷詞、去除停用詞並做 lemmatization。

### 🔹 2. 切分資料與建立詞表

- 以 `test_size=0.2, random_state=42, stratify=target` 切分訓練/驗證集，與本 repo 其他模型使用相同的驗證集。
- 詞表只用**訓練集**建立（最常見的 10,000 個字，另加 `<PAD>`、`<UNK>`），避免驗證集字彙洩漏。
- 每則推文轉成長度 50 的 id 序列（不足補 `<PAD>`，過長截斷）。

### 🔹 3. 模型架構

```
Embedding(128) → BiLSTM(hidden 128) → 串接正反向最後的 hidden state → Dropout(0.5) → Linear → 1 個 logit
```

- 依每則推文的實際長度使用 `pack_padded_sequence`，最後的 hidden state 不會被補上的 `<PAD>` 影響。

### 🔹 4. 訓練

- 損失函數 `BCEWithLogitsLoss`，Adam 優化器（lr = 1e-3），batch size 32，最多 10 個 epoch。
- Early stopping 監看驗證 loss（patience 3）。
- 每個 epoch 計算驗證集 F1 與 Accuracy，保留 **F1 最高那一輪的權重**，訓練結束後還原該權重。
- 固定隨機種子（42），結果可重現。

### 🔹 5. 評估與視覺化

- 輸出最佳 epoch 的 F1、Accuracy 與 classification report。
- 繪製訓練/驗證 loss 曲線與混淆矩陣。

### 🔹 6. 測試集預測

- 以最佳權重預測 `test.csv`，輸出 `submission.csv`（格式：`id,target`）。

---

## ⚙️ 環境需求

見 `requirements.txt`（Python 3.11、PyTorch、pandas、scikit-learn、NLTK、emoji、matplotlib、seaborn）。

---

## 🚀 使用方式

1. 在 Google Colab 開啟 `BiLSTM.ipynb`。
2. 將 repo 中 `data/train.csv`、`data/test.csv` 上傳到 `/content/`。
3. 依序執行，完成後會自動下載 `submission.csv`。

> Notebook 中保存的輸出結果是修正前執行的，重新執行後數字會不同。

---

## 📬 聯絡方式

如有任何問題，請聯絡作者或開啟 Issue。
