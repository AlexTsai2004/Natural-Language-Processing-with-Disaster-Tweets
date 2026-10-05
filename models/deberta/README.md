
# 災難推文分類模型 (使用 DeBERTa 與 Focal Loss)

這個專案使用微調後的 `microsoft/deberta-v3-base` 預訓練模型，結合 Focal Loss，對推文進行二分類，判斷是否為災難相關推文。

---

## 所需檔案
請先將以下三個檔案上傳至 Colab：
- `train.csv`
- `test.csv`
- `sample_submission.csv`

---

## 執行步驟說明

### 1. 匯入套件與設定隨機種子  
載入 pandas、PyTorch、transformers、sklearn 等必要套件，並固定隨機種子以確保實驗可重現性。

### 2. 下載 NLTK 資源與定義停用詞  
下載 stopwords，並排除否定詞以保留關鍵語意。

### 3. 定義文字清理函式  
包含去除網址、HTML 標籤、emoji、標點符號與停用詞，保留否定詞。

### 4. 定義自訂 Dataset  
建立 `TweetDataset`，將文字轉換成模型輸入的 `input_ids` 與 `attention_mask`。

### 5. 讀取並清理資料  
讀取 `train.csv` 與 `test.csv`，對文本欄位套用清理函式。

### 6. 切分訓練與驗證資料  
以 80% 訓練、20% 驗證比例分層切分（stratified）資料，與其他模型使用相同切分。

### 7. 載入 Tokenizer 與模型  
使用 `microsoft/deberta-v3-base` 的 tokenizer 與 `AutoModelForSequenceClassification`，設定二分類。

### 8. 建立 DataLoader  
分別為訓練與驗證資料建立 PyTorch DataLoader。

### 9. 定義優化器與學習率排程器  
使用 AdamW 優化器，搭配 cosine 學習率排程。

### 10. 實作 Focal Loss  
自訂 Focal Loss 損失函數以處理類別不平衡。

### 11. 實作訓練迴圈與 EarlyStopping  
每個 epoch 訓練並驗證，依驗證 F1 分數儲存最佳模型；F1 連續 patience 次沒有改善則停止訓練。

### 12. 紀錄 TensorBoard  
訓練過程中紀錄 Loss 與 F1 分數。

### 13. 繪製 Loss 曲線圖  
訓練結束後繪製並儲存訓練與驗證的 Loss 曲線。

### 14. 載入最佳模型並評估  
載入驗證 F1 最佳模型，輸出分類報告與 F1 分數。

### 15. 繪製混淆矩陣  
視覺化驗證資料的混淆矩陣。

### 16. 預測測試資料  
對測試集進行推論，產生 `submission.csv`。

---

## 模型與訓練設定（超參數）

- 預訓練模型：`microsoft/deberta-v3-base`  
- 最大輸入長度：128（Dataset 內固定）  
- 批次大小（batch size）：訓練 32，驗證 64  
- 學習率（learning rate）：2e-5  
- 優化器：AdamW  
- 學習率排程器：cosine  
- 損失函數：Focal Loss (alpha=0.25, gamma=2.0)  
- 訓練 epoch 數：最多 10（含 Early Stopping）  
- EarlyStopping：patience = 2  
- 驗證集比例：20%  
- 評估指標：F1 Score、Classification Report、Confusion Matrix

---

## 執行結果

- `submission.csv`：測試集預測結果，可直接上傳至 Kaggle。  
- `loss_curve.png`：訓練與驗證 Loss 曲線圖，視覺化模型收斂狀況。  
- `confusion_matrix.png`：驗證集混淆矩陣圖，展示分類效果。  
- TensorBoard 日誌：紀錄 Loss 與 F1，可使用 TensorBoard 監控訓練狀況。

---
