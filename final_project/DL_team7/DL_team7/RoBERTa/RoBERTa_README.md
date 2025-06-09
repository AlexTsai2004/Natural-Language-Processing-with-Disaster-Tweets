# 災難推文分類模型 (使用 RoBERTa)

這個專案使用微調後的 `roberta-base` 模型，對推文進行二分類，判斷是否為災難相關推文。

---

## 所需檔案
請先將以下三個檔案上傳至 Colab：
- `train.csv`
- `test.csv`
- `sample_submission.csv`

---

## 執行步驟說明

### 1. 匯入套件與設定隨機種子  
載入所有必要的套件，並固定隨機種子以確保實驗可重現性。

### 2. 載入資料集  
使用 pandas 讀取訓練與測試資料。

### 3. 初始化 RoBERTa 的 tokenizer  
使用 `roberta-base` 的預訓練 tokenizer 進行文字前處理。

### 4. 定義自訂 Dataset 類別  
建立 `RobertaTweetDataset`，處理文字轉換為模型輸入格式。

### 5. 切分訓練與驗證資料  
使用 80% 訓練、20% 驗證的比例進行切分。

### 6. 建立 DataLoader  
使用 PyTorch 的 DataLoader 包裝資料集。

### 7. 計算類別權重並指定損失函數  
處理不平衡資料問題，使用 `CrossEntropyLoss` 加入 class weights。

### 8. 載入 RoBERTa 預訓練模型  
使用 `RobertaForSequenceClassification`，設定輸出為兩類。

### 9. 設定 TensorBoard 日誌記錄  
追蹤訓練與驗證過程的 Loss 和 Accuracy。

### 10. 建立優化器與學習率排程器  
使用 AdamW 優化器與線性學習率排程器（含 warm-up）。

### 11. 實作 EarlyStopping  
當驗證損失連續 N 次未改善時提前停止訓練。

### 12. 定義驗證函數  
計算模型在驗證集的 Loss 與 Accuracy。

### 13. 模型訓練主迴圈  
進行多個 epoch 的訓練，每次皆會做驗證與早停檢查。

### 14. 載入最佳模型並繪製 Loss 圖  
載入最佳驗證損失的模型，並畫出訓練與驗證 Loss 曲線。

### 15. 最終模型評估  
印出分類報告與混淆矩陣。

### 16. 測試資料預測  
對 `test.csv` 做預測並產生 `submission.csv` 供 Kaggle 上傳。

### 17. 匯出 requirements.txt（選用）  
產生 Colab 環境的套件需求清單供本地端安裝。

---

## 模型與訓練設定（超參數）

- 預訓練模型：`roberta-base`  
- 最大輸入長度：128  
- 批次大小（batch size）：32  
- 學習率（learning rate）：1.7e-5  
- 優化器：AdamW  
- 權重衰減（weight decay）：0.01  
- 訓練 epoch 數：最多 5（含 Early Stopping）  
- Warm-up steps：300  
- 損失函數：CrossEntropyLoss（含 class weight）  
- EarlyStopping：patience = 2  
- 驗證集比例：20%  
- 評估指標：Accuracy、Confusion Matrix、Classification Report

---
## 執行結果

- `submission.csv`: 預測結果，上傳 Kaggle。  
- 訓練/驗證 Loss 圖：視覺化模型收斂狀況。  
- 混淆矩陣：顯示模型預測的正確與錯誤分佈。  
