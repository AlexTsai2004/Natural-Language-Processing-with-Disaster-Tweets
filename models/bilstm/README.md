
# 📘 BiLSTM 時間序列預測專案

本專案展示了如何使用 **雙向 LSTM（BiLSTM）** 模型來進行單變數時間序列預測，實作上基於 PyTorch 框架，並透過 Jupyter Notebook 撰寫完整流程。以下為各段程式碼的詳細說明。

---

## 📁 專案檔案

- `BiLSTM.ipynb`：主程式 Notebook，內含資料處理、模型建立、訓練與測試全流程。

---

## 📌 程式碼區塊說明

### 🔹 1. 引入套件

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
```
這部分引入所有所需的函式庫，包括數據處理、模型建立與評估函數。

---

### 🔹 2. 讀取與預處理資料

```python
df = pd.read_csv("your_data.csv")
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df.values.reshape(-1, 1))
```
- 使用 `pandas` 讀入資料。
- 使用 `MinMaxScaler` 對資料進行 0~1 正規化。

---

### 🔹 3. 建立時間序列資料集

```python
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq
```
- 定義函數以滑動視窗（time window）方式建立模型輸入序列與對應標籤。

---

### 🔹 4. 資料分割

```python
train_size = int(len(data_normalized) * 0.7)
val_size = int(len(data_normalized) * 0.2)
test_size = len(data_normalized) - train_size - val_size
```
- 將資料依比例劃分為訓練、驗證與測試集。

---

### 🔹 5. 定義 BiLSTM 模型

```python
class BiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        out = self.linear(lstm_out[:, -1])
        return out
```
- 使用雙向 LSTM 結構，輸出接 Linear 層做回歸預測。
- `batch_first=True` 代表輸入維度為 `(batch, seq, features)`。

---

### 🔹 6. 模型訓練

```python
model = BiLSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

```python
for epoch in range(num_epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        y_pred = model(seq)
        loss = loss_function(y_pred, labels)
        loss.backward()
        optimizer.step()
```
- 訓練模型，使用 Adam 優化器與 MSE 作為損失函數。
- 每個 epoch 中遍歷所有訓練資料並進行梯度更新。

---

### 🔹 7. 模型評估與預測

```python
model.eval()
predictions = []
for seq, _ in test_seq:
    with torch.no_grad():
        predictions.append(model(seq).item())
```

- 將模型切換為評估模式，防止 dropout 或 batchnorm 影響。
- 逐筆預測測試資料。

---

### 🔹 8. 性能指標

```python
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = MAPE(y_test, y_pred)
```
- 計算 MAE、RMSE 與 MAPE 等評估指標。

---

### 🔹 9. 視覺化結果

```python
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
```
- 畫出實際值與預測值的對比圖。

---

## ⚙️ 環境需求

```text
Python 3.8+
torch
pandas
numpy
scikit-learn
matplotlib
```

---

## 🚀 使用方式

```bash
pip install -r requirements.txt
jupyter notebook
# 打開 BiLSTM.ipynb 執行每段程式碼
```

---

## 📬 聯絡方式

如有任何問題，請聯絡作者或開啟 Issue。

