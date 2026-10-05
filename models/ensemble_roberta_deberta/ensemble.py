import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType # 導入 TaskType
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score # Kaggle 競賽通常看 F1 分數
import os


# 設置 CUDA 記憶體配置，在某些情況下有助於避免 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 嘗試清理 CUDA 緩存，有助於釋放佔用的 GPU 記憶體
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# ================================ 數據準備部分 (新增/修改) ================================
import pandas as pd

train_file_path = "train.csv"
test_file_path = "test.csv"
sample_submission_file_path = "sample_submission.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
sample_submission_df = pd.read_csv(sample_submission_file_path)

X_train_text = train_df['text'].values
y_train_labels = train_df['target'].values
X_test_text = test_df['text'].values
test_ids = test_df['id'].values # 保存測試集 ID 以便生成提交文件

# ================================ 自定義 Dataset (方便 Hugging Face Trainer 使用) ================================
class DisasterTweetsDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

# ================================ 模型訓練和預測流程 (結合數據) ================================
learning_rate = 1.7e-5
# 1. 設置模型和 Tokenizer
roberta_model_name = "roberta-base"
deberta_model_name = "microsoft/deberta-v3-base" # DeBERTa-base 是一個常用的選擇

roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)

# 2. 數據分詞 (Tokenization)
# 對訓練數據進行分詞
roberta_train_encodings = roberta_tokenizer(list(X_train_text), truncation=True, padding=True, max_length=141)
deberta_train_encodings = deberta_tokenizer(list(X_train_text), truncation=True, padding=True, max_length=141)

# 對測試數據進行分詞
roberta_test_encodings = roberta_tokenizer(list(X_test_text), truncation=True, padding=True, max_length=141)
deberta_test_encodings = deberta_tokenizer(list(X_test_text), truncation=True, padding=True, max_length=141)

# 3. 創建 Dataset 對象
roberta_train_dataset = DisasterTweetsDataset(roberta_train_encodings, y_train_labels)
deberta_train_dataset = DisasterTweetsDataset(deberta_train_encodings, y_train_labels)

# 測試集只有 encodings，沒有 labels
roberta_test_dataset = DisasterTweetsDataset(roberta_test_encodings)
deberta_test_dataset = DisasterTweetsDataset(deberta_test_encodings)


# 4. 設置訓練參數 (Hugging Face Trainer)
num_labels = 2 # 0: non-disaster, 1: disaster

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    learning_rate=learning_rate,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=130,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,
    eval_strategy="epoch",     # Evaluate every epoch
    save_strategy="epoch",           # Save checkpoint every epoch
    load_best_model_at_end=True,     # Load the best model at the end of training
    metric_for_best_model="f1",      # Use F1-score to determine the best model
    greater_is_better=True,
    report_to="none"                 # 禁用 wandb 等報告，如果需要啟用可配置
)

# 評估指標
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# ================================ 訓練 RoBERTa 模型 ================================
print("\n=== 開始訓練 RoBERTa 模型 ===")
# 1. 加載預訓練模型
roberta_base_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name, num_labels=num_labels)

# 2. **修正：設置 LoRA 配置，並將 task_type 設置為 "SEQ_CLS"**
peft_config_roberta = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "key", "value"], # RoBERTa 的 attention 層通常是 q, k, v，而不是 q_proj, k_proj, v_proj
    lora_dropout=0.2,
    bias="none",
    task_type=TaskType.SEQ_CLS, # **修正：對於文本分類是 SEQ_CLS**
)
# 3. **修正：將 LoRA 適配器應用到模型**
roberta_model_peft = get_peft_model(roberta_base_model, peft_config_roberta)
roberta_model_peft.print_trainable_parameters() # 打印可訓練參數數量

roberta_trainer = Trainer(
    model=roberta_model_peft, # **修正：傳入應用 LoRA 的模型**
    args=training_args,
    train_dataset=roberta_train_dataset,
    eval_dataset=roberta_train_dataset, # 建議使用單獨的驗證集
    compute_metrics=compute_metrics,
)
roberta_trainer.train()
roberta_trainer.save_model("roberta_final_model") # 保存訓練好的模型

# ================================ 訓練 DeBERTa 模型 ================================
print("\n=== 開始訓練 DeBERTa 模型 ===")
deberta_model = AutoModelForSequenceClassification.from_pretrained(deberta_model_name, num_labels=num_labels)

deberta_trainer = Trainer(
    model=deberta_model,
    args=training_args,
    train_dataset=deberta_train_dataset,
    eval_dataset=deberta_train_dataset, # 同上，簡化用訓練集演示
    compute_metrics=compute_metrics
)
deberta_trainer.train()
deberta_trainer.save_model("deberta_final_model") # 保存訓練好的模型


# ================================ 獲取測試集預測 (用於融合) ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 確保模型在評估模式並移到 GPU
roberta_model_peft.eval().to(device)
deberta_model.eval().to(device)

# RoBERTa 預測
roberta_test_dataloader = DataLoader(roberta_test_dataset, batch_size=training_args.per_device_eval_batch_size)
roberta_logits_list = []
for batch in roberta_test_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = roberta_model_peft(input_ids=input_ids, attention_mask=attention_mask)
        roberta_logits_list.append(outputs.logits.cpu().numpy())

roberta_test_logits = np.vstack(roberta_logits_list)
#roberta_test_probabilities = torch.softmax(torch.tensor(roberta_test_logits), dim=-1).numpy()


# DeBERTa 預測
deberta_test_dataloader = DataLoader(deberta_test_dataset, batch_size=training_args.per_device_eval_batch_size)
deberta_logits_list = []
for batch in deberta_test_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = deberta_model(input_ids=input_ids, attention_mask=attention_mask)
        deberta_logits_list.append(outputs.logits.cpu().numpy())

deberta_test_logits = np.vstack(deberta_logits_list)

#deberta_test_probabilities = torch.softmax(torch.tensor(deberta_test_logits), dim=-1).numpy()


# ================================ 模型融合 (Logit 平均) ================================
print("\n=== 進行模型融合 (加權 Logit 平均) ===")

# 定義權重
# 這裡設定 RoBERTa 的權重為 0.6, DeBERTa 的權重為 0.4。
# 你可以根據模型在驗證集上的表現來調整這些權重。
# 通常，在驗證集上表現更好的模型可以給予更高的權重。
weight_roberta = 0.3
weight_deberta = 0.7

# 確保權重總和為 1 (如果自行調整，請確保總和為 1)
if not np.isclose(weight_roberta + weight_deberta, 1.0):
    print("警告: 權重總和不為 1，請檢查！")
    # 可以選擇標準化權重，但通常建議手動調整使其總和為 1
    # total_weight = weight_roberta + weight_deberta
    # weight_roberta /= total_weight
    # weight_deberta /= total_weight


# 執行加權 Logit 平均
averaged_logits = (weight_roberta * roberta_test_logits) + (weight_deberta * deberta_test_logits)

# 將 NumPy 陣列轉換為 PyTorch 張量，然後進行 Softmax 運算
# 如果 averaged_logits 已經是 PyTorch 張量，則可以直接使用
final_predictions_probs = torch.softmax(torch.tensor(averaged_logits), dim=-1).numpy()

# 取得最終預測的類別 (機率最高的類別)
final_predictions = np.argmax(final_predictions_probs, axis=-1)

print("融合後的預測結果前五個:", final_predictions[:5])

# ================================ 生成提交文件 ================================
submission_df = pd.DataFrame({'id': test_ids, 'target': final_predictions})
submission_df.to_csv('essemble_submission.csv', index=False)

print("\n提交文件已生成：submission.csv")
print(submission_df.head())