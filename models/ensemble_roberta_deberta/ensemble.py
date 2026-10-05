import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
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

# ================================ 數據準備部分 ================================
import pandas as pd

train_file_path = "train.csv"
test_file_path = "test.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# 切出驗證集：原本 eval_dataset 直接用訓練集，load_best_model_at_end 等於用訓練集 F1 挑模型，
# 融合權重也沒有任何依據。所有模型共用同一個切分：test_size=0.2, random_state=42, stratify=target
train_split_df, val_split_df = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df['target']
)

X_train_text = train_split_df['text'].values
y_train_labels = train_split_df['target'].values
X_val_text = val_split_df['text'].values
y_val_labels = val_split_df['target'].values
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

# ================================ 模型訓練和預測流程 ================================
# 1. 設置模型和 Tokenizer
roberta_model_name = "roberta-base"
deberta_model_name = "microsoft/deberta-v3-base" # DeBERTa-base 是一個常用的選擇

roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)

# 2. 數據分詞 (Tokenization)
def encode(tokenizer, texts):
    return tokenizer(list(texts), truncation=True, padding=True, max_length=141)

# 3. 創建 Dataset 對象（測試集只有 encodings，沒有 labels）
roberta_train_dataset = DisasterTweetsDataset(encode(roberta_tokenizer, X_train_text), y_train_labels)
roberta_val_dataset   = DisasterTweetsDataset(encode(roberta_tokenizer, X_val_text), y_val_labels)
roberta_test_dataset  = DisasterTweetsDataset(encode(roberta_tokenizer, X_test_text))

deberta_train_dataset = DisasterTweetsDataset(encode(deberta_tokenizer, X_train_text), y_train_labels)
deberta_val_dataset   = DisasterTweetsDataset(encode(deberta_tokenizer, X_val_text), y_val_labels)
deberta_test_dataset  = DisasterTweetsDataset(encode(deberta_tokenizer, X_test_text))


# 4. 設置訓練參數 (Hugging Face Trainer)
num_labels = 2 # 0: non-disaster, 1: disaster

def make_training_args(output_dir, learning_rate):
    # 兩個模型各自的 output_dir：原本共用 ./results，DeBERTa 的 checkpoint 會蓋掉 RoBERTa 的
    return TrainingArguments(
        output_dir=output_dir,           # output directory
        learning_rate=learning_rate,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=130,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=os.path.join(output_dir, 'logs'),  # directory for storing logs
        logging_steps=100,
        eval_strategy="epoch",     # Evaluate every epoch
        save_strategy="epoch",           # Save checkpoint every epoch
        save_total_limit=1,
        load_best_model_at_end=True,     # Load the best model at the end of training
        metric_for_best_model="f1",      # Use F1-score to determine the best model
        greater_is_better=True,
        report_to="none"                 # 禁用 wandb 等報告，如果需要啟用可配置
    )

# LoRA 只訓練少量參數，學習率通常要比全參數微調高一個數量級（1e-4 ~ 5e-4）；
# 原本兩者共用 1.7e-5，RoBERTa-LoRA 很可能訓練不足
roberta_training_args = make_training_args('./results_roberta', 2e-4)
deberta_training_args = make_training_args('./results_deberta', 1.7e-5)

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

# 2. 設置 LoRA 配置，task_type 設置為 SEQ_CLS
peft_config_roberta = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "key", "value"], # RoBERTa 的 attention 層通常是 q, k, v，而不是 q_proj, k_proj, v_proj
    lora_dropout=0.2,
    bias="none",
    task_type=TaskType.SEQ_CLS, # 對於文本分類是 SEQ_CLS
)
# 3. 將 LoRA 適配器應用到模型
roberta_model_peft = get_peft_model(roberta_base_model, peft_config_roberta)
roberta_model_peft.print_trainable_parameters() # 打印可訓練參數數量

roberta_trainer = Trainer(
    model=roberta_model_peft, # 傳入應用 LoRA 的模型
    args=roberta_training_args,
    train_dataset=roberta_train_dataset,
    eval_dataset=roberta_val_dataset,
    compute_metrics=compute_metrics,
)
roberta_trainer.train()
roberta_trainer.save_model("roberta_final_model") # 保存訓練好的模型

# ================================ 訓練 DeBERTa 模型 ================================
print("\n=== 開始訓練 DeBERTa 模型 ===")
deberta_model = AutoModelForSequenceClassification.from_pretrained(deberta_model_name, num_labels=num_labels)

deberta_trainer = Trainer(
    model=deberta_model,
    args=deberta_training_args,
    train_dataset=deberta_train_dataset,
    eval_dataset=deberta_val_dataset,
    compute_metrics=compute_metrics
)
deberta_trainer.train()
deberta_trainer.save_model("deberta_final_model") # 保存訓練好的模型


# ================================ 取得驗證集與測試集預測 ================================
def predict_probs(trainer, dataset):
    logits = trainer.predict(dataset).predictions
    return torch.softmax(torch.tensor(logits), dim=-1).numpy()

roberta_val_probs  = predict_probs(roberta_trainer, roberta_val_dataset)
deberta_val_probs  = predict_probs(deberta_trainer, deberta_val_dataset)
roberta_test_probs = predict_probs(roberta_trainer, roberta_test_dataset)
deberta_test_probs = predict_probs(deberta_trainer, deberta_test_dataset)


# ================================ 模型融合 (加權機率平均) ================================
print("\n=== 進行模型融合 (加權機率平均) ===")

# 兩個模型的 logits 尺度不同（一個 LoRA、一個全參數微調），先轉成機率再平均較合理。
# 權重在驗證集上搜尋 F1 最佳值；原本 0.3 / 0.7 是手動設定、沒有經過驗證。
# 注意：權重和回報的融合 F1 用的是同一份驗證集，回報值會略為樂觀。
def val_f1(w):
    probs = w * roberta_val_probs + (1 - w) * deberta_val_probs
    return f1_score(y_val_labels, probs.argmax(-1))

weight_roberta = max(np.linspace(0, 1, 21), key=val_f1)
weight_deberta = 1 - weight_roberta

print(f"RoBERTa 驗證集 F1: {val_f1(1.0):.4f}")
print(f"DeBERTa 驗證集 F1: {val_f1(0.0):.4f}")
print(f"融合權重: RoBERTa {weight_roberta:.2f} / DeBERTa {weight_deberta:.2f}，融合後驗證集 F1: {val_f1(weight_roberta):.4f}")

final_predictions_probs = weight_roberta * roberta_test_probs + weight_deberta * deberta_test_probs

# 取得最終預測的類別 (機率最高的類別)
final_predictions = np.argmax(final_predictions_probs, axis=-1)

print("融合後的預測結果前五個:", final_predictions[:5])

# ================================ 生成提交文件 ================================
submission_df = pd.DataFrame({'id': test_ids, 'target': final_predictions})
submission_df.to_csv('ensemble_submission.csv', index=False)

print("\n提交文件已生成：ensemble_submission.csv")
print(submission_df.head())
