import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# --- 0. 設定和參數 ---
model_id = "deepseek-ai/deepseek-llm-7b-chat"
output_dir = "./deepseek_lora_disaster_tweets"
learning_rate = 1.7e-5
batch_size = 4
gradient_accumulation_steps = 1
num_train_epochs = 2# 可以設定一個較大的數字，因為早停會幫你決定何時停止

# 早停參數
early_stopping_patience = 3 # 如果在這些步驟內驗證指標沒有改善，則停止訓練
early_stopping_threshold = 0.01 # 驗證指標改善的最小閾值，低於此值不算改善

# --- 1. 載入數據集 ---
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("錯誤:train.csv 或 test.csv 未找到。請確保文件在正確路徑。")
    print("在 Kaggle Notebook 中，路徑通常是 /kaggle/input/nlp-getting-started/train.csv 等。")
    exit()

print(f"訓練數據集大小：{len(train_df)}")
print(f"測試數據集大小：{len(test_df)}")

train_subset_df = train_df.sample(frac=0.8, random_state=42)
train_data, eval_data = train_test_split(train_subset_df, test_size=0.1, random_state=42, stratify=train_subset_df['target'])

# --- 2. 載入 Tokenizer (修正: 提前到這裡) ---
# 提前載入 tokenizer，以便在數據格式化函數中使用
print(f"\n正在載入 Tokenizer: {model_id}...")


tokenizer = AutoTokenizer.from_pretrained(model_id, truncation=True, padding=True, max_length=141)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # 為生成任務設置為右側填充
tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer 載入完成。")
# 移除可能導致錯誤的行
# del your_tensor_name
# del your_model_instance
gc.collect()

# --- 3. 格式化數據集為指令微調格式 (Instruction Tuning Format) ---
def format_data_for_sft(sample):
    text = sample['text']
    target = str(sample['target'])

    messages = [
        {"role": "user", "content": f"任務：判斷以下推文是否與真實災難相關。\n如果推文與真實災難相關 請回答 '1'。\n如果推文與真實災難不相關 請回答 '0'。\n請只回答 '1' 或 '0'，不要包含其他文字。\n\n推文: \"{text}\"\n答案:"},
        {"role": "assistant", "content": target}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": formatted_text}

# 創建 Hugging Face Dataset 對象
train_dataset = Dataset.from_pandas(train_data).map(format_data_for_sft, remove_columns=['id', 'keyword', 'location', 'text', 'target'])
eval_dataset = Dataset.from_pandas(eval_data).map(format_data_for_sft, remove_columns=['id', 'keyword', 'location', 'text', 'target'])

print("\n數據集格式化範例 (訓練集第一條):")
print(train_dataset[0]['text'])

# --- 4. 載入 4-bit 量化模型 (原步驟 3，現在是 4) ---
print(f"\n正在載入 4-bit 量化模型: {model_id}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", # 這裡會自動將模型載入到 CUDA
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
print("模型已為 k-bit 訓練準備就緒。")

# --- 5. 配置 LoRA (原步驟 4，現在是 5) ---
peft_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.2,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
print("模型已應用 LoRA 配置。")
model.print_trainable_parameters()

# --- 6. 定義 compute_metrics 函數 (新增加的部分) ---
# 這個函數會在 evaluation_strategy 步驟時被調用
def compute_metrics(eval_preds):
    predictions_ids = eval_preds.predictions.argmax(axis=-1)
    labels_ids = eval_preds.label_ids

    token_0_id = tokenizer.convert_tokens_to_ids('0')
    token_1_id = tokenizer.convert_tokens_to_ids('1')

    parsed_preds = []
    parsed_labels = []

    for i in range(predictions_ids.shape[0]):
        sample_pred_ids = predictions_ids[i]
        sample_label_ids = labels_ids[i]

        answer_start_idx = -1
        for j in range(len(sample_label_ids)):
            if sample_label_ids[j] != -100:
                answer_start_idx = j
                break

        if answer_start_idx != -1:
            true_token_id = sample_label_ids[answer_start_idx]
            if true_token_id == token_0_id:
                parsed_labels.append(0)
            elif true_token_id == token_1_id:
                parsed_labels.append(1)
            else:
                continue # Skip if the label is not 0 or 1

            predicted_token_id = sample_pred_ids[answer_start_idx]
            if predicted_token_id == token_0_id:
                parsed_preds.append(0)
            elif predicted_token_id == token_1_id:
                parsed_preds.append(1)
            else:
                # If prediction is neither '0' nor '1', assign a default (e.g., 0)
                # or handle as an error/unclear prediction.
                # For classification, it's often safer to assign a default.
                parsed_preds.append(0)

    if len(parsed_labels) == 0:
        # Avoid division by zero if no valid labels were found
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    acc = accuracy_score(parsed_labels, parsed_preds)
    precision = precision_score(parsed_labels, parsed_preds, average='binary', zero_division=0)
    recall = recall_score(parsed_labels, parsed_preds, average='binary', zero_division=0)
    f1 = f1_score(parsed_labels, parsed_preds, average='binary', zero_division=0)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# --- 7. 配置訓練參數 ---
training_arguments = TrainingArguments(
    output_dir=output_dir,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs, # 可以設定一個較大的數字，早停會控制實際的訓練步數
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    warmup_steps=130,
    eval_strategy="steps",     # Evaluate every epoch
    save_strategy="steps", 
    eval_steps=100,
    fp16=True, # 啟用混合精度訓練，這將大大幫助 GPU 顯存使用
    optim="paged_adamw_8bit", # 8-bit 優化器，進一步節省顯存
    report_to="none",
    push_to_hub=False,
    load_best_model_at_end=True, # 確保在訓練結束時載入最佳模型
    metric_for_best_model="f1", # 早停監控的指標
    greater_is_better=True, # 對於 F1 分數，越大越好
)

# --- 8. 創建 SFTTrainer ---
# 實例化 EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=early_stopping_patience,
    early_stopping_threshold=early_stopping_threshold
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=training_arguments,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback], # 將早停回調添加到這裡
)

# --- 8. 開始訓練 (原步驟 7，現在是 8) ---
print("\n開始 LoRA 微調...")
# 確認 PyTorch 是否偵測到 CUDA，並在訓練前列印出來
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 設備數量: {torch.cuda.device_count()}")
    print(f"當前 CUDA 設備名稱: {torch.cuda.get_device_name(0)}")
else:
    print("警告: CUDA 不可用，訓練將在 CPU 上進行，這將非常緩慢。")

trainer.train() # SFTTrainer 會自動處理數據到 GPU 的傳輸

print("微調完成！")

# 由於設置了 load_best_model_at_end=True，這裡保存的就是最佳模型
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"微調後的 LoRA 適配器和 Tokenizer 已保存到 {output_dir}")

# --- 9. 推理 (使用微調後的模型) (原步驟 8，現在是 9) ---
print("\n開始使用微調後的模型進行推理...")

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", # 這裡會自動將模型載入到 CUDA
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
base_model.eval()

from peft import PeftModel
model_for_inference = PeftModel.from_pretrained(base_model, output_dir)
print("微調後的模型已載入。")

texts_to_classify = test_df['text'].tolist()
tweet_ids = test_df['id'].tolist()
final_predictions = []

for i, tweet_text in enumerate(texts_to_classify):
    prompt = f"""
    任務：判斷以下推文是否與真實災難相關。
    如果推文與真實災難相關，請回答 '1'。
    如果推文與真實災難不相關，請回答 '0'。\n請只回答 '1' 或 '0'，不要包含其他文字。\n\n推文: \"{tweet_text}\"\n答案:
    """
    messages = [
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = model_for_inference.generate(
            inputs,
            max_new_tokens=5,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()

    predicted_label = -1
    if generated_response.startswith('1'):
        predicted_label = 1
    elif generated_response.startswith('0'):
        predicted_label = 0
    else:
        print(f"警告: 推文 ID {tweet_ids[i]} 的模型輸出無法解析: '{generated_response}'。設定為 0。")
        predicted_label = 0

    final_predictions.append(predicted_label)
    print(f"推文 ID: {tweet_ids[i]}, 文本: '{tweet_text[:50]}...', 預測 (微調後): {predicted_label}")

# --- 10. 創建提交文件 (原步驟 9，現在是 10) ---
submission_df = pd.DataFrame({'id': tweet_ids, 'target': final_predictions})
submission_df.to_csv('submission_deepseek_lora.csv', index=False)
print("\n提交文件 'submission_deepseek_lora.csv' 已生成！")

# --- 11. 清理顯存 (原步驟 10，現在是 11) ---
del model
del base_model
del model_for_inference
del tokenizer
del inputs
del outputs
gc.collect()
torch.cuda.empty_cache()
print("\n顯存清理完成。")