ase_model: deepseek-ai/deepseek-llm-7b-chat
library_name: peft

Model Card for Disaster Tweet Classifier using DeepSeek-LLM + LoRA

這個模型是基於 deepseek-ai/deepseek-llm-7b-chat 所進行的微調，目的是分類推文是否與真實災難相關。本模型使用了 4-bit 量化技術與 LoRA 微調，在資源有限的情況下實現高效訓練與推論。

Model Details

Model Description

Developed by: Tsai Jr-Shiang / Team7

Funded by [optional]: [More Information Needed]

Shared by [optional]: [More Information Needed]

Model type: Causal Language Model

Language(s) (NLP): English

License: Same as base model (DeepSeek)

Finetuned from model [optional]: deepseek-ai/deepseek-llm-7b-chat

Model Sources [optional]

Repository: [More Information Needed]

Paper [optional]: [More Information Needed]

Demo [optional]: [More Information Needed]

Uses

Direct Use

自動分類英文推文是否描述真實災難。適合用於災難通報、內容篩選等。

Downstream Use [optional]

作為指令微調（instruction tuning）示範，針對二分類任務設計 prompt。

Out-of-Scope Use

不適用於非英文、長文本、多類別或醫療、法律等高風險場景。

Bias, Risks, and Limitations

可能無法辨別假新聞、幽默語句、諷刺文等語義複雜內容，且依賴標記品質。

Recommendations

建議搭配人工審核流程使用，不適合作為唯一判斷依據。

How to Get Started with the Model

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("your_model_path")
base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "your_model_path")

prompt = '''
任務：判斷以下推文是否與真實災難相關。
如果推文與真實災難相關，請回答 '1'。
如果不相關，請回答 '0'。
請只回答 '1' 或 '0'，不要包含其他文字。

推文: "Forest fire near downtown."
答案:
'''

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

Training Details

Training Data

來自 Kaggle NLP 災難推文資料集 (train.csv & test.csv)

Training Procedure

格式轉為指令式 prompt

4-bit NF4 量化 + LoRA 微調 (r=4, alpha=8)

F1 score 早停監控

訓練 epoch=2（實際由早停決定）

Training Hyperparameters

Learning Rate: 2e-4（LoRA 常用範圍）

Batch Size: 4

Optimizer: paged_adamw_8bit

Scheduler: cosine

Metric: F1

Evaluation

使用 Accuracy, Precision, Recall, F1 score，驗證集為與其他模型相同的 stratified 20% 切分（test_size=0.2, random_state=42）。評估時只在每個樣本的答案 token 位置比較預測（causal LM 需往前位移一格），訓練與推理使用同一個 prompt。

Environmental Impact

使用 A100 進行 4bit + LoRA 訓練。預估碳排建議使用 ML CO2 Calculator。

Technical Specifications [optional]

使用 transformers, peft, trl 套件

硬體使用 GPU 加速 (CUDA)

Citation [optional]

@misc{deepseek,
  title={DeepSeek LLM 7B},
  author={DeepSeek AI},
  howpublished = {\url{https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat}},
  year={2024}
}

Glossary [optional]

LoRA: Low-Rank Adaptation，用於參數高效微調

SFT: Supervised Fine-Tuning，監督式微調方法

More Information [optional]

參見程式碼 deepseek_lora.py

Model Card Authors [optional]

Tsai Jr-Shiang / Team7

Model Card Contact

Email: alex93.tsai@gmail.com

Framework versions

PEFT 0.15.2

