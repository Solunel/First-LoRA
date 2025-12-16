import json
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

# --- 1. 配置参数 (架构师要学会看这几个核心参数) ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" #模型名
DATA_FILE = "dataset.json" #数据文件
OUTPUT_DIR = "./output_jarvis" #输出目录
MAX_LENGTH = 384


# --- 2. 加载模型与分词器 ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")


# --- 3. 配置 LoRA (这就是微调的核心：只训练这一小部分参数) ---
config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    inference_mode = False,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, config)
# model.enable_input_require_grads()


if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def process_func(example):
    # 构建 Qwen 聊天格式
    instruction = tokenizer(
        "<|im_start|>system\n你是一个很有用的助手。<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        "<|im_start|>assistant\n",
        add_special_tokens=False
    )
    response = tokenizer(
        f"{example['output']}<|im_end|>\n",
        add_special_tokens=False
    )

    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

    # 截断（保持三者对齐）
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 读取与映射
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)
ds = Dataset.from_list(data)
tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)


# --- 5. 开始训练 (Trainer 会帮你管梯度下降) ---
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2, # 显存小就调小
    gradient_accumulation_steps=4,
    logging_steps=5,
    num_train_epochs=10,           # 因为只有8条数据，多跑几轮过拟合它！
    save_steps=50,
    learning_rate=1e-4,
    save_on_each_node=True,
    # gradient_checkpointing=True
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
print("开始训练... 请盯着 Loss 看！")
trainer.train()


# --- 6. 保存成果 ---
print("训练结束，正在保存...")
trainer.save_model(OUTPUT_DIR)
print(f"恭喜！你的专属模型已保存在 {OUTPUT_DIR}")