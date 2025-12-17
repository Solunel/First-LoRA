import json
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

# 参数
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" #模型名
DATA_FILE = "dataset.json" #数据文件
OUTPUT_DIR = "./output_jarvis" #输出目录
MAX_LENGTH = 384 #最大长度


# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    inference_mode = False,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, config)


# 数据处理
def process_func(example):
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
      "<|im_start|>system\n你是一个很有用的助手。<|im_end|>\n"
      f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
      "<|im_start|>assistant\n",
      add_special_tokens=False)
    response = tokenizer(f"{example['output']}<|im_end|>\n", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)
ds = Dataset.from_list(data)
tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)


# 训练
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    save_strategy="epoch",
    logging_steps=1,
    num_train_epochs=10,

    learning_rate=1e-4,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()


# 保存模型
trainer.save_model(OUTPUT_DIR)