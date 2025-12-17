import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 参数设置
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./output_jarvis"
TEST_INSTRUCTION = "你是谁" # 测试问题

# 2. 加载模型和分词器
print("正在加载底模...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)

# 3. 加载你微调好的 LoRA 适配器
print(f"正在加载 LoRA: {LORA_PATH}...")
model = PeftModel.from_pretrained(model, LORA_PATH)


# 4. 手动构建 Prompt
prompt = (
    "<|im_start|>system\n你是一个很有用的助手。<|im_end|>\n"
    f"<|im_start|>user\n{TEST_INSTRUCTION}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# 5. 编码输入
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 6. 生成 (Generate)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=384,  # 最多生成多少个新字

        pad_token_id=tokenizer.eos_token_id, # 遇到结束符停止
        eos_token_id=tokenizer.eos_token_id,

        do_sample=True,      # 随机采样，让回答更生动（False则是贪婪搜索，每次只选概率最大的）
        temperature=0.1      # 随机程度
    )

# 7. 解码输出
# outputs 包含了 [输入的问题 + 模型生成的回答]，我们需要把输入部分切掉，只看新生成的
generated_ids = outputs[0][len(inputs.input_ids[0]):]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("模型回答：")
print(response)