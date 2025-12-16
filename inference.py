import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 1. 配置路径 ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # 你的底座模型（必须和训练时一样）
LORA_PATH = "./output_jarvis"  # 你刚才训练生成的文件夹路径

# --- 2. 加载底座模型 (Base Model) ---
print("正在加载底座模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# --- 3. 加载 LoRA 补丁 (Load Adapter) ---
print(f"正在挂载 LoRA 补丁: {LORA_PATH} ...")
# 这一步是关键：把训练好的权重“贴”到底座上
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()  # 切换到评估模式

print(">>> 模型加载完毕！我是 Jarvis。请输入你的问题（输入 q 退出）：")

# --- 4. 开始对话 ---
while True:
    query = input("\nUser: ")
    if query.strip().lower() == 'q':
        break

    # 构建对话格式 (Qwen 官方格式)
    messages = [
        {"role": "system", "content": "你是一个很有用的助手。"},
        {"role": "user", "content": query}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.1,  # 稍微给点随机性
            top_p=0.1,  # 只选概率最高的词
            repetition_penalty=1.1  # 防止它复读
        )

    # 提取回答（去掉输入的 prompt 部分）
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Jarvis: {response}")