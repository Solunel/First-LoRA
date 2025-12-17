import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. 架构配置 ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # 回归底座
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在加载底座模型 {MODEL_ID} (用于评估 Prompt 能力)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# --- 2. 定义“微型”知识库 (Schema Context) ---
# 既然是微型锚点，我们就假定只有一个员工表。
# 在真实 RAG 中，这部分是由向量库检索出来的。
DB_SCHEMA = """
            CREATE TABLE employees \
            ( \
                id         INT PRIMARY KEY, \
                name       VARCHAR(50), \
                department VARCHAR(50), \
                salary     INT, \
                join_date  DATE
            ); \
            """


# --- 3. 架构师核心逻辑：构建 Prompt ---
def generate_sql(user_query):
    # System Prompt: 稍微精简一下，把重点放在表结构上
    # System Prompt: 强制要求先思考，后写 SQL
    system_prompt = f"""你是一个SQL生成助手。
    【表结构】：
    {DB_SCHEMA}

    【输出格式要求】：
    请严格按照以下两步输出：
    Step 1: 检查用户请求的字段是否在表结构中存在。如果存在，列出字段名；如果不存在，说明缺失。
    Step 2: 如果字段存在，输出 SQL；如果缺失，输出 ERROR。
    """

    messages = [
        {"role": "system", "content": system_prompt},

        # 样例 1：正常查询
        {"role": "user", "content": "查询所有员工的姓名"},
        {"role": "assistant",
         "content": "Step 1: 用户请求“姓名”，对应字段 name，存在。\nStep 2: SELECT name FROM employees;"},

        # 样例 2：拒绝查询（关键！）
        # 我们直接把你的 Case C 做成样例，但这叫“过拟合”。
        # 为了通用性，我们还是用“身高”做例子，但强制它“把思考过程写出来”。
        {"role": "user", "content": "查询员工的身高"},
        {"role": "assistant",
         "content": "Step 1: 用户请求“身高”，表结构中有 id, name, department, salary, join_date。未找到“身高”相关字段。\nStep 2: ERROR: 缺少相关字段无法查询"},

        # 用户的真实问题
        {"role": "user", "content": user_query}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.01,  # 保持低温
            top_p=0.95
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    final_output = response.strip()

    # --- 架构师的后处理逻辑 (Parser) ---
    # 我们需要从模型的“碎碎念”中提取出真正的 SQL 或 错误信息

    # 情况 1: 模型决定报错
    if "ERROR:" in final_output:
        # 提取 ERROR 后面的内容
        error_msg = final_output.split("ERROR:")[-1].strip()
        return f"[拒识] {error_msg}"

    # 情况 2: 模型生成了 SQL
    # 我们假设 SQL 在 Step 2 里，通常包含 SELECT
    if "SELECT" in final_output.upper():
        # 简单粗暴提取：找到 SELECT 开始的位置
        import re
        # 提取 SQL 语句 (假设它以 SELECT 开头，分号结尾)
        match = re.search(r"(SELECT.*?;)", final_output, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    # 情况 3: 兜底
    return final_output

# --- 5. 测试循环 ---
if __name__ == "__main__":
    print("\n>>> SQL 助手已启动。表结构：employees(id, name, department, salary, join_date)")
    print(">>> 请输入问题 (输入 q 退出):")

    while True:
        query = input("\nUser: ")
        if query.strip().lower() == 'q':
            break

        sql = generate_sql(query)
        print(f"SQL: {sql}")

        # 简单的自检逻辑
        if "SELECT" not in sql.upper():
            print("[警告] 模型生成的似乎不是 SQL！")