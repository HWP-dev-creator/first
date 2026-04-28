from openai import OpenAI
import json

# 配置 DeepSeek
client = OpenAI(
    api_key="sk-31ecfce4e0164bc0950fa25e8db33c09",
    base_url="https://api.deepseek.com"
)
def analyze_sentiment(comment: str) -> dict:
    prompt = f"""请判断以下评论的情感倾向。
    
示例：
评论：这家店的服务态度超级好，环境也很干净，强烈推荐！
分析：{{"sentiment": "positive", "confidence": 0.95, "keywords": ["服务态度好", "环境干净", "推荐"]}}

评论：一般般吧，没什么特别的，价格还偏贵。
分析：{{"sentiment": "neutral", "confidence": 0.7, "keywords": ["一般", "价格偏贵"]}}

评论：等了一个小时才上菜，味道还很难吃，再也不会来了。
分析：{{"sentiment": "negative", "confidence": 0.92, "keywords": ["上菜慢", "味道难吃"]}}

现在请分析：
评论：{comment}
分析："""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # 低温度，确保输出稳定
        max_tokens=200
    )
    
    # 解析JSON（实际要做异常处理）
    import json
    result_text = response.choices[0].message.content.strip()
    
    # 有时候模型会返回 ```json ... ``` 包裹的内容，需要清理
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```")[1].split("```")[0].strip()
    
    return json.loads(result_text)

# 测试
test_comments = [
    "快递速度真快，包装也很严实，五星好评！",
    "和描述不符，质量太差了，退货麻烦死了",
    "还行吧，凑合用"
]

for c in test_comments:
    result = analyze_sentiment(c)
    print(f"评论：{c}")
    print(f"结果：{result}")
    print("-" * 40)


import re

def solve_math(problem: str, use_cot: bool = False) -> str:
    if use_cot:
        prompt = f"""请解决以下数学问题，并一步一步详细说明推理过程。

问题：{problem}

请按以下格式回答：
步骤1：...
步骤2：...
...
最终答案：..."""
    else:
        prompt = f"请直接给出答案：{problem}"
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # 数学题要确定性
        max_tokens=500
    )
    
    return response.choices[0].message.content

# 测试题目（故意选容易错的）
problems = [
    "一个水池，甲管单独注满要6小时，乙管单独注满要4小时。如果两管同时打开，多久能注满？",
    "小明今年8岁，他爸爸今年35岁。几年后爸爸的年龄是小明的2倍？",
    "某商品先涨价20%，再降价20%，最终价格是原价的百分之几？"
]

print("=" * 60)
print("【不加CoT】直接给答案：")
print("=" * 60)
for p in problems:
    print(f"\n题目：{p}")
    print(f"答案：{solve_math(p, use_cot=False)}")
    print("-" * 40)

print("\n" + "=" * 60)
print("【加CoT】逐步推导：")
print("=" * 60)
for p in problems:
    print(f"\n题目：{p}")
    print(f"答案：{solve_math(p, use_cot=True)}")
    print("-" * 40)
def extract_info(text: str) -> dict:
    prompt = f"""请从以下文本中提取关键信息，并以JSON格式返回。

示例1：
文本："我叫张伟，电话13800138000，住在北京市朝阳区建国路88号，想咨询一下贷款业务。"
提取结果：
{{
    "name": "张伟",
    "phone": "13800138000",
    "address": "北京市朝阳区建国路88号",
    "intent": "咨询贷款"
}}

示例2：
文本："李小姐，手机号是15912345678，她对你们的产品很感兴趣，想了解价格和功能。"
提取结果：
{{
    "name": "李小姐",
    "phone": "15912345678",
    "address": null,
    "intent": "了解产品价格和功能"
}}

现在请提取：
文本："{text}"
提取结果："""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300
    )
    
    import json
    result_text = response.choices[0].message.content.strip()
    
    # 清理可能的markdown代码块
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```")[1].split("```")[0].strip()
    
    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        return {"raw": result_text, "error": "JSON解析失败"}

# 测试
test_texts = [
    "你好，我是王强，号码是18666668888，公司地址在上海浦东新区张江高科，想预约下周的产品演示。",
    "联系一下赵经理，电话13777779999，他说想投诉你们的服务态度。",
    "有个客户留言说'产品不错，就是太贵了'，没留联系方式。"
]

for t in test_texts:
    print(f"原文：{t}")
    print(f"提取：{extract_info(t)}")
    print("=" * 50)