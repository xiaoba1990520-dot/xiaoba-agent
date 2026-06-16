"""审稿技能 - 对已有文案进行多维度审阅和修改建议"""

from openai import OpenAI

from agent.config import LLM_TEMPERATURE, get_llm_api_key, get_llm_base_url, get_llm_model
from agent.persona import SYSTEM_PROMPT, REVIEW_DIMENSIONS


def review_script(
    content: str,
    category: str = "文学与文字",
    focus: str = "",
) -> str:
    """
    审阅短视频文案，给出多维度修改建议

    Args:
        content: 待审阅的文案内容
        category: 版块类型
        focus: 特别关注的维度（可选）

    Returns:
        审阅意见和修改建议
    """
    dimensions_text = chr(10).join(
        f"- **{d['name']}**：{d['description']}" for d in REVIEW_DIMENSIONS
    )

    prompt = f"""请以「小八」的视角和专业标准，审阅以下短视频文案：

## 版块类型
{category}

## 待审文案
---
{content}
---

## 审阅维度
{dimensions_text}

{"## 特别关注" + chr(10) + focus if focus else ""}

## 输出格式

### 总体评价
（一句话概括文案的整体感觉）

### 各维度评分（1-5分）
（逐项打分并说明扣分原因）

### 亮点
（列出文案中做得好的地方）

### 问题与修改建议
（逐条列出问题，每条给出具体的修改方向或改写示例）

### 修改后参考版本
（根据建议重写一版，保留原文的亮点，修复问题）"""

    client = OpenAI(api_key=get_llm_api_key(), base_url=get_llm_base_url())
    response = client.chat.completions.create(
        model=get_llm_model(),
        temperature=0.5,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


REVIEWER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "review_script",
        "description": "审阅已有的短视频文案，从钩子力、节奏感、金句度、共鸣点、人格感、平台适配等维度给出评分和修改建议。",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "待审阅的文案全文",
                },
                "category": {
                    "type": "string",
                    "enum": ["文学与文字", "心理与情绪", "哲学与思考"],
                    "description": "内容版块类型",
                },
                "focus": {
                    "type": "string",
                    "description": "特别关注的审阅维度，如'金句度'、'钩子力'，可选",
                },
            },
            "required": ["content", "category"],
        },
    },
}
