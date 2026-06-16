"""发布辅助技能 - 生成标题、标签、封面文案"""

from openai import OpenAI

from agent.config import LLM_TEMPERATURE, get_llm_api_key, get_llm_base_url, get_llm_model
from agent.persona import SYSTEM_PROMPT


def generate_meta(
    content: str,
    platform: str = "小红书",
    count: int = 3,
) -> str:
    """
    为已写好的内容生成发布配套：标题、标签、封面文案

    Args:
        content: 已写好的正文内容
        platform: 发布平台，如"小红书"、"抖音"、"公众号"
        count: 标题备选数量

    Returns:
        标题 + 标签 + 封面文案
    """
    prompt = f"""请为以下内容生成发布配套信息：

## 正文内容
---
{content}
---

## 发布平台
{platform}

## 输出要求

### 标题（{count}个备选）
每个标题用不同风格：
- 1个「悬念型」（留问号/省略号，引发好奇）
- 1个「共鸣型」（直击情绪，让人想点进去）
- 1个「反转型」（打破认知，制造冲突感）

### 标签/话题词（5-8个）
适配{platform}的热门标签格式，包含：
- 2-3个泛领域大标签（如#读书 #心理学）
- 2-3个精准内容标签（与正文主题相关）
- 1-2个情绪/共鸣标签（如#情感共鸣 #治愈）

### 封面文案（1句）
一句话概括核心卖点，适合放在封面图上。要求：
- 不超过20字
- 有画面感或冲击力
- 让人想点击

### 发布建议
针对{platform}的格式特点，给出1-2条发布建议（如最佳发布时间、配图建议、互动引导等）。
"""

    client = OpenAI(api_key=get_llm_api_key(), base_url=get_llm_base_url())
    response = client.chat.completions.create(
        model=get_llm_model(),
        temperature=0.8,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


META_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "generate_meta",
        "description": "为已写好的内容生成发布配套信息：标题备选、标签/话题词、封面文案、发布建议。支持小红书、抖音、公众号等平台。",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "已写好的正文内容",
                },
                "platform": {
                    "type": "string",
                    "description": "发布平台，如'小红书'、'抖音'、'公众号'，默认小红书",
                },
                "count": {
                    "type": "integer",
                    "description": "标题备选数量，默认3",
                },
            },
            "required": ["content"],
        },
    },
}
