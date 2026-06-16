"""写稿技能 - 根据版块类型和主题生成短视频文案"""

import json
from openai import OpenAI

from agent.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_TEMPERATURE
from agent.persona import SYSTEM_PROMPT, WRITER_TEMPLATES


def write_script(
    topic: str,
    category: str = "文学书籍解读",
    book_name: str = "",
    target_duration: str = "1-2分钟",
    style_note: str = "",
) -> str:
    """
    生成短视频口播文案

    Args:
        topic: 主题或核心观点
        category: 版块类型，可选 "文学书籍解读" / "心理类感悟" / "女性成长感悟"
        book_name: 书名（可选）
        target_duration: 目标时长，如 "1-2分钟"
        style_note: 风格补充说明（可选）

    Returns:
        生成的口播文案
    """
    template = WRITER_TEMPLATES.get(category, WRITER_TEMPLATES["文学书籍解读"])

    prompt = f"""请以「小八」的身份，为以下需求写一篇短视频口播文案：

## 版块类型
{category}

## 主题/核心观点
{topic}

{"## 涉及书籍" + chr(10) + book_name if book_name else ""}

## 目标时长
{target_duration}

## 写作结构
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(template["structure"]))}

## 语气基调
{template["tone"]}

{"## 风格补充" + chr(10) + style_note if style_note else ""}

## 字数范围
{template["word_range"][0]}-{template["word_range"][1]}字

## 要求
- 直接输出文案内容，不要加标题、标签等格式
- 保持「小八」的口语化风格，像在和朋友聊天
- 结尾留一个开放式问题引发讨论
- 不要用"家人们""姐妹们"等称呼
- 开头不要说"今天给大家推荐"
"""

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


# 供 OpenAI function calling 使用的工具定义
WRITER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_script",
        "description": "根据主题和版块类型生成短视频口播文案。可选择的版块类型：文学书籍解读、心理类感悟、女性成长感悟。",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "文案的主题或核心观点，如'余华《活着》中福贵的坚韧'",
                },
                "category": {
                    "type": "string",
                    "enum": ["文学书籍解读", "心理类感悟", "女性成长感悟"],
                    "description": "内容版块类型",
                },
                "book_name": {
                    "type": "string",
                    "description": "涉及的书籍名称，可选",
                },
                "target_duration": {
                    "type": "string",
                    "description": "目标口播时长，如'1-2分钟'，默认1-2分钟",
                },
                "style_note": {
                    "type": "string",
                    "description": "风格补充说明，如'更温柔一些'、'加入个人经历'，可选",
                },
            },
            "required": ["topic", "category"],
        },
    },
}
