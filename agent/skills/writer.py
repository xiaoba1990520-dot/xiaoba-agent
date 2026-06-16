"""写稿技能 - 根据版块类型、主题和输出格式生成内容"""

from openai import OpenAI

from agent.config import LLM_TEMPERATURE, get_llm_api_key, get_llm_base_url, get_llm_model
from agent.persona import (
    OUTPUT_FORMATS,
    QUOTE_STANDARDS,
    SENSITIVITY_CHECKLIST,
    SYSTEM_PROMPT,
    TABOO_EXPRESSIONS,
    WRITER_TEMPLATES,
)


def write_script(
    topic: str,
    category: str = "文学与文字",
    output_format: str = "视频脚本",
    book_name: str = "",
    style_note: str = "",
) -> str:
    """
    生成内容稿件

    Args:
        topic: 主题或核心观点
        category: 版块类型，可选 "文学与文字" / "心理与情绪" / "哲学与思考"
        output_format: 输出格式，可选 "视频脚本" / "图文长文" / "金句卡片"
        book_name: 书名（可选）
        style_note: 风格补充说明（可选）

    Returns:
        生成的内容
    """
    template = WRITER_TEMPLATES.get(category, WRITER_TEMPLATES["文学与文字"])
    fmt = OUTPUT_FORMATS.get(output_format, OUTPUT_FORMATS["视频脚本"])

    prompt = f"""请以「小八」的身份，为以下需求写一份内容：

## 版块类型
{category}

## 主题/核心观点
{topic}

{"## 涉及书籍" + chr(10) + book_name if book_name else ""}

## 输出格式
{output_format} —— {fmt["description"]}

## 写作结构
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(template["structure"]))}

## 语气基调
{template["tone"]}

## 格式要求
{fmt["style_note"]}

{"## 风格补充" + chr(10) + style_note if style_note else ""}

## 字数范围
{fmt["word_range"][0]}-{fmt["word_range"][1]}字

## 执行标准（必须对照）

### 禁用红线（绝对禁止）
以下表达一旦出现，整段不合格：
{chr(10).join(f"- {t}" for t in TABOO_EXPRESSIONS["鸡汤句式"])}

### 细腻度检查清单
{SENSITIVITY_CHECKLIST["五感优先"]}
{SENSITIVITY_CHECKLIST["小动作放大"]}

### 金句标准（如果输出包含金句）
{QUOTE_STANDARDS["检验方法"]}

## 要求
- 直接输出内容，不要加多余的标题、标签等格式
- 严格遵循系统提示中的「文字操作手册」：句式节奏、开头规则、结尾规则、口语化清单、人称规则
- 保持「小八」的独特风格：细腻、有洞察、不鸡汤
- 结尾必须符合3种留白方式之一（开放式画面/没回答的问题/动作中止）
- 不要用"家人们""姐妹们"等称呼
- 开头不要说"今天给大家推荐"
"""

    client = OpenAI(api_key=get_llm_api_key(), base_url=get_llm_base_url())
    response = client.chat.completions.create(
        model=get_llm_model(),
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
        "description": "根据主题、版块类型和输出格式生成内容。支持三种输出格式：视频脚本（口播/创意/分镜/剧情等，适合抖音/视频号/B站/小红书视频）、图文长文（适合公众号/小红书图文）、金句卡片（适合朋友圈/小红书）。",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "内容的主题或核心观点，如'余华《活着》中福贵的坚韧'",
                },
                "category": {
                    "type": "string",
                    "enum": ["文学与文字", "心理与情绪", "哲学与思考"],
                    "description": "内容版块类型",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["视频脚本", "图文长文", "金句卡片"],
                    "description": "输出格式，默认视频脚本",
                },
                "book_name": {
                    "type": "string",
                    "description": "涉及的书籍名称，可选",
                },
                "style_note": {
                    "type": "string",
                    "description": "风格补充说明，如'更温柔一些'、'以小见大'、'独特视角'，可选",
                },
            },
            "required": ["topic", "category"],
        },
    },
}
