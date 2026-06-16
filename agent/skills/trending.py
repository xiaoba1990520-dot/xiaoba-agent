"""找热点技能 - 结合时事热点找到内容创作角度"""

from openai import OpenAI

from agent.config import LLM_TEMPERATURE, get_llm_api_key, get_llm_base_url, get_llm_model
from agent.persona import SYSTEM_PROMPT, TRENDING_ANGLES


def find_trending(
    topic: str = "",
    category: str = "",
    count: int = 3,
) -> str:
    """
    结合当前热点，找到与读书博主内容方向的关联角度

    Args:
        topic: 指定的热点话题（可选，不指定则让AI自行联想近期热点）
        category: 限定版块类型（可选，不限定则覆盖三大版块）
        count: 返回的角度数量

    Returns:
        热点内容角度建议
    """
    angles_text = chr(10).join(f"- {a}" for a in TRENDING_ANGLES)

    topic_instruction = f"围绕热点话题「{topic}」，" if topic else "联想近期社会热点，"
    category_instruction = f"聚焦「{category}」版块，" if category else "覆盖三大版块（文学书籍解读、心理类感悟、女性成长感悟），"

    prompt = f"""请以「小八」的视角，{topic_instruction}{category_instruction}找到至少{count}个可以创作短视频内容的角度。

## 可用的关联角度思路
{angles_text}

## 输出格式

为每个角度提供：

### 角度N：[吸引人的标题]
- **关联热点**：与哪个热点话题关联
- **版块归属**：文学书籍解读 / 心理类感悟 / 女性成长感悟
- **核心观点**：一句话概括你想表达什么
- **推荐书籍**：1-2本可以引用的书
- **开场钩子**：文案开头可以用什么钩子
- **预估共鸣度**：高/中/低，以及理由
- **注意事项**：是否有敏感点需要规避或需要特别表达方式

## 要求
- 角度要独特，不是大家都想到的那个
- 每个角度必须有明确的书籍关联
- 钩子要有冲击力，能让人停下刷手机的手
- 注意女性视角和情感共鸣
"""

    client = OpenAI(api_key=get_llm_api_key(), base_url=get_llm_base_url())
    response = client.chat.completions.create(
        model=get_llm_model(),
        temperature=0.9,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


TRENDING_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_trending",
        "description": "结合当前社会热点，找到与读书博主内容方向（文学/心理/女性成长）关联的创作角度。可指定热点话题或让AI自行联想。",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "指定的热点话题，如'春节返乡'、'职场PUA'，不指定则AI自行联想近期热点",
                },
                "category": {
                    "type": "string",
                    "enum": ["文学书籍解读", "心理类感悟", "哲学与生活"],
                    "description": "限定版块类型，不指定则覆盖三大版块",
                },
                "count": {
                    "type": "integer",
                    "description": "返回的角度数量，默认3",
                },
            },
            "required": [],
        },
    },
}
