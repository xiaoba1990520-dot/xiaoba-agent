"""校对技能 - 检查错别字、语病、标点、语言流畅度"""

from openai import OpenAI

from agent.config import LLM_TEMPERATURE, get_llm_api_key, get_llm_base_url, get_llm_model
from agent.persona import SYSTEM_PROMPT


def proofread(content: str) -> str:
    """
    校对文字：检查错别字、语病、标点、语言流畅度

    Args:
        content: 待校对的文字内容

    Returns:
        校对结果：列出问题 + 修改后的版本
    """
    prompt = f"""请以专业文字编辑的标准，对以下内容进行校对。

## 待校对内容
---
{content}
---

## 校对维度

1. **错别字与用词**：是否有错别字、生造词、网络用语误用、同音字混淆
2. **语病与语法**：是否有成分残缺、搭配不当、语序混乱、表意不明
3. **标点符号**：逗号、句号、引号、破折号等使用是否规范；是否存在一逗到底或标点缺失
4. **语言流畅度**：句子是否通顺自然、有无重复啰嗦、有无逻辑跳跃、读起来是否舒服
5. **语气一致性**：全文语气是否统一，有无突然的语气断裂

## 输出格式

### 问题清单
逐条列出发现的问题，每条说明：
- 原文位置（第几句/哪个词）
- 问题类型（错别字/语病/标点/流畅度）
- 具体问题描述
- 修改建议

### 校对后版本
提供修改后的完整文字，保留原文好的地方，只修正问题。

### 总体评价
用一句话概括：这篇文字的文字功底如何，主要问题集中在哪方面。"""

    client = OpenAI(api_key=get_llm_api_key(), base_url=get_llm_base_url())
    response = client.chat.completions.create(
        model=get_llm_model(),
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


PROOFREADER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "proofread",
        "description": "对文字进行校对，检查错别字、语病、标点符号和语言流畅度，返回问题清单和修改后的版本。",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "待校对的文字全文",
                },
            },
            "required": ["content"],
        },
    },
}
