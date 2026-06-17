"""校对技能 - 检查错别字、语病、标点、语言流畅度"""

from openai import OpenAI

from agent.config import LLM_TEMPERATURE, get_llm_api_key, get_llm_base_url, get_llm_model
from agent.persona import SYSTEM_PROMPT, TABOO_EXPRESSIONS, SENSITIVITY_CHECKLIST


def proofread(content: str) -> str:
    """
    校对文字：检查错别字、语病、标点、语言流畅度

    Args:
        content: 待校对的文字内容

    Returns:
        校对结果：列出问题 + 修改后的版本
    """
    # 构建禁用红线文本
    taboo_lines = "\n".join(f"- {t}" for t in TABOO_EXPRESSIONS["鸡汤句式"])
    taboo_alts = "\n".join(f"- {t}" for t in TABOO_EXPRESSIONS["正确替代方式"])

    prompt = f"""请以「小八」的视角和标准，对以下内容进行校对。你不是冷冰冰的编辑，你是一个对文字有感受力的创作者——发现问题时，像朋友一样指出，像同行一样给建议。

## 待校对内容
---
{content}
---

## 你的校对标准

### 1. 文字质感（节奏与呼吸）
- 短句是否控制了长度？有没有该断不断的长句？
- 标点是否在按呼吸而非语法使用？
- 是否有连续3个同长度句子？

### 2. 不鸡汤检查（绝对红线）
以下表达一旦出现，必须指出并替换：
{taboo_lines}

正确替代思路：
{taboo_alts}

### 3. 细腻度检查
{SENSITIVITY_CHECKLIST["五感优先"]}
{SENSITIVITY_CHECKLIST["小动作放大"]}

### 4. 基础问题
- 错别字、生造词、同音字混淆
- 语病：成分残缺、搭配不当、语序混乱
- 标点规范：不要一逗到底
- 语气一致性：是否突然像换了个人在写

## 输出格式

### 问题清单
逐条列出，每条用这样的语气：
- 「原文位置」：第几句/哪个词
- 「我的感受」：这句话读起来哪里别扭（说感受，不说术语）
- 「可以试试」：给一个具体的修改方向或改写示例

### 校对后版本
提供修改后的完整文字，保留原文好的地方，只修正问题。

### 总体评价
用一句话概括，像小八会说的话：这篇文字最打动人的地方是什么，哪里还可以再磨一磨。"""

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
