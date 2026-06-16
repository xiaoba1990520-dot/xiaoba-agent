"""智能体核心调度 - 基于 OpenAI function calling 的 Agent 循环（兼容 DeepSeek 等不支持 tools 的平台）"""

import json
import re
from openai import OpenAI

from agent.config import LLM_TEMPERATURE, get_llm_api_key, get_llm_base_url, get_llm_model
from agent.persona import SYSTEM_PROMPT
from agent.skills import TOOL_SCHEMAS, TOOL_FUNCTIONS


def _supports_tools() -> bool:
    """检测当前 API 是否支持 function calling"""
    base = get_llm_base_url()
    return "openai.com" in base


def _build_tools_prompt() -> str:
    """为不支持 function calling 的模型构建工具描述"""
    lines = [
        "\n## 你的可用技能",
        "当你需要执行以下操作时，请在回复末尾以 JSON 格式输出工具调用指令：",
    ]
    for tool in TOOL_SCHEMAS:
        func = tool["function"]
        lines.append(f"\n### {func['name']}")
        lines.append(f"{func['description']}")
        params = func.get("parameters", {}).get("properties", {})
        for name, info in params.items():
            req = "（必填）" if name in func.get("parameters", {}).get("required", []) else "（可选）"
            lines.append(f"- {name}{req}: {info.get('description', '')}")
    lines.append("\n工具调用格式（放在回复最后，单独一行）：")
    lines.append('{"tool": "工具名", "args": {"参数名": "参数值"}}')
    lines.append("如果不需调用工具，正常回复即可，不要输出 JSON。")
    return "\n".join(lines)


def _parse_tool_call(text: str):
    """从文本中解析工具调用 JSON"""
    # 尝试从末尾提取 JSON
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                if "tool" in data and "args" in data:
                    return data["tool"], data["args"]
            except json.JSONDecodeError:
                continue
    # 尝试正则匹配
    match = re.search(r'\{[^}]*"tool"[^}]*\}', text)
    if match:
        try:
            data = json.loads(match.group())
            if "tool" in data and "args" in data:
                return data["tool"], data["args"]
        except json.JSONDecodeError:
            pass
    return None, None


def _format_memory(memory: dict) -> str:
    """将记忆格式化为 system prompt 附加文本"""
    if not memory:
        return ""
    lines = ["\n## 关于这位创作者的记忆"]
    for key, value in memory.items():
        lines.append(f"- {key}：{value}")
    lines.append("\n请基于以上记忆，在对话中自然地体现对创作者偏好的了解。")
    return "\n".join(lines)


class BookBloggerAgent:
    """文字型内容创作者智能体「小八」"""

    def __init__(self):
        self.client = OpenAI(api_key=get_llm_api_key(), base_url=get_llm_base_url())
        self.supports_tools = _supports_tools()
        self.memory = {}  # 用户记忆系统
        self._build_system_message()

    def _build_system_message(self):
        """构建包含记忆的 system message"""
        system_content = SYSTEM_PROMPT
        if not self.supports_tools:
            system_content += _build_tools_prompt()
        system_content += _format_memory(self.memory)
        self.messages = [
            {"role": "system", "content": system_content},
        ]

    def set_memory(self, key: str, value: str):
        """设置用户记忆"""
        self.memory[key] = value
        # 重建 system message 以包含新记忆
        self._build_system_message()

    def get_memory(self) -> dict:
        """获取当前记忆"""
        return self.memory.copy()

    def _extract_memory(self, user_input: str):
        """从用户输入中自动提取值得记忆的信息。返回 (key, value) 或 None"""
        hint_keywords = [
            "喜欢", "爱", "我是", "我想", "我觉得", "不喜欢", "讨厌",
            "偏好", "习惯", "常用", "擅长", "想要", "希望", "一般",
            "通常", "主要", "专业", "工作", "职业", "风格", "细腻",
            "深刻", "温柔", "犀利", "关注", "感兴趣", "最近在看",
        ]
        if not any(k in user_input for k in hint_keywords):
            return None

        try:
            response = self.client.chat.completions.create(
                model=get_llm_model(),
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一位信息提取助手。从用户的话中提取值得长期记忆的个人偏好、"
                            "身份特征或创作需求。只提取事实性信息，不要推测。"
                            "如果没有值得记忆的信息，只回复'无'。格式要求：标签 = 内容"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"用户说的话：\"{user_input}\"\n\n"
                            "请提取值得记忆的信息。注意：\n"
                            "- 只提取用户明确表达或强烈暗示的偏好/特征\n"
                            "- 标签要简洁（2-6字），如'偏好作家''职业身份''风格偏好'\n"
                            "- 内容要具体，不要抽象\n"
                            "- 如果没有，回复\"无\""
                        ),
                    },
                ],
            )
            result = response.choices[0].message.content.strip()
            if result in ("无", "", None) or "=" not in result:
                return None
            key, value = result.split("=", 1)
            key = key.strip().strip("\"'")
            value = value.strip().strip("\"'")
            if not key or not value or len(value) < 2:
                return None
            return key, value
        except Exception:
            return None

    def _summarize_old_history(self):
        """当历史消息过长时，对早期消息做摘要并截断"""
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        dialog_msgs = [m for m in self.messages if m["role"] != "system"]

        MAX_HISTORY_MESSAGES = 20
        SUMMARY_BATCH = 10

        if len(dialog_msgs) <= MAX_HISTORY_MESSAGES:
            return

        to_summarize = dialog_msgs[:SUMMARY_BATCH]

        summary_prompt = "请对以下对话进行简明摘要，保留核心信息：\n\n"
        for m in to_summarize:
            role = "用户" if m["role"] == "user" else "小八"
            content = m.get("content", "")[:200]
            summary_prompt += f"{role}：{content}\n"
        summary_prompt += "\n请用1-2句话概括这段对话的核心内容和用户需求。"

        try:
            response = self.client.chat.completions.create(
                model=get_llm_model(),
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "你是一位对话摘要助手。用简洁的语言概括对话核心。"},
                    {"role": "user", "content": summary_prompt},
                ],
            )
            summary = response.choices[0].message.content.strip()

            # 更新 memory
            existing = self.memory.get("对话概要", "")
            self.memory["对话概要"] = (existing + "；" + summary) if existing else summary

            # 截断历史：保留后面的
            remaining = dialog_msgs[SUMMARY_BATCH:]
            self.messages = system_msgs + remaining

            # 同步更新 system message 内容（因为 memory 变了）
            new_system = SYSTEM_PROMPT
            if not self.supports_tools:
                new_system += _build_tools_prompt()
            new_system += _format_memory(self.memory)
            self.messages[0] = {"role": "system", "content": new_system}

        except Exception:
            # 摘要失败时直接截断，保留最近的
            remaining = dialog_msgs[-MAX_HISTORY_MESSAGES:]
            self.messages = system_msgs + remaining

    def chat(self, user_input: str) -> str:
        """处理用户输入，支持记忆提取和多轮工具调用"""
        # 检查是否是记忆设置指令
        memory_match = re.match(r'^记住[：:](.+)$', user_input.strip())
        if memory_match:
            content = memory_match.group(1).strip()
            # 尝试解析 "key = value" 格式
            kv_match = re.match(r'(.+?)[=＝](.+)', content)
            if kv_match:
                key = kv_match.group(1).strip()
                value = kv_match.group(2).strip()
                self.set_memory(key, value)
                return f"记住了：{key} → {value}\n\n当前记忆：\n" + "\n".join(f"- {k}：{v}" for k, v in self.memory.items())
            else:
                # 作为自由文本存储
                self.set_memory("偏好", content)
                return f"记住了：{content}\n\n当前记忆：\n" + "\n".join(f"- {k}：{v}" for k, v in self.memory.items())

        self.messages.append({"role": "user", "content": user_input})

        max_iterations = 5
        reply = None
        for _ in range(max_iterations):
            if self.supports_tools:
                reply = self._chat_with_tools()
            else:
                reply = self._chat_without_tools()

            if reply is not None:
                break

        if reply is None:
            # fallback
            response = self.client.chat.completions.create(
                model=get_llm_model(),
                temperature=LLM_TEMPERATURE,
                messages=self.messages,
            )
            self.messages.append(response.choices[0].message)
            reply = response.choices[0].message.content

        # 自动记忆提取
        auto_memory = self._extract_memory(user_input)
        if auto_memory:
            key, value = auto_memory
            self.set_memory(key, value)
            reply += f"\n\n💡 我注意到：{key} → {value}，已经记下了。"

        # 历史截断与摘要
        self._summarize_old_history()

        return reply

    def _chat_with_tools(self):
        """OpenAI 官方 API，支持 function calling"""
        response = self.client.chat.completions.create(
            model=get_llm_model(),
            temperature=LLM_TEMPERATURE,
            messages=self.messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            self.messages.append(msg)
            return msg.content

        self.messages.append(msg)

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            if func_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[func_name](**func_args)
            else:
                result = f"未知工具: {func_name}"

            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )
        return None

    def _chat_without_tools(self):
        """DeepSeek 等不支持 function calling 的 API"""
        response = self.client.chat.completions.create(
            model=get_llm_model(),
            temperature=LLM_TEMPERATURE,
            messages=self.messages,
        )

        msg = response.choices[0].message
        content = msg.content or ""

        # 尝试解析工具调用
        func_name, func_args = _parse_tool_call(content)

        if func_name and func_name in TOOL_FUNCTIONS:
            # 去掉 JSON 指令后的内容作为展示文本
            clean_content = re.sub(r'\n?\{[^}]*"tool"[^}]*\}\n?', '', content).strip()
            if clean_content:
                self.messages.append({"role": "assistant", "content": clean_content})

            # 执行工具
            result = TOOL_FUNCTIONS[func_name](**func_args)

            # 把工具结果作为用户消息追加
            self.messages.append(
                {"role": "user", "content": f"【工具结果】\n{result}\n\n请基于以上结果继续回复。"}
            )
            return None

        self.messages.append(msg)
        return content

    def reset(self):
        """重置对话历史，但保留记忆"""
        self._build_system_message()

    def get_history(self) -> list:
        return self.messages
