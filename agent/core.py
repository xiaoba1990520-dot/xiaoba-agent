"""智能体核心调度 - 基于 OpenAI function calling 的 Agent 循环"""

import json
from openai import OpenAI

from agent.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_TEMPERATURE
from agent.persona import SYSTEM_PROMPT
from agent.skills import TOOL_SCHEMAS, TOOL_FUNCTIONS


class BookBloggerAgent:
    """读书博主智能体「小八」"""

    def __init__(self):
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    def chat(self, user_input: str) -> str:
        """
        与智能体对话，支持多轮 function calling

        Args:
            user_input: 用户输入

        Returns:
            智能体的回复
        """
        self.messages.append({"role": "user", "content": user_input})

        # Agent 循环：不断调用 LLM 直到不再触发 function calling
        max_iterations = 5
        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )

            msg = response.choices[0].message

            # 如果没有工具调用，直接返回文本回复
            if not msg.tool_calls:
                self.messages.append(msg)
                return msg.content

            # 有工具调用，先记录 assistant 消息
            self.messages.append(msg)

            # 逐个执行工具调用
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                # 执行工具
                if func_name in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[func_name](**func_args)
                else:
                    result = f"未知工具: {func_name}"

                # 将工具结果追加到消息列表
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

        # 超过最大迭代次数，请求 LLM 给出总结回复
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=self.messages,
        )
        self.messages.append(response.choices[0].message)
        return response.choices[0].message.content

    def reset(self):
        """重置对话历史"""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    def get_history(self) -> list:
        """获取对话历史"""
        return self.messages
