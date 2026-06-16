"""智能体核心调度 - 基于 OpenAI function calling 的 Agent 循环"""

import json
from openai import OpenAI

from agent.config import LLM_TEMPERATURE, get_llm_api_key, get_llm_base_url, get_llm_model
from agent.persona import SYSTEM_PROMPT
from agent.skills import TOOL_SCHEMAS, TOOL_FUNCTIONS


class BookBloggerAgent:
    """读书博主智能体「小八」"""

    def __init__(self):
        # 运行时读取，确保拿到 Streamlit Secrets
        self.client = OpenAI(api_key=get_llm_api_key(), base_url=get_llm_base_url())
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        max_iterations = 5
        for _ in range(max_iterations):
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

        response = self.client.chat.completions.create(
            model=get_llm_model(),
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
        return self.messages
