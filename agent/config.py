"""读书博主智能体 - 配置管理（支持环境变量和 Streamlit Secrets）"""

import os
from dotenv import load_dotenv

load_dotenv()

def _get(key: str, section: str = "llm", default: str = "") -> str:
    """优先从 Streamlit Secrets 读取，其次从环境变量读取"""
    try:
        import streamlit as st
        val = st.secrets.get(section, {}).get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key.upper(), default)


def get_llm_api_key() -> str:
    return _get("api_key", default=os.getenv("LLM_API_KEY", ""))

def get_llm_base_url() -> str:
    return _get("base_url", default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))

def get_llm_model() -> str:
    return _get("model", default=os.getenv("LLM_MODEL", "gpt-4o"))


# 兼容旧代码的静态变量（本地运行用）
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# 智能体人格标签
PERSONA_TAGS = ["文学", "心理学", "情感细腻", "女性成长", "读书博主"]
