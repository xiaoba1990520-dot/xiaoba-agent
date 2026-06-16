"""读书博主智能体 - 配置管理（支持环境变量和 Streamlit Secrets）"""

import os
from dotenv import load_dotenv

load_dotenv()

# 尝试从 Streamlit secrets 读取配置
try:
    import streamlit as st
    _secrets = st.secrets
    _LLM_API_KEY = _secrets.get("llm", {}).get("api_key", "") or os.getenv("LLM_API_KEY", "")
    _LLM_BASE_URL = _secrets.get("llm", {}).get("base_url", "") or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    _LLM_MODEL = _secrets.get("llm", {}).get("model", "") or os.getenv("LLM_MODEL", "gpt-4o")
except (ImportError, FileNotFoundError):
    _LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    _LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    _LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# LLM 配置
LLM_MODEL = _LLM_MODEL
LLM_API_KEY = _LLM_API_KEY
LLM_BASE_URL = _LLM_BASE_URL
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# 智能体人格标签
PERSONA_TAGS = ["文学", "心理学", "情感细腻", "女性成长", "读书博主"]
