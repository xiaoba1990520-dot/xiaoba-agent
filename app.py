"""读书博主智能体「小八」- Streamlit PWA 对话界面"""

import streamlit as st
import streamlit.components.v1 as components
from agent import BookBloggerAgent

st.set_page_config(page_title="小八 - 读书博主智能体", page_icon="📖", layout="wide")

# PWA meta tags for mobile "Add to Home Screen"
components.html("""
<link rel="manifest" href="/static/manifest.json">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="小八">
<link rel="apple-touch-icon" href="/static/icon-192.png">
<meta name="theme-color" content="#FF9B85">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<style>
    /* Mobile optimizations */
    @media (max-width: 768px) {
        .stApp { padding: 0.5rem; }
        .main .block-container { padding: 1rem 0.5rem; }
        h1 { font-size: 1.5rem !important; }
    }
</style>
""", height=0)

# 初始化智能体（会话级别）
if "agent" not in st.session_state:
    st.session_state.agent = BookBloggerAgent()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 侧边栏
with st.sidebar:
    st.title("📖 小八")
    st.caption("短视频读书博主智能体")
    st.divider()

    st.subheader("人格标签")
    st.markdown("`文学` `心理学` `情感细腻` `女性成长` `读书博主`")

    st.divider()
    st.subheader("技能")
    st.markdown("- **写稿**：生成短视频口播文案")
    st.markdown("- **审稿**：多维度审阅修改建议")
    st.markdown("- **找热点**：热点话题创作角度")

    st.divider()
    if st.button("🔄 重置对话", use_container_width=True):
        st.session_state.agent.reset()
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("提示：直接和小八对话即可，她会根据你的需求自动调用技能。")

# 主界面
st.title("和小八聊聊")

# 快捷操作按钮
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("✍️ 写稿", use_container_width=True):
        st.session_state.quick_input = "帮我写一篇稿子"
with col2:
    if st.button("🔍 审稿", use_container_width=True):
        st.session_state.quick_input = "帮我审阅一篇文案"
with col3:
    if st.button("🔥 找热点", use_container_width=True):
        st.session_state.quick_input = "帮我找找最近可以写的内容角度"

st.divider()

# 显示对话历史
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# 输入区域
def get_default_input():
    return st.session_state.pop("quick_input", "")

user_input = st.chat_input("和小八说点什么...", key="chat_input")

# 处理快捷按钮输入
if "quick_input" not in st.session_state:
    st.session_state.quick_input = ""
quick_msg = get_default_input()

if user_input or quick_msg:
    text = user_input or quick_msg

    # 显示用户消息
    st.session_state.chat_history.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)

    # 获取智能体回复
    with st.chat_message("assistant"):
        with st.spinner("小八正在思考..."):
            reply = st.session_state.agent.chat(text)
        st.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
