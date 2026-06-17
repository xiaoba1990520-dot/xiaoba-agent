"""「小八」- 文字型内容创作者智能体 - Streamlit PWA 对话界面"""

import json
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from agent import BookBloggerAgent

st.set_page_config(page_title="小八", page_icon="📖", layout="wide")

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

if "works" not in st.session_state:
    st.session_state.works = []

if "copy_notification" not in st.session_state:
    st.session_state.copy_notification = None


def copy_button(text: str, key: str):
    """注入 HTML/JS 实现一键复制"""
    safe_text = text.replace("`", "\\`").replace("$", "\\$")
    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{safe_text}`);"
                id="btn-{key}"
                style="background:none;border:none;cursor:pointer;font-size:12px;color:#888;padding:0;margin-right:8px;">
            📋 复制
        </button>
        <script>
            document.getElementById('btn-{key}').addEventListener('click', function() {{
                this.innerText = '✓ 已复制';
                this.style.color = 'green';
                setTimeout(() => {{ this.innerText = '📋 复制'; this.style.color = '#888'; }}, 2000);
            }});
        </script>
        """,
        height=25,
        key=key,
    )

# 侧边栏
with st.sidebar:
    st.title("📖 小八")
    st.caption("文字型内容创作者")
    st.divider()

    st.subheader("人格标签")
    st.markdown("`文学` `心理学` `哲学` `文字创作` `情感细腻` `洞察`")

    st.divider()
    st.subheader("技能")
    st.markdown("- **写稿**：生成脚本/长文/金句")
    st.markdown("- **润稿**：内容质量优化")
    st.markdown("- **校对**：错别字、语病、流畅度")
    st.markdown("- **发布**：标题、标签、封面文案")
    st.markdown("- **找热点**：选题角度")

    # 当前记忆
    st.divider()
    st.subheader("🧠 小八记得")
    memory = st.session_state.agent.get_memory()
    if memory:
        for k, v in memory.items():
            st.caption(f"**{k}**：{v}")
    else:
        st.caption("还没有记住什么…")
    st.caption("💡 你可以说：记住：偏好作家 = 余华")

    # 作品库
    st.divider()
    st.subheader("💾 作品库")
    if st.session_state.works:
        for idx, work in enumerate(st.session_state.works):
            with st.expander(f"{work['time']} · {work['type']}"):
                st.markdown(work["content"][:200] + "…" if len(work["content"]) > 200 else work["content"])
                if st.button("🗑️ 删除", key=f"del_work_{idx}"):
                    st.session_state.works.pop(idx)
                    st.rerun()
    else:
        st.caption("还没有保存的作品")

    # 数据导出/导入
    st.divider()
    st.subheader("📦 数据管理")

    export_data = {
        "works": st.session_state.works,
        "chat_history": st.session_state.chat_history,
        "memory": st.session_state.agent.get_memory(),
        "exported_at": datetime.now().isoformat(),
    }
    export_json = json.dumps(export_data, ensure_ascii=False, indent=2)
    st.download_button(
        label="📥 导出全部数据",
        data=export_json,
        file_name=f"小八数据_{datetime.now().strftime('%m%d_%H%M')}.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded = st.file_uploader("📤 导入数据", type=["json"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            imported = json.loads(uploaded.read().decode("utf-8"))
            if "works" in imported:
                st.session_state.works = imported["works"]
            if "chat_history" in imported:
                st.session_state.chat_history = imported["chat_history"]
                st.session_state.agent.restore_history(imported["chat_history"])
            if "memory" in imported:
                for k, v in imported["memory"].items():
                    st.session_state.agent.set_memory(k, v)
            st.toast("数据导入成功")
            st.rerun()
        except Exception as e:
            st.error(f"导入失败：{str(e)}")

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
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("✍️ 写稿", use_container_width=True):
        st.session_state.quick_input = "帮我写一篇稿子"
with col2:
    if st.button("💎 润稿", use_container_width=True):
        st.session_state.quick_input = "帮我润色一篇文案"
with col3:
    if st.button("✓ 校对", use_container_width=True):
        st.session_state.quick_input = "帮我校对一段文字"
with col4:
    if st.button("📤 发布", use_container_width=True):
        st.session_state.quick_input = "帮我生成发布配套"
with col5:
    if st.button("🔥 热点", use_container_width=True):
        st.session_state.quick_input = "帮我找找热点"

st.divider()

# 显示对话历史
for i, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            # 操作按钮
            cols = st.columns([1, 1, 8])
            with cols[0]:
                copy_button(msg["content"], key=f"copy_{i}")
            with cols[1]:
                if st.button("💾", key=f"save_{i}", help="保存到作品库"):
                    st.session_state.works.append({
                        "content": msg["content"],
                        "type": "对话",
                        "time": datetime.now().strftime("%m-%d %H:%M"),
                    })
                    st.toast("已保存到作品库")

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
            try:
                reply = st.session_state.agent.chat(text)
            except Exception as e:
                reply = (
                    f"⚠️ 出错了：{str(e)}\n\n"
                    "可能的原因：\n"
                    "- 网络连接不稳定\n"
                    "- API 暂时不可用\n"
                    "- 请求内容过长\n\n"
                    "请稍后再试，或换一种方式提问。"
                )
        st.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # 如果模型内部发生了历史截断，同步界面上的 chat_history
    if st.session_state.agent.history_trimmed:
        st.session_state.chat_history = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in st.session_state.agent.get_history()
            if m["role"] in ("user", "assistant")
        ]
