"""技能模块"""

from agent.skills.writer import write_script, WRITER_TOOL_SCHEMA
from agent.skills.reviewer import review_script, REVIEWER_TOOL_SCHEMA
from agent.skills.trending import find_trending, TRENDING_TOOL_SCHEMA
from agent.skills.proofreader import proofread, PROOFREADER_TOOL_SCHEMA
from agent.skills.meta import generate_meta, META_TOOL_SCHEMA

# 所有工具 schema 注册表
TOOL_SCHEMAS = [
    WRITER_TOOL_SCHEMA,
    REVIEWER_TOOL_SCHEMA,
    TRENDING_TOOL_SCHEMA,
    PROOFREADER_TOOL_SCHEMA,
    META_TOOL_SCHEMA,
]

# 工具名 -> 执行函数 映射
TOOL_FUNCTIONS = {
    "write_script": write_script,
    "review_script": review_script,
    "find_trending": find_trending,
    "proofread": proofread,
    "generate_meta": generate_meta,
}
