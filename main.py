import streamlit as st
from langchain_deepseek import ChatDeepSeek
import os
from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache  # 从 community 导入
from langchain_core.globals import set_llm_cache
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy import create_engine
import sqlite3
from datetime import datetime

# 3. 使用 SQLite 持久化存储（替代内存中的 store）
engine = create_engine("sqlite:///chat_history.db")
def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=engine  # 数据保存到文件
    )

def init_database():
    """初始化数据库，确保表和索引存在"""
    db_path = "chat_history.db"
    
    # 确保数据库文件存在并创建表
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建 message_store 表（如果不存在）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS message_store (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message TEXT,
            type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 创建索引以提高查询性能
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_id 
        ON message_store (session_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON message_store (timestamp DESC)
    """)
    
    conn.commit()
    conn.close()
    print("✅ 数据库初始化完成")

# ============ 2. UI 会话管理 ============

def get_all_sessions():
    """从数据库获取所有会话列表"""
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    # 查询所有不同的 session_id 及其最后更新时间
    cursor.execute("""
        SELECT session_id, MAX(timestamp) as last_time 
        FROM message_store 
        GROUP BY session_id 
        ORDER BY last_time DESC
    """)
    sessions = cursor.fetchall()
    conn.close()
    return [s[0] for s in sessions]

def delete_session(session_id: str):
    """删除指定会话的所有记录"""
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM message_store WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

# 从数据库加载当前会话的历史消息
def load_messages_from_db(session_id: str):
    """从 SQLite 加载当前会话的所有消息"""
    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    return messages

def main():
    load_dotenv()
    init_database()

    # 1. 设置缓存
    set_llm_cache(SQLiteCache(database_path="langchain_cache.db"))

    st.set_page_config(
        page_title="🤖 DeepSeek 聊天助手", 
        page_icon="💬",
        layout="wide"
    )

    # 左侧边栏 - 会话管理
    with st.sidebar:
        st.title("📋 会话管理")
        
        # 创建新会话按钮
        if st.button("➕ 新建会话", use_container_width=True):
            # 生成新的 session_id
            import uuid
            new_session_id = f"user_{uuid.uuid4().hex[:8]}"
            st.session_state.session_id = new_session_id
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # 显示所有历史会话
        st.subheader("📚 历史会话")
        sessions = get_all_sessions()
        
        if sessions:
            for sid in sessions:
                col1, col2 = st.columns([4, 1])
                with col1:
                    # 显示会话 ID 的简短版本
                    display_name = sid[:12] + "..." if len(sid) > 12 else sid
                    if st.button(display_name, key=f"load_{sid}", use_container_width=True):
                        st.session_state.session_id = sid
                        st.session_state.messages = []  # 清空前端缓存，重新从数据库加载
                        st.rerun()
                with col2:
                    # 删除按钮
                    if st.button("🗑️", key=f"del_{sid}"):
                        delete_session(sid)
                        st.rerun()
        else:
            st.info("暂无历史会话")

    # 主界面
    st.title("💬 DeepSeek 聊天助手")

    # 初始化 session_state
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = f"user_{uuid.uuid4().hex[:8]}"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 如果前端消息为空，从数据库加载
    if not st.session_state.messages:
        st.session_state.messages = load_messages_from_db(st.session_state.session_id)

    # 显示当前会话 ID（便于调试）
    st.caption(f"会话 ID: {st.session_state.session_id}")

    # 显示消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    if prompt := st.chat_input("输入你的问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 调用 LangChain（会自动保存到 SQLite）
        config = {"configurable": {"session_id": st.session_state.session_id}}

            # 初始化模型
        llm = ChatDeepSeek(
            model="deepseek-chat",  # 日常对话用 deepseek-chat，复杂推理用 deepseek-reasoner [citation:1][citation:9]
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # 4. 包装模型
        with_history = RunnableWithMessageHistory(
            llm,
            get_session_history,
        )

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    response = with_history.invoke(
                        [HumanMessage(content=prompt)],
                        config=config
                    )
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                except Exception as e:
                    st.error(f"出错了: {str(e)}")

if __name__ == '__main__':
    main()