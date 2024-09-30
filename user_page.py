import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings

# Ollama 配置
OLLAMA_BASE_URL = 'http://192.168.124.18:11434'  # Ollama 的 IP 和端口
LLM_MODEL_NAME = 'qwen2'  # LLM 模型名称
EMBEDDING_MODEL_NAME = 'mxbai-embed-large'  # 嵌入模型名称

# 初始化 OllamaEmbeddings
EMBEDDINGS = OllamaEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    base_url=OLLAMA_BASE_URL
)

# 持久化存储目录
PERSIST_DIRECTORY = 'db'

def user_page():
    st.title('💬 用户界面')
    st.write('与 AI 进行聊天。')

    # 检查向量数据库是否为空
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=EMBEDDINGS
    )

    if vectorstore._collection.count() == 0:
        st.warning('知识库为空，请先在管理员界面上传文件。')
        return

    # 初始化检索器，增加检索参数
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})  # 调整 k 值，增加返回的文档数量

    # 初始化语言模型（Ollama）
    llm = Ollama(
        model=LLM_MODEL_NAME,
        base_url=OLLAMA_BASE_URL
    )

    # 创建 ConversationalRetrievalChain，并设置 return_source_documents=True
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # 聊天历史记录
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # 用户输入
    query = st.text_input('请输入您的问题：')

    if query:
        with st.spinner('AI 正在思考...'):
            result = qa({'question': query, 'chat_history': st.session_state['chat_history']})
            st.session_state['chat_history'].append((query, result['answer']))

            st.markdown(f"**用户**：{query}")
            st.markdown(f"**AI**：{result['answer']}")

            # 显示引用的源文档内容
            if 'source_documents' in result:
                st.subheader('🔍 引用的源文档内容：')
                for idx, doc in enumerate(result['source_documents']):
                    st.markdown(f"**文档 {idx + 1}**")
                    st.markdown(f"*内容：* {doc.page_content}")
                    if 'source' in doc.metadata:
                        st.markdown(f"*来源：* {doc.metadata['source']}")
                    st.markdown("---")

    # 显示聊天记录
    if st.session_state['chat_history']:
        st.subheader('📝 聊天记录')
        for i, (user_q, ai_a) in enumerate(st.session_state['chat_history']):
            st.markdown(f"**用户**：{user_q}")
            st.markdown(f"**AI**：{ai_a}")
