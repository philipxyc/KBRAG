import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

# Ollama 配置
OLLAMA_BASE_URL = 'http://192.168.124.18:11434'  # Ollama 的 IP 和端口
EMBEDDING_MODEL_NAME = 'mxbai-embed-large'  # 嵌入模型名称

# 初始化 OllamaEmbeddings
EMBEDDINGS = OllamaEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    base_url=OLLAMA_BASE_URL
)

# 持久化存储目录
PERSIST_DIRECTORY = 'db'

def admin_page():
    st.title('📂 管理员界面')
    st.write('在此页面上传或删除知识库文件。')

    # 创建必要的目录
    if not os.path.exists('documents'):
        os.makedirs('documents')

    # 更新上传文件类型，支持 .md 文件
    uploaded_files = st.file_uploader('选择 PDF、TXT 或 MD 文件上传', type=['pdf', 'txt', 'md'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join('documents', uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.success(f'文件 {uploaded_file.name} 上传成功！')
            # 加载并嵌入文件
            load_and_embed_file(file_path)
        st.info('所有文件已处理完毕。')

    # 显示已上传的文件
    st.subheader('已上传的文件')
    files = os.listdir('documents')
    if files:
        for file in files:
            st.write(file)
    else:
        st.write('暂无文件。')

    # 删除文件
    delete_file = st.selectbox('选择要删除的文件', [''] + files)
    if delete_file:
        if st.button('删除文件'):
            file_path = os.path.join('documents', delete_file)
            os.remove(file_path)
            # 从向量数据库中删除相应的嵌入
            delete_embeddings(delete_file)
            st.success(f'文件 {delete_file} 已删除！')

def load_and_embed_file(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_path.endswith('.md'):
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        st.error('仅支持 PDF、TXT 和 MD 文件')
        return

    documents = loader.load()
    # 在每个文档的元数据中添加文件名
    for doc in documents:
        doc.metadata['source'] = os.path.basename(file_path)
    # 将文档添加到向量数据库
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=EMBEDDINGS
    )
    vectorstore.add_documents(documents)
    # 保存到持久化存储
    vectorstore.persist()

def delete_embeddings(delete_file):
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=EMBEDDINGS
    )
    # 获取所有文档的元数据和 IDs
    docs = vectorstore._collection.get(include=['metadatas'])
    ids_to_delete = []
    for doc_id, metadata in zip(docs['ids'], docs['metadatas']):
        if metadata.get('source') == delete_file:
            ids_to_delete.append(doc_id)
    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)
        # 保存更改
        vectorstore.persist()
