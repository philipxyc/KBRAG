import streamlit as st
from admin_page import admin_page
from user_page import user_page

def main():
    st.set_page_config(page_title='RAG 平台', layout='wide')
    st.sidebar.title('导航')
    page = st.sidebar.radio('选择页面', ('用户界面', '管理员界面'))

    if page == '用户界面':
        user_page()
    elif page == '管理员界面':
        admin_page()

if __name__ == '__main__':
    main()
