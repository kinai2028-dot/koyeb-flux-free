"""
Flux AI 圖像生成器 Pro - 主應用入口
優化版本 - 模塊化架構
"""

import streamlit as st
from config.settings import Config
from ui.pages import MainPage
from ui.sidebar import Sidebar
from utils.logger import setup_logger
from utils.cache import CacheManager

# 設定頁面配置
st.set_page_config(
    page_title="Flux AI 圖像生成器 Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_app():
    """初始化應用程序"""
    setup_logger()
    
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = CacheManager()
    
    if 'current_images' not in st.session_state:
        st.session_state.current_images = []
    
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []

def main():
    """主應用函數"""
    initialize_app()
    
    sidebar = Sidebar()
    sidebar.render()
    
    main_page = MainPage()
    main_page.render()

if __name__ == "__main__":
    main()
