import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
import datetime
import base64
from typing import Dict, List, Optional, Tuple
import time
import random
import asyncio
import threading
import json

# ==============================================================================
# 1. 應用程式全域設定
# ==============================================================================

# 設定頁面配置
st.set_page_config(
    page_title="Flux AI 圖像生成器 Pro",
    page_icon="🎨",
    layout="wide"
)

# API 提供商配置
API_PROVIDERS = {
    "OpenAI Compatible": {
        "name": "OpenAI Compatible API",
        "base_url_default": "https://api.openai.com/v1",
        "key_prefix": "sk-",
        "description": "OpenAI 官方或兼容的 API 服務",
        "icon": "🤖"
    },
    "Navy": {
        "name": "Navy API",
        "base_url_default": "https://api.navy/v1",
        "key_prefix": "sk-",
        "description": "Navy 提供的 AI 圖像生成服務",
        "icon": "⚓"
    },
    "Custom": {
        "name": "自定義 API",
        "base_url_default": "",
        "key_prefix": "",
        "description": "自定義的 API 端點",
        "icon": "🔧"
    }
}

# Flux 模型配置
FLUX_MODELS = {
    "flux.1-schnell": {
        "name": "FLUX.1 Schnell",
        "description": "最快的生成速度，開源模型",
        "icon": "⚡",
        "type": "快速生成",
        "test_prompt": "A simple cat sitting on a table",
        "expected_size": "1024x1024",
        "priority": 1
    },
    "flux.1-krea-dev": {
        "name": "FLUX.1 Krea Dev",
        "description": "創意開發版本，適合實驗性生成",
        "icon": "🎨",
        "type": "創意開發",
        "test_prompt": "Creative digital art of a futuristic city",
        "expected_size": "1024x1024",
        "priority": 2
    },
    "flux.1.1-pro": {
        "name": "FLUX.1.1 Pro",
        "description": "改進的旗艦模型，最佳品質",
        "icon": "👑",
        "type": "旗艦版本",
        "test_prompt": "Professional portrait of a person in business attire",
        "expected_size": "1024x1024",
        "priority": 3
    }
}


# ==============================================================================
# 2. 核心 API 與模型處理函式
# ==============================================================================

def validate_api_key(api_key: str, base_url: str) -> Tuple[bool, str]:
    """驗證 API 密鑰是否有效"""
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        client.models.list()
        return True, "API 密鑰驗證成功"
    except Exception as e:
        return False, f"API 驗證失敗: {str(e)[:150]}"

def get_available_models(client: OpenAI) -> Tuple[bool, List[str]]:
    """獲取可用的模型列表"""
    try:
        response = client.models.list()
        return True, [model.id for model in response.data]
    except Exception as e:
        return False, [str(e)]

def test_model_availability(client: OpenAI, model_name: str) -> Dict:
    """測試特定模型的可用性"""
    test_prompt = FLUX_MODELS.get(model_name, {}).get('test_prompt', 'test')
    result = {'model': model_name, 'available': False, 'error': None}
    try:
        client.images.generate(model=model_name, prompt=test_prompt, n=1, size="1024x1024")
        result['available'] = True
    except Exception as e:
        result['error'] = str(e)
    return result

def generate_images_with_retry(client, **params) -> Tuple[bool, any]:
    """帶重試機制的圖像生成"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return True, client.images.generate(**params)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return False, str(e)

def init_api_client():
    """初始化 API 客戶端"""
    api_key = st.session_state.api_config.get('api_key')
    base_url = st.session_state.api_config.get('base_url')
    if api_key and base_url:
        try:
            return OpenAI(api_key=api_key, base_url=base_url)
        except Exception:
            return None
    return None

# ==============================================================================
# 3. Session State 與 UI 函式
# ==============================================================================

def init_session_state():
    """初始化會話狀態"""
    if 'api_config' not in st.session_state:
        st.session_state.api_config = {'provider': 'Navy', 'api_key': '', 'base_url': 'https://api.navy/v1', 'validated': False}
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    if 'model_test_results' not in st.session_state:
        st.session_state.model_test_results = {}

def show_api_settings():
    """顯示 API 設置界面"""
    st.subheader("🔑 API 設置")
    provider = st.selectbox("選擇 API 提供商", options=list(API_PROVIDERS.keys()), format_func=lambda x: API_PROVIDERS[x]['name'])
    api_key = st.text_input("API 密鑰", type="password", value=st.session_state.api_config.get('api_key', ''))
    base_url = st.text_input("API 端點 URL", value=API_PROVIDERS[provider]['base_url_default'])
    
    if st.button("💾 保存並測試"):
        st.session_state.api_config = {'provider': provider, 'api_key': api_key, 'base_url': base_url}
        with st.spinner("正在測試 API 連接..."):
            is_valid, msg = validate_api_key(api_key, base_url)
            st.session_state.api_config['validated'] = is_valid
            if is_valid:
                st.success(msg)
            else:
                st.error(msg)

# ==============================================================================
# 4. 主應用程式執行流程
# ==============================================================================

def main():
    init_session_state()
    client = init_api_client()

    with st.sidebar:
        show_api_settings()

    st.title("🎨 Flux AI 圖像生成器")

    if not client or not st.session_state.api_config.get('validated'):
        st.warning("請在側邊欄配置並驗證您的 API 密鑰。")
        st.stop()

    tab1, tab2 = st.tabs(["🚀 圖像生成", "🧪 模型測試"])

    with tab1:
        st.subheader("✏️ 輸入提示詞")
        model = st.selectbox("選擇模型", options=list(FLUX_MODELS.keys()), format_func=lambda x: FLUX_MODELS[x]['name'])
        prompt = st.text_area("描述你想要的圖像", height=100)
        
        if st.button("生成圖像", type="primary", disabled=not prompt):
            with st.spinner(f"正在使用 {model} 生成圖像..."):
                params = {"model": model, "prompt": prompt, "n": 1, "size": "1024x1024"}
                success, result = generate_images_with_retry(client, **params)
                
                if success:
                    st.success("圖像生成成功！")
                    image_url = result.data[0].url
                    st.image(image_url, caption=prompt)
                    # 添加到歷史記錄
                    st.session_state.generation_history.insert(0, {'prompt': prompt, 'model': model, 'url': image_url})
                else:
                    st.error(f"生成失敗: {result}")

    with tab2:
        st.subheader("🧪 模型可用性測試")
        if st.button("開始測試"):
            with st.spinner("正在測試所有模型..."):
                results = {}
                for model_name in FLUX_MODELS.keys():
                    results[model_name] = test_model_availability(client, model_name)
                st.session_state.model_test_results = results
            st.success("測試完成！")
        
        if st.session_state.model_test_results:
            st.json(st.session_state.model_test_results)

if __name__ == "__main__":
    main()
