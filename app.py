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
import json
import sqlite3
import uuid
import zipfile
import psutil
import os

# 檢查 Streamlit 版本並定義重新運行函數
def rerun_app():
    """兼容不同 Streamlit 版本的重新運行函數"""
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    else:
        # 作為最後的回退
        st.stop()

# 設定頁面配置
st.set_page_config(
    page_title="Flux AI 圖像生成器 Pro - Koyeb Edition",
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
    "Hugging Face": {
        "name": "Hugging Face Inference",
        "base_url_default": "https://api-inference.huggingface.co",
        "key_prefix": "hf_",
        "description": "Hugging Face Inference API",
        "icon": "🤗"
    },
    "Custom": {
        "name": "自定義 API",
        "base_url_default": "",
        "key_prefix": "",
        "description": "自定義的 API 端點",
        "icon": "🔧"
    }
}

# Flux 模型配置（增強版）
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
    "flux.1-dev": {
        "name": "FLUX.1 Dev", 
        "description": "開發版本，平衡速度與質量",
        "icon": "🔧",
        "type": "開發版本",
        "test_prompt": "A beautiful landscape with mountains",
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
    },
    "flux.1-kontext-pro": {
        "name": "FLUX.1 Kontext Pro",
        "description": "支持圖像編輯和上下文理解",
        "icon": "🎯",
        "type": "編輯專用",
        "test_prompt": "Abstract geometric shapes in vibrant colors",
        "expected_size": "1024x1024",
        "priority": 4
    }
}

# 數據庫初始化
def init_database():
    """初始化 SQLite 數據庫"""
    conn = sqlite3.connect('flux_ai_pro.db')
    cursor = conn.cursor()
    
    # 創建 API 配置表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_configs (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            api_key TEXT NOT NULL,
            base_url TEXT NOT NULL,
            validated BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 創建模型測試結果表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_test_results (
            id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            available BOOLEAN,
            response_time REAL,
            error_message TEXT,
            test_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 創建生成歷史表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generation_history (
            id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            model_name TEXT,
            api_provider TEXT,
            image_urls TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 創建收藏表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS favorites (
            id TEXT PRIMARY KEY,
            image_url TEXT NOT NULL,
            prompt TEXT,
            model_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def validate_api_key(api_key: str, base_url: str, provider: str) -> Tuple[bool, str]:
    """驗證 API 密鑰是否有效"""
    try:
        if provider == "Hugging Face":
            headers = {"Authorization": f"Bearer {api_key}"}
            test_url = f"{base_url}/models/black-forest-labs/FLUX.1-schnell"
            response = requests.get(test_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return True, "Hugging Face API 密鑰驗證成功"
            else:
                return False, f"HTTP {response.status_code}: 驗證失敗"
        else:
            test_client = OpenAI(api_key=api_key, base_url=base_url)
            response = test_client.models.list()
            return True, "API 密鑰驗證成功"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return False, "API 密鑰無效或已過期"
        elif "403" in error_msg or "Forbidden" in error_msg:
            return False, "API 密鑰沒有足夠權限"
        elif "404" in error_msg:
            return False, "API 端點不存在或不正確"
        elif "timeout" in error_msg.lower():
            return False, "API 連接超時"
        else:
            return False, f"API 驗證失敗: {error_msg[:100]}"

def get_available_models(client: OpenAI) -> Tuple[bool, List[str]]:
    """獲取可用的模型列表"""
    try:
        response = client.models.list()
        model_ids = [model.id for model in response.data]
        return True, model_ids
    except Exception as e:
        return False, [str(e)]

def test_model_availability(client, model_name: str, provider: str, api_key: str, base_url: str, test_prompt: str = None) -> Dict:
    """測試特定模型的可用性"""
    if test_prompt is None:
        test_prompt = FLUX_MODELS.get(model_name, {}).get('test_prompt', 'A simple test image')
    
    test_result = {
        'model': model_name,
        'available': False,
        'response_time': 0,
        'error': None,
        'details': {}
    }
    
    try:
        start_time = time.time()
        
        if provider == "Hugging Face":
            # Hugging Face API 調用
            headers = {"Authorization": f"Bearer {api_key}"}
            data = {"inputs": test_prompt}
            response = requests.post(
                f"{base_url}/models/black-forest-labs/{model_name}",
                headers=headers,
                json=data,
                timeout=30
            )
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                test_result.update({
                    'available': True,
                    'response_time': response_time,
                    'details': {
                        'status_code': response.status_code,
                        'test_prompt': test_prompt
                    }
                })
            else:
                test_result.update({
                    'available': False,
                    'error': f"HTTP {response.status_code}",
                    'details': {'status_code': response.status_code}
                })
        else:
            # OpenAI Compatible API 調用
            response = client.images.generate(
                model=model_name,
                prompt=test_prompt,
                n=1,
                size="1024x1024"
            )
            end_time = time.time()
            response_time = end_time - start_time
            
            test_result.update({
                'available': True,
                'response_time': response_time,
                'details': {
                    'image_count': len(response.data),
                    'test_prompt': test_prompt,
                    'image_url': response.data[0].url if response.data else None
                }
            })
            
    except Exception as e:
        error_msg = str(e)
        test_result.update({
            'available': False,
            'error': error_msg,
            'details': {
                'error_type': 'generation_failed',
                'test_prompt': test_prompt
            }
        })
    
    return test_result

def batch_test_models(client, provider: str, api_key: str, base_url: str, models_to_test: List[str] = None) -> Dict[str, Dict]:
    """批量測試多個模型的可用性"""
    if models_to_test is None:
        models_to_test = list(FLUX_MODELS.keys())
    
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, model_name in enumerate(models_to_test):
        progress = (i + 1) / len(models_to_test)
        progress_bar.progress(progress)
        status_text.text(f"🧪 正在測試 {FLUX_MODELS.get(model_name, {}).get('name', model_name)}... ({i+1}/{len(models_to_test)})")
        
        result = test_model_availability(client, model_name, provider, api_key, base_url)
        results[model_name] = result
        
        # 避免請求過於頻繁
        time.sleep(1)
    
    progress_bar.empty()
    status_text.empty()
    return results

def show_model_status_dashboard():
    """顯示模型狀態儀表板"""
    if 'model_test_results' not in st.session_state:
        st.session_state.model_test_results = {}
    
    st.subheader("🎯 模型可用性狀態")
    
    # 控制按鈕
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        test_all_btn = st.button("🧪 測試所有模型", type="primary")
    
    with col_btn2:
        refresh_btn = st.button("🔄 刷新狀態")
    
    with col_btn3:
        clear_cache_btn = st.button("🗑️ 清除緩存")
    
    # 執行批量測試
    if test_all_btn:
        if 'api_config' in st.session_state and st.session_state.api_config.get('api_key'):
            config = st.session_state.api_config
            
            if config['provider'] == "Hugging Face":
                client = None
            else:
                client = OpenAI(
                    api_key=config['api_key'],
                    base_url=config['base_url']
                )
            
            with st.spinner("正在批量測試所有模型..."):
                st.session_state.model_test_results = batch_test_models(
                    client, config['provider'], config['api_key'], config['base_url']
                )
                st.session_state.last_test_time = datetime.datetime.now()
            st.success("✅ 批量測試完成！")
            rerun_app()
        else:
            st.error("❌ 請先配置 API 密鑰")
    
    # 刷新狀態
    if refresh_btn:
        rerun_app()
    
    # 清除緩存
    if clear_cache_btn:
        st.session_state.model_test_results = {}
        if 'last_test_time' in st.session_state:
            del st.session_state.last_test_time
        st.success("緩存已清除")
        rerun_app()
    
    # 顯示測試結果
    if st.session_state.model_test_results:
        # 顯示最後測試時間
        if 'last_test_time' in st.session_state:
            st.caption(f"最後測試時間: {st.session_state.last_test_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 統計概覽
        total_models = len(st.session_state.model_test_results)
        available_models = sum(1 for result in st.session_state.model_test_results.values() if result.get('available', False))
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("總模型數", total_models)
        with col_stat2:
            st.metric("可用模型", available_models)
        with col_stat3:
            availability_rate = (available_models / total_models * 100) if total_models > 0 else 0
            st.metric("可用率", f"{availability_rate:.1f}%")
        
        # 詳細結果表格
        st.subheader("📊 詳細測試結果")
        
        # 按可用性和優先級排序
        sorted_results = sorted(
            st.session_state.model_test_results.items(),
            key=lambda x: (
                not x[1].get('available', False),
                FLUX_MODELS.get(x[0], {}).get('priority', 999)
            )
        )
        
        for model_name, result in sorted_results:
            model_info = FLUX_MODELS.get(model_name, {})
            
            # 創建展開框
            status_icon = "✅" if result.get('available', False) else "❌"
            response_time = result.get('response_time', 0)
            time_display = f" ({response_time:.2f}s)" if response_time > 0 else ""
            
            with st.expander(
                f"{status_icon} {model_info.get('icon', '🔧')} {model_info.get('name', model_name)}{time_display}"
            ):
                col_info, col_test = st.columns([2, 1])
                
                with col_info:
                    st.markdown(f"**模型ID**: `{model_name}`")
                    st.markdown(f"**描述**: {model_info.get('description', 'N/A')}")
                    st.markdown(f"**類型**: {model_info.get('type', 'N/A')}")
                    
                    if result.get('available', False):
                        st.success("✅ 模型可用")
                        st.markdown(f"**響應時間**: {response_time:.2f} 秒")
                    else:
                        st.error("❌ 模型不可用")
                        error_msg = result.get('error', 'Unknown error')
                        st.markdown(f"**錯誤信息**: {error_msg}")
                        
                        # 根據錯誤類型提供建議
                        if "401" in error_msg or "403" in error_msg:
                            st.warning("💡 建議檢查 API 密鑰權限")
                        elif "404" in error_msg:
                            st.warning("💡 模型可能不存在或暫時不可用")
                        elif "429" in error_msg:
                            st.warning("💡 請求過於頻繁，稍後再試")
                        elif "500" in error_msg:
                            st.warning("💡 服務器錯誤，模型可能暫時離線")
                
                with col_test:
                    st.markdown("**單獨測試**")
                    custom_prompt = st.text_input(
                        "自定義測試提示詞",
                        value=model_info.get('test_prompt', 'A simple test image'),
                        key=f"test_prompt_{model_name}"
                    )
                    
                    if st.button(f"🔬 測試此模型", key=f"test_{model_name}"):
                        if 'api_config' in st.session_state and st.session_state.api_config.get('api_key'):
                            config = st.session_state.api_config
                            
                            if config['provider'] == "Hugging Face":
                                client = None
                            else:
                                client = OpenAI(
                                    api_key=config['api_key'],
                                    base_url=config['base_url']
                                )
                            
                            with st.spinner(f"正在測試 {model_name}..."):
                                test_result = test_model_availability(
                                    client, model_name, config['provider'], 
                                    config['api_key'], config['base_url'], custom_prompt
                                )
                                st.session_state.model_test_results[model_name] = test_result
                            rerun_app()
                        else:
                            st.error("請先配置 API 密鑰")
    
    else:
        st.info("🧪 點擊 '測試所有模型' 開始檢查模型可用性")

def get_recommended_models() -> List[str]:
    """基於測試結果推薦最佳模型"""
    if 'model_test_results' not in st.session_state:
        return []
    
    # 篩選可用的模型
    available_models = [
        model_name for model_name, result in st.session_state.model_test_results.items()
        if result.get('available', False)
    ]
    
    # 按優先級和響應時間排序
    recommended = sorted(
        available_models,
        key=lambda x: (
            FLUX_MODELS.get(x, {}).get('priority', 999),
            st.session_state.model_test_results[x].get('response_time', 999)
        )
    )
    
    return recommended[:3]

def show_model_recommendations():
    """顯示模型推薦"""
    recommended = get_recommended_models()
    if recommended:
        st.subheader("⭐ 推薦模型")
        for i, model_name in enumerate(recommended):
            model_info = FLUX_MODELS.get(model_name, {})
            result = st.session_state.model_test_results.get(model_name, {})
            
            col_icon, col_info, col_metrics = st.columns([1, 3, 2])
            
            with col_icon:
                st.markdown(f"### {i+1}. {model_info.get('icon', '🔧')}")
            
            with col_info:
                st.markdown(f"**{model_info.get('name', model_name)}**")
                st.caption(model_info.get('description', 'N/A'))
            
            with col_metrics:
                response_time = result.get('response_time', 0)
                st.metric("響應時間", f"{response_time:.2f}s")
        
        # 自動選擇最佳模型
        if st.button("🚀 使用推薦的最佳模型"):
            st.session_state.recommended_model = recommended[0]
            st.success(f"已選擇: {FLUX_MODELS.get(recommended[0], {}).get('name', recommended[0])}")
            rerun_app()
    else:
        st.info("請先測試模型可用性以獲取推薦")

def init_api_client():
    """初始化 API 客戶端"""
    if 'api_config' in st.session_state and st.session_state.api_config.get('api_key'):
        config = st.session_state.api_config
        if config['provider'] == "Hugging Face":
            return None  # Hugging Face 使用直接請求
        try:
            return OpenAI(
                api_key=config['api_key'],
                base_url=config['base_url']
            )
        except Exception:
            return None
    return None

def show_api_settings():
    """顯示 API 設置界面"""
    st.subheader("🔑 API 設置")
    
    provider_options = list(API_PROVIDERS.keys())
    current_provider = st.session_state.api_config.get('provider', 'Navy')
    selected_provider = st.selectbox(
        "選擇 API 提供商",
        options=provider_options,
        index=provider_options.index(current_provider) if current_provider in provider_options else 1,
        format_func=lambda x: f"{API_PROVIDERS[x]['icon']} {API_PROVIDERS[x]['name']}"
    )
    
    provider_info = API_PROVIDERS[selected_provider]
    st.info(f"📋 {provider_info['description']}")
    
    current_key = st.session_state.api_config.get('api_key', '')
    masked_key = '*' * 20 + current_key[-8:] if len(current_key) > 8 else ''
    
    api_key_input = st.text_input(
        "API 密鑰",
        value="",
        type="password",
        placeholder=f"請輸入 {provider_info['name']} 的 API 密鑰...",
        help=f"API 密鑰通常以 '{provider_info['key_prefix']}' 開頭"
    )
    
    if current_key and not api_key_input:
        st.caption(f"🔐 當前密鑰: {masked_key}")
    
    base_url_input = st.text_input(
        "API 端點 URL",
        value=st.session_state.api_config.get('base_url', provider_info['base_url_default']),
        placeholder=provider_info['base_url_default'],
        help="API 服務的基礎 URL"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        save_btn = st.button("💾 保存設置", type="primary")
    
    with col2:
        test_btn = st.button("🧪 測試連接")
    
    with col3:
        clear_btn = st.button("🗑️ 清除設置", type="secondary")
    
    if save_btn:
        if not api_key_input and not current_key:
            st.error("❌ 請輸入 API 密鑰")
        elif not base_url_input:
            st.error("❌ 請輸入 API 端點 URL")
        else:
            final_api_key = api_key_input if api_key_input else current_key
            st.session_state.api_config = {
                'provider': selected_provider,
                'api_key': final_api_key,
                'base_url': base_url_input,
                'validated': False
            }
            st.success("✅ API 設置已保存")
            # 清除舊的模型測試結果
            st.session_state.model_test_results = {}
            rerun_app()
    
    if test_btn:
        test_api_key = api_key_input if api_key_input else current_key
        if not test_api_key:
            st.error("❌ 請先輸入 API 密鑰")
        elif not base_url_input:
            st.error("❌ 請輸入 API 端點 URL")
        else:
            with st.spinner("正在測試 API 連接..."):
                is_valid, message = validate_api_key(test_api_key, base_url_input, selected_provider)
                if is_valid:
                    st.success(f"✅ {message}")
                    st.session_state.api_config['validated'] = True
                else:
                    st.error(f"❌ {message}")
                    st.session_state.api_config['validated'] = False
    
    if clear_btn:
        st.session_state.api_config = {
            'provider': 'Navy',
            'api_key': '',
            'base_url': 'https://api.navy/v1',
            'validated': False
        }
        st.session_state.model_test_results = {}
        st.success("🗑️ API 設置已清除")
        rerun_app()
    
    # 顯示當前狀態
    if st.session_state.api_config['api_key']:
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            if st.session_state.api_config.get('validated', False):
                st.success("🟢 API 已驗證")
            else:
                st.warning("🟡 API 未驗證")
        
        with status_col2:
            st.info(f"🔧 使用: {provider_info['name']}")

def generate_images_with_retry(client, provider: str, api_key: str, base_url: str, **params) -> Tuple[bool, any]:
    """帶重試機制的圖像生成"""
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                st.info(f"🔄 嘗試重新生成 (第 {attempt + 1}/{max_retries} 次)")
            
            if provider == "Hugging Face":
                # Hugging Face API 調用
                headers = {"Authorization": f"Bearer {api_key}"}
                data = {"inputs": params.get("prompt", "")}
                
                model_name = params.get("model", "FLUX.1-schnell")
                response = requests.post(
                    f"{base_url}/models/black-forest-labs/{model_name}",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    # 模擬 OpenAI 響應格式
                    class MockResponse:
                        def __init__(self, image_data):
                            self.data = [type('obj', (object,), {'url': f"data:image/png;base64,{base64.b64encode(image_data).decode()}"})()]
                    
                    return True, MockResponse(response.content)
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
            else:
                # OpenAI Compatible API 調用
                response = client.images.generate(**params)
                return True, response
            
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                should_retry = False
                if "500" in error_msg or "502" in error_msg or "503" in error_msg:
                    should_retry = True
                elif "429" in error_msg:
                    should_retry = True
                elif "timeout" in error_msg.lower():
                    should_retry = True
                
                if should_retry:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    st.warning(f"⚠️ 第 {attempt + 1} 次嘗試失敗，{delay:.1f} 秒後重試...")
                    time.sleep(delay)
                    continue
                else:
                    return False, error_msg
            else:
                return False, error_msg
    
    return False, "所有重試均失敗"

# 初始化會話狀態
def init_session_state():
    """初始化會話狀態"""
    if 'api_config' not in st.session_state:
        st.session_state.api_config = {
            'provider': 'Navy',
            'api_key': '',
            'base_url': 'https://api.navy/v1',
            'validated': False
        }
    
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'favorite_images' not in st.session_state:
        st.session_state.favorite_images = []
    
    if 'model_test_results' not in st.session_state:
        st.session_state.model_test_results = {}
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "生成器"

def add_to_history(prompt: str, model: str, images: List[str], metadata: Dict):
    """添加生成記錄到歷史"""
    history_item = {
        "timestamp": datetime.datetime.now(),
        "prompt": prompt,
        "model": model,
        "images": images,
        "metadata": metadata,
        "id": len(st.session_state.generation_history)
    }
    st.session_state.generation_history.insert(0, history_item)
    
    # 限制歷史記錄數量
    if len(st.session_state.generation_history) > 50:
        st.session_state.generation_history = st.session_state.generation_history[:50]

def display_image_with_actions(image_url: str, image_id: str, history_item: Dict = None):
    """顯示圖像和相關操作"""
    try:
        # 處理 base64 圖像
        if image_url.startswith('data:image'):
            # 提取 base64 數據
            base64_data = image_url.split(',')[1]
            img_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(img_data))
        else:
            # 普通 URL
            img_response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(img_response.content))
            img_data = img_response.content
        
        st.image(img, use_column_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            st.download_button(
                label="📥 下載",
                data=img_buffer.getvalue(),
                file_name=f"flux_generated_{image_id}.png",
                mime="image/png",
                key=f"download_{image_id}",
                use_container_width=True
            )
        
        with col2:
            is_favorite = any(fav['id'] == image_id for fav in st.session_state.favorite_images)
            if st.button(
                "⭐ 已收藏" if is_favorite else "☆ 收藏",
                key=f"favorite_{image_id}",
                use_container_width=True
            ):
                if is_favorite:
                    st.session_state.favorite_images = [
                        fav for fav in st.session_state.favorite_images if fav['id'] != image_id
                    ]
                    st.success("已取消收藏")
                else:
                    favorite_item = {
                        "id": image_id,
                        "image_url": image_url,
                        "timestamp": datetime.datetime.now(),
                        "history_item": history_item
                    }
                    st.session_state.favorite_images.append(favorite_item)
                    st.success("已加入收藏")
                rerun_app()
        
        with col3:
            if history_item and st.button(
                "🔄 重新生成",
                key=f"regenerate_{image_id}",
                use_container_width=True
            ):
                st.session_state.regenerate_prompt = history_item['prompt']
                st.session_state.regenerate_model = history_item['model']
                st.session_state.current_page = "生成器"
                rerun_app()
    
    except Exception as e:
        st.error(f"圖像顯示錯誤: {str(e)}")

def get_system_metrics():
    """獲取系統資源信息"""
    try:
        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # 內存信息
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024**2)
        memory_total_mb = memory.total / (1024**2)
        memory_percent = memory.percent
        
        return {
            "cpu": {"percent": cpu_percent, "count": cpu_count},
            "memory": {
                "used_mb": memory_used_mb,
                "total_mb": memory_total_mb,
                "percent": memory_percent
            }
        }
    except Exception as e:
        return {"error": str(e)}

# 初始化
init_session_state()
init_database()

# 初始化 API 客戶端
client = init_api_client()
api_configured = client is not None or (st.session_state.api_config.get('provider') == "Hugging Face" and st.session_state.api_config.get('api_key'))

# 側邊欄
with st.sidebar:
    show_api_settings()
    st.markdown("---")
    
    # API 狀態顯示
    if api_configured:
        st.success("🟢 API 已配置")
        provider = st.session_state.api_config.get('provider', 'Unknown')
        st.caption(f"使用: {API_PROVIDERS.get(provider, {}).get('name', provider)}")
    else:
        st.error("🔴 API 未配置")
    
    # 系統資源監控
    st.markdown("### 📊 Koyeb 資源監控")
    metrics = get_system_metrics()
    if "error" not in metrics:
        st.metric("CPU 使用率", f"{metrics['cpu']['percent']:.1f}%")
        st.metric("內存使用", f"{metrics['memory']['used_mb']:.0f}MB")
        st.metric("內存使用率", f"{metrics['memory']['percent']:.1f}%")
    
    # 模型狀態概覽
    st.markdown("### 🎯 模型狀態")
    if st.session_state.model_test_results:
        available_count = sum(1 for result in st.session_state.model_test_results.values() if result.get('available', False))
        total_count = len(st.session_state.model_test_results)
        st.metric("可用模型", f"{available_count}/{total_count}")
        
        # 顯示推薦模型
        recommended = get_recommended_models()
        if recommended:
            st.markdown("**推薦模型:**")
            for model in recommended[:2]:
                model_name = FLUX_MODELS.get(model, {}).get('name', model)
                st.write(f"• {model_name}")
    else:
        st.info("未進行模型測試")
    
    # 使用統計
    st.markdown("### 📊 使用統計")
    total_generations = len(st.session_state.generation_history)
    total_favorites = len(st.session_state.favorite_images)
    st.metric("總生成數", total_generations)
    st.metric("收藏數量", total_favorites)

# 主標題
st.title("🎨 Flux AI 圖像生成器 Pro - Koyeb Edition")

# API 狀態警告
if not api_configured:
    st.error("⚠️ 請先配置 API 密鑰才能使用圖像生成功能")
    st.info("👈 點擊側邊欄的 'API 設置' 來配置你的密鑰")

# 頁面導航
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 圖像生成", 
    "🧪 模型測試", 
    "📚 歷史記錄", 
    "⭐ 收藏夾", 
    "💡 幫助"
])

# 圖像生成頁面
with tab1:
    if not api_configured:
        st.warning("⚠️ 請先在側邊欄配置 API 密鑰")
        st.info("配置完成後即可開始生成圖像")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 智能模型選擇
            st.subheader("🎯 智能模型選擇")
            
            # 顯示推薦模型
            recommended = get_recommended_models()
            if recommended:
                st.success("🌟 基於可用性測試的推薦模型:")
                rec_cols = st.columns(len(recommended))
                selected_model = None
                
                for i, model_name in enumerate(recommended):
                    with rec_cols[i]:
                        model_info = FLUX_MODELS.get(model_name, {})
                        result = st.session_state.model_test_results.get(model_name, {})
                        response_time = result.get('response_time', 0)
                        
                        if st.button(
                            f"{model_info.get('icon', '🔧')}\n{model_info.get('name', model_name)}\n⚡{response_time:.1f}s",
                            key=f"rec_model_{model_name}",
                            use_container_width=True,
                            help=f"{model_info.get('description', '')} (響應時間: {response_time:.2f}s)"
                        ):
                            selected_model = model_name
                
                # 如果點擊了推薦模型，更新選擇
                if selected_model:
                    st.session_state.selected_model = selected_model
            
            # 傳統模型選擇（備用）
            with st.expander("🔧 手動選擇模型"):
                model_cols = st.columns(len(FLUX_MODELS))
                for i, (model_key, model_info) in enumerate(FLUX_MODELS.items()):
                    with model_cols[i]:
                        # 顯示模型狀態
                        if model_key in st.session_state.model_test_results:
                            result = st.session_state.model_test_results[model_key]
                            if result.get('available', False):
                                status = f"✅ {result.get('response_time', 0):.1f}s"
                            else:
                                status = "❌ 不可用"
                        else:
                            status = "❓ 未測試"
                        
                        if st.button(
                            f"{model_info['icon']} {model_info['name']}\n{model_info['type']}\n{status}",
                            key=f"manual_model_{model_key}",
                            use_container_width=True,
                            help=model_info['description']
                        ):
                            st.session_state.selected_model = model_key
            
            # 最終模型選擇
            if 'selected_model' not in st.session_state:
                if recommended:
                    st.session_state.selected_model = recommended[0]
                else:
                    st.session_state.selected_model = list(FLUX_MODELS.keys())[0]
            
            final_selected_model = st.session_state.selected_model
            model_info = FLUX_MODELS[final_selected_model]
            
            # 顯示選中模型的詳細信息
            if final_selected_model in st.session_state.model_test_results:
                result = st.session_state.model_test_results[final_selected_model]
                if result.get('available', False):
                    st.success(f"✅ 已選擇: {model_info['icon']} {model_info['name']} (響應時間: {result.get('response_time', 0):.2f}s)")
                else:
                    st.error(f"❌ 選中模型不可用: {model_info['name']}")
                    st.warning("建議先測試模型可用性或選擇其他模型")
            else:
                st.info(f"📝 已選擇: {model_info['icon']} {model_info['name']} - {model_info['description']}")
                st.warning("⚠️ 未測試此模型可用性，建議先進行測試")
            
            # 提示詞輸入
            st.subheader("✏️ 輸入提示詞")
            
            # 重新生成檢查
            default_prompt = ""
            if hasattr(st.session_state, 'regenerate_prompt'):
                default_prompt = st.session_state.regenerate_prompt
                if hasattr(st.session_state, 'regenerate_model'):
                    st.session_state.selected_model = st.session_state.regenerate_model
                delattr(st.session_state, 'regenerate_prompt')
                if hasattr(st.session_state, 'regenerate_model'):
                    delattr(st.session_state, 'regenerate_model')
            
            prompt = st.text_area(
                "描述你想要生成的圖像",
                value=default_prompt,
                height=120,
                placeholder="例如：A cute cat wearing a wizard hat in a magical forest..."
            )
            
            # 高級設定
            with st.expander("🔧 高級設定"):
                col_size, col_num = st.columns(2)
                
                with col_size:
                    size_options = {
                        "1024x1024": "正方形 (1:1)",
                        "1152x896": "橫向 (4:3.5)",
                        "896x1152": "直向 (3.5:4)",
                        "1344x768": "寬屏 (16:9)",
                        "768x1344": "超高 (9:16)"
                    }
                    selected_size = st.selectbox(
                        "圖像尺寸",
                        options=list(size_options.keys()),
                        format_func=lambda x: f"{x} - {size_options[x]}",
                        index=0
                    )
                
                with col_num:
                    num_images = st.slider("生成數量", 1, 4, 1)
            
            # 快速提示詞
            st.subheader("💡 快速提示詞")
            
            model_type = model_info.get('type', '')
            if '快速' in model_type:
                category_default = "人物肖像"
            elif '創意' in model_type:
                category_default = "藝術創意"
            else:
                category_default = "自然風景"
            
            prompt_categories = {
                "人物肖像": [
                    "Professional headshot of a businesswoman in modern office",
                    "Portrait of an elderly man with wise eyes and gentle smile",
                    "Young artist with paint-splattered apron in studio"
                ],
                "自然風景": [
                    "Sunset over snow-capped mountains with alpine lake",
                    "Tropical beach with crystal clear water and palm trees", 
                    "Autumn forest with golden leaves and morning mist"
                ],
                "藝術創意": [
                    "Abstract geometric composition with vibrant colors",
                    "Watercolor painting of blooming cherry blossoms",
                    "Digital art of a dragon made of flowing water"
                ]
            }
            
            category = st.selectbox(
                "選擇類別",
                list(prompt_categories.keys()),
                index=list(prompt_categories.keys()).index(category_default)
            )
            
            prompt_cols = st.columns(len(prompt_categories[category]))
            for i, quick_prompt in enumerate(prompt_categories[category]):
                with prompt_cols[i]:
                    if st.button(
                        quick_prompt[:30] + "...",
                        key=f"quick_{category}_{i}",
                        use_container_width=True,
                        help=quick_prompt
                    ):
                        st.session_state.quick_prompt = quick_prompt
                        rerun_app()
            
            if hasattr(st.session_state, 'quick_prompt'):
                prompt = st.session_state.quick_prompt
                delattr(st.session_state, 'quick_prompt')
            
            # 生成按鈕
            generate_ready = (
                prompt.strip() and 
                api_configured and 
                (final_selected_model not in st.session_state.model_test_results or 
                 st.session_state.model_test_results[final_selected_model].get('available', True))
            )
            
            generate_btn = st.button(
                "🚀 生成圖像",
                type="primary",
                use_container_width=True,
                disabled=not generate_ready
            )
            
            # 顯示生成準備狀態
            if not generate_ready:
                if not prompt.strip():
                    st.warning("⚠️ 請輸入提示詞")
                elif not api_configured:
                    st.error("❌ 請配置 API 密鑰")
                elif (final_selected_model in st.session_state.model_test_results and 
                      not st.session_state.model_test_results[final_selected_model].get('available', True)):
                    st.error("❌ 選中的模型不可用，請選擇其他模型或重新測試")
        
        with col2:
            # 模型推薦面板
            if api_configured:
                show_model_recommendations()
            
            st.subheader("📋 使用說明")
            st.markdown(f"""
            **當前模型:** {FLUX_MODELS[final_selected_model]['name']}
            
            **Koyeb 部署特色:**
            - 🚀 自動縮放和 Scale-to-Zero
            - 🌐 全球 CDN 加速
            - 📊 實時資源監控
            - 🔒 安全的 API 密鑰管理
            
            **建議流程:**
            1. 測試模型可用性
            2. 選擇推薦的最佳模型  
            3. 輸入詳細提示詞
            4. 調整生成設定
            5. 開始生成
            """)
        
        # 圖像生成邏輯
        if generate_btn and generate_ready:
            config = st.session_state.api_config
            
            with st.spinner(f"正在使用 {FLUX_MODELS[final_selected_model]['name']} 生成圖像..."):
                generation_params = {
                    "model": final_selected_model,
                    "prompt": prompt,
                    "n": num_images,
                    "size": selected_size
                }
                
                success, result = generate_images_with_retry(
                    client, config['provider'], config['api_key'], 
                    config['base_url'], **generation_params
                )
                
                if success:
                    response = result
                    image_urls = [img.url for img in response.data]
                    
                    metadata = {
                        "size": selected_size,
                        "num_images": num_images,
                        "model_info": FLUX_MODELS[final_selected_model],
                        "api_provider": config['provider'],
                        "success": True,
                        "response_time": st.session_state.model_test_results.get(
                            final_selected_model, {}
                        ).get('response_time', 0)
                    }
                    
                    add_to_history(prompt, final_selected_model, image_urls, metadata)
                    st.success(f"✨ 成功生成 {len(response.data)} 張圖像！")
                    
                    # 顯示圖像
                    cols = st.columns(min(num_images, 2))
                    for i, image_data in enumerate(response.data):
                        with cols[i % len(cols)]:
                            st.subheader(f"圖像 {i+1}")
                            image_id = f"{len(st.session_state.generation_history)-1}_{i}"
                            display_image_with_actions(
                                image_data.url,
                                image_id,
                                st.session_state.generation_history[0]
                            )
                else:
                    st.error(f"❌ 生成失敗: {result}")
                    # 更新模型狀態
                    if final_selected_model in st.session_state.model_test_results:
                        st.session_state.model_test_results[final_selected_model]['available'] = False
                        st.session_state.model_test_results[final_selected_model]['error'] = result

# 模型測試頁面
with tab2:
    st.subheader("🧪 模型可用性測試")
    if not api_configured:
        st.warning("⚠️ 請先配置 API 密鑰")
        st.info("配置完成後即可測試模型可用性")
    else:
        show_model_status_dashboard()

# 歷史記錄頁面
with tab3:
    st.subheader("📚 生成歷史")
    
    if st.session_state.generation_history:
        # 搜索和篩選
        search_term = st.text_input("🔍 搜索提示詞", placeholder="輸入關鍵詞搜索...")
        
        filtered_history = st.session_state.generation_history
        if search_term:
            filtered_history = [
                item for item in st.session_state.generation_history
                if search_term.lower() in item['prompt'].lower()
            ]
        
        st.write(f"顯示 {len(filtered_history)} 條記錄")
        
        for item in filtered_history:
            with st.expander(f"🎨 {item['prompt'][:50]}... | {item['timestamp'].strftime('%m-%d %H:%M')}"):
                st.markdown(f"**提示詞**: {item['prompt']}")
                st.markdown(f"**模型**: {FLUX_MODELS.get(item['model'], {}).get('name', item['model'])}")
                st.markdown(f"**生成時間**: {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 顯示圖像
                cols = st.columns(min(len(item['images']), 3))
                for i, img_url in enumerate(item['images']):
                    with cols[i % len(cols)]:
                        display_image_with_actions(img_url, f"history_{item['id']}_{i}", item)
    else:
        st.info("尚無生成歷史")

# 收藏夾頁面
with tab4:
    st.subheader("⭐ 我的收藏")
    
    if st.session_state.favorite_images:
        cols = st.columns(3)
        for i, fav in enumerate(st.session_state.favorite_images):
            with cols[i % 3]:
                display_image_with_actions(fav['image_url'], fav['id'], fav.get('history_item'))
                if fav.get('history_item'):
                    st.caption(f"提示詞: {fav['history_item']['prompt'][:30]}...")
                st.caption(f"收藏時間: {fav['timestamp'].strftime('%m-%d %H:%M')}")
    else:
        st.info("尚無收藏圖像")

# 幫助頁面
with tab5:
    st.subheader("💡 使用幫助")
    
    st.markdown("### 🚀 Koyeb 部署優勢")
    st.markdown("""
    **Scale-to-Zero 自動縮放:**
    - 閒置時自動縮減到零成本
    - 有請求時快速啟動 (200ms)
    - 智能負載均衡

    **全球部署:**
    - 50+ 個地區可選
    - 自動 CDN 加速
    - 就近用戶訪問

    **安全可靠:**
    - 自動 HTTPS/SSL
    - 環境變量加密
    - 高可用性保障
    """)
    
    st.markdown("### 🎯 模型測試功能")
    st.markdown("""
    **模型測試的重要性:**
    - 🔍 確認模型是否可用
    - ⚡ 測量響應時間
    - 🎯 獲得最佳模型推薦
    - 📊 追蹤模型狀態變化

    **如何使用:**
    1. 配置 API 密鑰
    2. 點擊 "測試所有模型"
    3. 查看測試結果
    4. 選擇推薦的最佳模型
    5. 開始生成圖像
    """)
    
    st.markdown("### 🔧 故障排除")
    st.markdown("""
    **常見問題:**

    **模型不可用 (404 錯誤):**
    - 模型名稱可能不正確
    - API 提供商可能不支持該模型
    - 模型可能暫時離線

    **權限錯誤 (401/403):**
    - 檢查 API 密鑰是否正確
    - 確認帳戶權限和餘額
    - 檢查 API 端點設置

    **Koyeb 部署問題:**
    - 檢查環境變量設置
    - 確認端口配置 (8000)
    - 查看應用日誌排錯
    """)

# 頁腳
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🚀 部署在 Koyeb | 🎨 Powered by Flux AI | 
    ⚡ Scale-to-Zero 自動縮放 | 🌐 全球 CDN 加速
</div>
""", unsafe_allow_html=True)
