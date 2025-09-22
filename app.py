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
import re

# 兼容性函數
def rerun_app():
    """兼容不同 Streamlit 版本的重新運行函數"""
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    else:
        st.stop()

# 設定頁面配置
st.set_page_config(
    page_title="Flux AI 圖像生成器 Pro - 完整版",
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
    "Together AI": {
        "name": "Together AI",
        "base_url_default": "https://api.together.xyz/v1",
        "key_prefix": "",
        "description": "Together AI 平台",
        "icon": "🤝"
    },
    "Fireworks AI": {
        "name": "Fireworks AI",
        "base_url_default": "https://api.fireworks.ai/inference/v1",
        "key_prefix": "",
        "description": "Fireworks AI 快速推理",
        "icon": "🎆"
    },
    "Custom": {
        "name": "自定義 API",
        "base_url_default": "",
        "key_prefix": "",
        "description": "自定義的 API 端點",
        "icon": "🔧"
    }
}

# 基礎 Flux 模型配置
BASE_FLUX_MODELS = {
    "flux.1-schnell": {
        "name": "FLUX.1 Schnell",
        "description": "最快的生成速度，開源模型",
        "icon": "⚡",
        "type": "快速生成",
        "test_prompt": "A simple cat sitting on a table",
        "expected_size": "1024x1024",
        "priority": 1,
        "source": "base"
    },
    "flux.1-dev": {
        "name": "FLUX.1 Dev", 
        "description": "開發版本，平衡速度與質量",
        "icon": "🔧",
        "type": "開發版本",
        "test_prompt": "A beautiful landscape with mountains",
        "expected_size": "1024x1024",
        "priority": 2,
        "source": "base"
    },
    "flux.1.1-pro": {
        "name": "FLUX.1.1 Pro",
        "description": "改進的旗艦模型，最佳品質",
        "icon": "👑",
        "type": "旗艦版本",
        "test_prompt": "Professional portrait of a person in business attire",
        "expected_size": "1024x1024",
        "priority": 3,
        "source": "base"
    },
    "flux.1-kontext-pro": {
        "name": "FLUX.1 Kontext Pro",
        "description": "支持圖像編輯和上下文理解",
        "icon": "🎯",
        "type": "編輯專用",
        "test_prompt": "Abstract geometric shapes in vibrant colors",
        "expected_size": "1024x1024",
        "priority": 4,
        "source": "base"
    }
}

# 模型自動發現規則
FLUX_MODEL_PATTERNS = {
    r'flux[\.\-]?1[\.\-]?schnell': {
        "name_template": "FLUX.1 Schnell",
        "icon": "⚡",
        "type": "快速生成",
        "priority_base": 100
    },
    r'flux[\.\-]?1[\.\-]?dev': {
        "name_template": "FLUX.1 Dev",
        "icon": "🔧", 
        "type": "開發版本",
        "priority_base": 200
    },
    r'flux[\.\-]?1[\.\-]?pro': {
        "name_template": "FLUX.1 Pro",
        "icon": "👑",
        "type": "專業版本",
        "priority_base": 300
    },
    r'flux[\.\-]?1[\.\-]?kontext': {
        "name_template": "FLUX.1 Kontext",
        "icon": "🎯",
        "type": "上下文理解",
        "priority_base": 400
    }
}

# 提供商特定的模型端點
HF_FLUX_ENDPOINTS = [
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1.1-pro",
]

def auto_discover_flux_models(client, provider: str, api_key: str, base_url: str) -> Dict[str, Dict]:
    """自動發現 Flux 模型"""
    discovered_models = {}
    
    try:
        if provider == "Hugging Face":
            for endpoint in HF_FLUX_ENDPOINTS:
                model_id = endpoint.split('/')[-1]
                model_info = analyze_model_name(model_id, endpoint)
                model_info['source'] = 'huggingface'
                model_info['endpoint'] = endpoint
                discovered_models[model_id] = model_info
        else:
            response = client.models.list()
            for model in response.data:
                model_id = model.id.lower()
                if is_flux_model(model_id):
                    model_info = analyze_model_name(model.id)
                    model_info['source'] = 'api_discovery'
                    discovered_models[model.id] = model_info
        return discovered_models
    except Exception as e:
        st.warning(f"模型自動發現失敗: {str(e)}")
        return {}

def is_flux_model(model_name: str) -> bool:
    """檢查模型名稱是否為 Flux 模型"""
    model_lower = model_name.lower()
    flux_keywords = ['flux', 'black-forest-labs']
    return any(keyword in model_lower for keyword in flux_keywords)

def analyze_model_name(model_id: str, full_path: str = None) -> Dict:
    """分析模型名稱並生成模型信息"""
    model_lower = model_id.lower()
    
    for pattern, info in FLUX_MODEL_PATTERNS.items():
        if re.search(pattern, model_lower):
            analyzed_info = {
                "name": info["name_template"],
                "icon": info["icon"],
                "type": info["type"],
                "description": f"自動發現的 {info['name_template']} 模型",
                "test_prompt": "A beautiful scene with detailed elements",
                "expected_size": "1024x1024",
                "priority": info["priority_base"] + hash(model_id) % 100,
                "auto_discovered": True
            }
            
            if full_path:
                analyzed_info["full_path"] = full_path
                if '/' in full_path:
                    author = full_path.split('/')[0]
                    analyzed_info["name"] += f" ({author})"
            
            return analyzed_info
    
    return {
        "name": model_id.replace('-', ' ').replace('_', ' ').title(),
        "icon": "🤖",
        "type": "自動發現",
        "description": f"自動發現的模型: {model_id}",
        "test_prompt": "A detailed and beautiful image",
        "expected_size": "1024x1024", 
        "priority": 999,
        "auto_discovered": True,
        "full_path": full_path if full_path else model_id
    }

def merge_models() -> Dict[str, Dict]:
    """合併基礎模型和自動發現的模型"""
    discovered = st.session_state.get('discovered_models', {})
    merged_models = BASE_FLUX_MODELS.copy()
    
    for model_id, model_info in discovered.items():
        if model_id not in merged_models:
            merged_models[model_id] = model_info
    
    return merged_models

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

def test_model_availability(client, model_name: str, provider: str, api_key: str, base_url: str, test_prompt: str = None) -> Dict:
    """測試特定模型的可用性"""
    all_models = merge_models()
    if test_prompt is None:
        test_prompt = all_models.get(model_name, {}).get('test_prompt', 'A simple test image')
    
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
            headers = {"Authorization": f"Bearer {api_key}"}
            data = {"inputs": test_prompt}
            
            model_info = all_models.get(model_name, {})
            endpoint_path = model_info.get('full_path', f"black-forest-labs/{model_name}")
            
            response = requests.post(
                f"{base_url}/models/{endpoint_path}",
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
                        'test_prompt': test_prompt,
                        'endpoint': endpoint_path
                    }
                })
            else:
                test_result.update({
                    'available': False,
                    'error': f"HTTP {response.status_code}",
                    'details': {'status_code': response.status_code}
                })
        else:
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
                all_models = merge_models()
                model_info = all_models.get(model_name, {})
                endpoint_path = model_info.get('full_path', f"black-forest-labs/{model_name}")
                
                response = requests.post(
                    f"{base_url}/models/{endpoint_path}",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    # 模擬 OpenAI 響應格式
                    class MockResponse:
                        def __init__(self, image_data):
                            self.data = [type('obj', (object,), {
                                'url': f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                            })()]
                    
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
                if any(code in error_msg for code in ["500", "502", "503", "429"]):
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
    
    if 'discovered_models' not in st.session_state:
        st.session_state.discovered_models = {}

def add_to_history(prompt: str, model: str, images: List[str], metadata: Dict):
    """添加生成記錄到歷史"""
    history_item = {
        "timestamp": datetime.datetime.now(),
        "prompt": prompt,
        "model": model,
        "images": images,
        "metadata": metadata,
        "id": str(uuid.uuid4())
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
            base64_data = image_url.split(',')[1]
            img_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(img_data))
        else:
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
                rerun_app()
    
    except Exception as e:
        st.error(f"圖像顯示錯誤: {str(e)}")

def init_api_client():
    """初始化 API 客戶端"""
    if 'api_config' in st.session_state and st.session_state.api_config.get('api_key'):
        config = st.session_state.api_config
        if config['provider'] == "Hugging Face":
            return None
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
        st.success("🗑️ API 設置已清除")
        rerun_app()

def auto_discover_models():
    """執行自動模型發現"""
    if 'api_config' not in st.session_state or not st.session_state.api_config.get('api_key'):
        st.error("❌ 請先配置 API 密鑰")
        return
    
    config = st.session_state.api_config
    
    with st.spinner("🔍 正在自動發現 Flux 模型..."):
        if config['provider'] == "Hugging Face":
            client = None
        else:
            client = OpenAI(
                api_key=config['api_key'],
                base_url=config['base_url']
            )
        
        discovered = auto_discover_flux_models(
            client, config['provider'], config['api_key'], config['base_url']
        )
        
        if 'discovered_models' not in st.session_state:
            st.session_state.discovered_models = {}
        
        new_count = 0
        for model_id, model_info in discovered.items():
            if model_id not in st.session_state.discovered_models:
                new_count += 1
            st.session_state.discovered_models[model_id] = model_info
        
        if new_count > 0:
            st.success(f"✅ 發現 {new_count} 個新的 Flux 模型！")
        else:
            st.info("ℹ️ 未發現新的 Flux 模型")
        
        rerun_app()

# 初始化
init_session_state()
client = init_api_client()
api_configured = client is not None or (st.session_state.api_config.get('provider') == "Hugging Face" and st.session_state.api_config.get('api_key'))

# 側邊欄
with st.sidebar:
    show_api_settings()
    st.markdown("---")
    
    if api_configured:
        st.success("🟢 API 已配置")
        if st.button("🔍 發現模型", use_container_width=True):
            auto_discover_models()
    else:
        st.error("🔴 API 未配置")

# 主標題
st.title("🎨 Flux AI 圖像生成器 Pro - 完整版")

# 頁面導航
tab1, tab2, tab3 = st.tabs(["🚀 圖像生成", "📚 歷史記錄", "⭐ 收藏夾"])

# 圖像生成頁面
with tab1:
    if not api_configured:
        st.warning("⚠️ 請先在側邊欄配置 API 密鑰")
        st.info("配置完成後即可開始生成圖像")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎨 圖像生成")
            
            # 使用合併後的模型列表
            all_models = merge_models()
            
            if not all_models:
                st.warning("⚠️ 尚未發現任何模型，請點擊側邊欄的「發現模型」按鈕")
            else:
                # 模型選擇
                model_options = list(all_models.keys())
                if 'selected_model' not in st.session_state:
                    st.session_state.selected_model = model_options[0]
                
                if st.session_state.selected_model not in model_options:
                    st.session_state.selected_model = model_options[0]
                
                selected_model = st.selectbox(
                    "選擇模型:",
                    options=model_options,
                    index=model_options.index(st.session_state.selected_model),
                    format_func=lambda x: f"{all_models[x].get('icon', '🤖')} {all_models[x].get('name', x)}"
                )
                
                st.session_state.selected_model = selected_model
                
                # 顯示模型信息
                model_info = all_models[selected_model]
                st.info(f"**{model_info.get('name')}**: {model_info.get('description', 'N/A')}")
                
                # 檢查重新生成狀態
                default_prompt = ""
                if hasattr(st.session_state, 'regenerate_prompt'):
                    default_prompt = st.session_state.regenerate_prompt
                    if hasattr(st.session_state, 'regenerate_model') and st.session_state.regenerate_model in model_options:
                        st.session_state.selected_model = st.session_state.regenerate_model
                        selected_model = st.session_state.selected_model
                    delattr(st.session_state, 'regenerate_prompt')
                    if hasattr(st.session_state, 'regenerate_model'):
                        delattr(st.session_state, 'regenerate_model')
                
                # 提示詞輸入
                prompt = st.text_area(
                    "輸入提示詞:",
                    value=default_prompt,
                    height=120,
                    placeholder="描述您想要生成的圖像，例如：A majestic dragon flying over ancient mountains during sunset, highly detailed, fantasy art style"
                )
                
                # 高級設置
                with st.expander("🔧 高級設置"):
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
                
                # 快速提示詞模板
                st.subheader("💡 快速提示詞模板")
                
                prompt_categories = {
                    "人物肖像": [
                        "Professional headshot of a businesswoman in modern office",
                        "Portrait of an elderly man with wise eyes and gentle smile",
                        "Young artist with paint-splattered apron in creative studio"
                    ],
                    "自然風景": [
                        "Sunset over snow-capped mountains with alpine lake reflection",
                        "Tropical beach with crystal clear turquoise water and palm trees",
                        "Autumn forest with golden leaves and morning mist"
                    ],
                    "藝術創作": [
                        "Abstract geometric composition with vibrant colors and flowing lines",
                        "Watercolor painting of blooming cherry blossoms in spring",
                        "Digital art of a majestic dragon made of flowing water elements"
                    ],
                    "科幻幻想": [
                        "Futuristic cityscape with flying vehicles and neon-lit skyscrapers",
                        "Space station orbiting a distant planet with nebula background",
                        "Cyberpunk street scene with holographic advertisements and rain"
                    ]
                }
                
                category = st.selectbox("選擇類別", list(prompt_categories.keys()))
                
                cols = st.columns(len(prompt_categories[category]))
                for i, template_prompt in enumerate(prompt_categories[category]):
                    with cols[i]:
                        if st.button(
                            template_prompt[:25] + "...",
                            key=f"template_{category}_{i}",
                            use_container_width=True,
                            help=template_prompt
                        ):
                            st.session_state.quick_prompt = template_prompt
                            rerun_app()
                
                # 應用快速提示詞
                if hasattr(st.session_state, 'quick_prompt'):
                    prompt = st.session_state.quick_prompt
                    delattr(st.session_state, 'quick_prompt')
                    rerun_app()
                
                # 生成按鈕
                generate_ready = prompt.strip() and api_configured
                
                generate_btn = st.button(
                    "🚀 生成圖像",
                    type="primary",
                    use_container_width=True,
                    disabled=not generate_ready
                )
                
                if not generate_ready:
                    if not prompt.strip():
                        st.warning("⚠️ 請輸入提示詞")
                    elif not api_configured:
                        st.error("❌ 請配置 API 密鑰")
                
                # 圖像生成邏輯
                if generate_btn and generate_ready:
                    config = st.session_state.api_config
                    
                    with st.spinner(f"🎨 使用 {model_info.get('name', selected_model)} 正在生成圖像..."):
                        # 顯示進度信息
                        progress_info = st.empty()
                        progress_info.info(f"⏳ 模型: {model_info.get('name')} | 尺寸: {selected_size} | 數量: {num_images}")
                        
                        generation_params = {
                            "model": selected_model,
                            "prompt": prompt,
                            "n": num_images,
                            "size": selected_size
                        }
                        
                        success, result = generate_images_with_retry(
                            client, config['provider'], config['api_key'], 
                            config['base_url'], **generation_params
                        )
                        
                        progress_info.empty()
                        
                        if success:
                            response = result
                            image_urls = [img.url for img in response.data]
                            
                            metadata = {
                                "size": selected_size,
                                "num_images": num_images,
                                "model_name": model_info.get('name', selected_model),
                                "api_provider": config['provider'],
                                "generation_time": time.time()
                            }
                            
                            add_to_history(prompt, selected_model, image_urls, metadata)
                            st.success(f"✨ 成功生成 {len(response.data)} 張圖像！")
                            
                            # 顯示生成的圖像
                            if len(response.data) == 1:
                                # 單張圖像，全寬顯示
                                st.subheader("🎨 生成結果")
                                image_id = f"{st.session_state.generation_history[0]['id']}_0"
                                display_image_with_actions(
                                    response.data[0].url,
                                    image_id,
                                    st.session_state.generation_history[0]
                                )
                            else:
                                # 多張圖像，網格顯示
                                st.subheader("🎨 生成結果")
                                cols = st.columns(min(num_images, 2))
                                for i, image_data in enumerate(response.data):
                                    with cols[i % len(cols)]:
                                        st.markdown(f"**圖像 {i+1}**")
                                        image_id = f"{st.session_state.generation_history[0]['id']}_{i}"
                                        display_image_with_actions(
                                            image_data.url,
                                            image_id,
                                            st.session_state.generation_history[0]
                                        )
                        else:
                            st.error(f"❌ 生成失敗: {result}")
                            
                            # 提供錯誤解決建議
                            error_suggestions = {
                                "401": "🔐 檢查 API 密鑰是否正確",
                                "403": "🚫 檢查 API 密鑰權限",
                                "404": "🔍 檢查模型名稱是否正確",
                                "429": "⏳ 請求過於頻繁，稍後再試",
                                "500": "🔧 服務器錯誤，請稍後重試"
                            }
                            
                            for error_code, suggestion in error_suggestions.items():
                                if error_code in str(result):
                                    st.info(f"💡 建議: {suggestion}")
                                    break
        
        with col2:
            st.subheader("ℹ️ 生成信息")
            
            all_models = merge_models()
            base_count = len([m for m in all_models.values() if m.get('source') == 'base'])
            discovered_count = len([m for m in all_models.values() if m.get('auto_discovered')])
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("可用模型", len(all_models))
            with col_stat2:
                st.metric("生成記錄", len(st.session_state.generation_history))
            
            st.markdown("### 📋 使用建議")
            st.markdown("""
            **提示詞優化技巧:**
            - 🎯 使用具體描述而非抽象概念
            - 🎨 加入藝術風格關鍵詞
            - 📐 指定構圖和視角
            - 🌈 描述色彩和光線效果
            
            **Koyeb 部署特色:**
            - 🚀 Scale-to-Zero 自動縮放
            - 🌐 全球 CDN 加速
            - 📊 實時資源監控
            - 🔒 安全的 API 管理
            """)

# 歷史記錄頁面
with tab2:
    st.subheader("📚 生成歷史")
    
    if st.session_state.generation_history:
        # 搜索功能
        search_term = st.text_input("🔍 搜索歷史記錄", placeholder="輸入關鍵詞搜索提示詞...")
        
        filtered_history = st.session_state.generation_history
        if search_term:
            filtered_history = [
                item for item in st.session_state.generation_history
                if search_term.lower() in item['prompt'].lower()
            ]
        
        st.info(f"顯示 {len(filtered_history)} / {len(st.session_state.generation_history)} 條記錄")
        
        for item in filtered_history:
            with st.expander(
                f"🎨 {item['prompt'][:60]}{'...' if len(item['prompt']) > 60 else ''} | {item['timestamp'].strftime('%m-%d %H:%M')}"
            ):
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
                    st.markdown(f"**提示詞**: {item['prompt']}")
                    all_models = merge_models()
                    model_name = all_models.get(item['model'], {}).get('name', item['model'])
                    st.markdown(f"**模型**: {model_name}")
                    st.markdown(f"**時間**: {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if 'metadata' in item:
                        metadata = item['metadata']
                        st.markdown(f"**尺寸**: {metadata.get('size', 'N/A')}")
                        st.markdown(f"**數量**: {metadata.get('num_images', 'N/A')}")
                
                with col_actions:
                    if st.button("🔄 重新生成", key=f"regen_{item['id']}"):
                        st.session_state.regenerate_prompt = item['prompt']
                        st.session_state.regenerate_model = item['model']
                        rerun_app()
                
                # 顯示圖像
                if item['images']:
                    cols = st.columns(min(len(item['images']), 3))
                    for i, img_url in enumerate(item['images']):
                        with cols[i % len(cols)]:
                            display_image_with_actions(img_url, f"history_{item['id']}_{i}", item)
    else:
        st.info("📭 尚無生成歷史，開始生成一些圖像吧！")

# 收藏夾頁面
with tab3:
    st.subheader("⭐ 我的收藏")
    
    if st.session_state.favorite_images:
        # 批量操作
        col_batch1, col_batch2 = st.columns(2)
        with col_batch1:
            if st.button("📥 批量下載", use_container_width=True):
                st.info("🚧 批量下載功能開發中")
        with col_batch2:
            if st.button("🗑️ 清空收藏", use_container_width=True):
                if st.checkbox("確認清空所有收藏", key="confirm_clear_favorites"):
                    st.session_state.favorite_images = []
                    st.success("已清空所有收藏")
                    rerun_app()
        
        # 顯示收藏的圖像
        cols = st.columns(3)
        for i, fav in enumerate(st.session_state.favorite_images):
            with cols[i % 3]:
                display_image_with_actions(fav['image_url'], fav['id'], fav.get('history_item'))
                
                # 顯示收藏信息
                if fav.get('history_item'):
                    st.caption(f"💭 {fav['history_item']['prompt'][:40]}...")
                st.caption(f"⭐ 收藏於: {fav['timestamp'].strftime('%m-%d %H:%M')}")
    else:
        st.info("⭐ 尚無收藏圖像，在生成的圖像上點擊收藏按鈕來添加收藏！")

# 頁腳
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    🚀 <strong>部署在 Koyeb</strong> | 
    🎨 <strong>Powered by Flux AI</strong> | 
    ⚡ <strong>自動縮放</strong> | 
    🌐 <strong>全球加速</strong>
    <br><br>
    <small>完整的圖像生成、模型發現、歷史管理功能</small>
</div>
""", unsafe_allow_html=True)
