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
from urllib.parse import urlencode, quote

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
    "Pollinations.ai": {
        "name": "Pollinations.ai",
        "base_url_default": "https://image.pollinations.ai",
        "key_prefix": "",
        "description": "免費、開源的圖像生成 API (無需密鑰)",
        "icon": "🌸"
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
    """自動發現模型，現已支持 Pollinations.ai"""
    discovered_models = {}
    
    try:
        if provider == "Pollinations.ai":
            models_url = f"{base_url}/models"
            response = requests.get(models_url, timeout=10)
            if response.status_code == 200:
                models_list = response.json()
                for model_name in models_list:
                    model_id = model_name
                    model_info = analyze_model_name(model_id)
                    model_info['source'] = 'pollinations'
                    model_info['type'] = '圖像專用'
                    model_info['icon'] = '🌸'
                    discovered_models[model_id] = model_info
            else:
                st.warning(f"無法從 Pollinations.ai 獲取模型列表 (HTTP {response.status_code})")

        elif provider == "Hugging Face":
            for endpoint in HF_FLUX_ENDPOINTS:
                model_id = endpoint.split('/')[-1]
                model_info = analyze_model_name(model_id, endpoint)
                model_info['source'] = 'huggingface'
                model_info['endpoint'] = endpoint
                discovered_models[model_id] = model_info
        else:
            response = client.models.list()
            for model in response.data:
                model_id = model.id
                # 限制只顯示包含 'flux' 的圖像模型
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
            
    # 按 'priority' 排序
    sorted_models = sorted(merged_models.items(), key=lambda item: item[1].get('priority', 999))
    return dict(sorted_models)


def validate_api_key(api_key: str, base_url: str, provider: str) -> Tuple[bool, str]:
    """驗證 API 密鑰是否有效，新增 Pollinations.ai 驗證"""
    try:
        if provider == "Pollinations.ai":
            test_url = f"{base_url}/models"
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                return True, "Pollinations.ai 服務連接成功"
            else:
                return False, f"HTTP {response.status_code}: Pollinations.ai 連接失敗"

        elif provider == "Hugging Face":
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
    """帶重試機制的圖像生成，新增 Pollinations.ai 支持"""
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                st.info(f"🔄 嘗試重新生成 (第 {attempt + 1}/{max_retries} 次)")

            if provider == "Pollinations.ai":
                # Pollinations.ai GET 請求
                prompt = params.get("prompt", "")
                width, height = params.get("size", "1024x1024").split('x')
                
                query_params = {
                    "model": params.get("model"),
                    "width": width,
                    "height": height,
                    "seed": random.randint(0, 1000000), # Pollinations uses seed
                    "nologo": "true"
                }
                
                # 清理 None 值
                query_params = {k: v for k, v in query_params.items() if v is not None}
                
                encoded_prompt = quote(prompt)
                request_url = f"{base_url}/prompt/{encoded_prompt}?{urlencode(query_params)}"
                
                response = requests.get(request_url, timeout=120)

                if response.status_code == 200:
                    # 模擬 OpenAI 響應格式
                    class MockResponse:
                        def __init__(self, image_data):
                            self.data = [type('obj', (object,), {
                                'url': f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                            })()]
                    
                    # Pollinations.ai 只支持一次生成一張圖
                    num_images = params.get("n", 1)
                    if num_images > 1:
                        st.warning("Pollinations.ai 每次僅支持生成一張圖像，已自動設為 1。")

                    return True, MockResponse(response.content)
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            elif provider == "Hugging Face":
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
                key=f"download_{image
