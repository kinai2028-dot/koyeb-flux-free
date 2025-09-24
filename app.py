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

# API 提供商配置 (新增 FLUX Krea AI Studio 和註冊地址)
API_PROVIDERS = {
    "OpenAI Compatible": {
        "name": "OpenAI Compatible API",
        "base_url_default": "https://api.openai.com/v1",
        "key_prefix": "sk-",
        "description": "OpenAI 官方或兼容的 API 服務",
        "icon": "🤖",
        "register_url": "https://platform.openai.com/signup",
        "api_docs": "https://platform.openai.com/docs",
        "pricing_url": "https://openai.com/pricing"
    },
    "Navy": {
        "name": "Navy API",
        "base_url_default": "https://api.navy/v1",
        "key_prefix": "sk-",
        "description": "Navy 提供的 AI 圖像生成服務",
        "icon": "⚓",
        "register_url": "https://api.navy",
        "api_docs": "https://api.navy/docs",
        "pricing_url": "https://api.navy/pricing"
    },
    "Pollinations.ai": {
        "name": "Pollinations.ai",
        "base_url_default": "https://image.pollinations.ai",
        "key_prefix": "",
        "description": "支援免費和認證模式的圖像生成 API",
        "icon": "🌸",
        "auth_modes": ["free", "referrer", "token"],
        "register_url": "https://auth.pollinations.ai",
        "api_docs": "https://docs.pollinations.ai",
        "pricing_url": "https://pollinations.ai/pricing"
    },
    "FLUX Krea AI Studio": {
        "name": "FLUX Krea AI Studio",
        "base_url_default": "https://api.krea.ai/v1",
        "key_prefix": "krea_",
        "description": "專業美學圖像生成平台，專注於藝術級品質",
        "icon": "🎭",
        "register_url": "https://krea.ai/signup",
        "api_docs": "https://docs.krea.ai",
        "pricing_url": "https://krea.ai/pricing",
        "features": ["高品質美學", "藝術風格", "專業控制"]
    },
    "Hugging Face": {
        "name": "Hugging Face Inference",
        "base_url_default": "https://api-inference.huggingface.co",
        "key_prefix": "hf_",
        "description": "Hugging Face Inference API",
        "icon": "🤗",
        "register_url": "https://huggingface.co/join",
        "api_docs": "https://huggingface.co/docs/api-inference/index",
        "pricing_url": "https://huggingface.co/pricing"
    },
    "Together AI": {
        "name": "Together AI",
        "base_url_default": "https://api.together.xyz/v1",
        "key_prefix": "",
        "description": "Together AI 平台",
        "icon": "🤝",
        "register_url": "https://api.together.xyz/signup",
        "api_docs": "https://docs.together.ai",
        "pricing_url": "https://www.together.ai/pricing"
    },
    "Fireworks AI": {
        "name": "Fireworks AI",
        "base_url_default": "https://api.fireworks.ai/inference/v1",
        "key_prefix": "",
        "description": "Fireworks AI 快速推理",
        "icon": "🎆",
        "register_url": "https://fireworks.ai/login",
        "api_docs": "https://readme.fireworks.ai",
        "pricing_url": "https://fireworks.ai/pricing"
    },
    "Custom": {
        "name": "自定義 API",
        "base_url_default": "",
        "key_prefix": "",
        "description": "自定義的 API 端點",
        "icon": "🔧",
        "register_url": "",
        "api_docs": "",
        "pricing_url": ""
    }
}

# 基礎 Flux 模型配置 (新增 Krea AI Studio 專用模型)
BASE_FLUX_MODELS = {
    "flux.1-schnell": {
        "name": "FLUX.1 Schnell",
        "description": "最快的生成速度，開源模型",
        "icon": "⚡",
        "type": "快速生成",
        "test_prompt": "A simple cat sitting on a table",
        "expected_size": "1024x1024",
        "priority": 1,
        "source": "base",
        "auth_required": False
    },
    "flux.1-dev": {
        "name": "FLUX.1 Dev",
        "description": "開發版本，平衡速度與質量",
        "icon": "🔧",
        "type": "開發版本",
        "test_prompt": "A beautiful landscape with mountains",
        "expected_size": "1024x1024",
        "priority": 2,
        "source": "base",
        "auth_required": False
    },
    "flux.1.1-pro": {
        "name": "FLUX.1.1 Pro",
        "description": "改進的旗艦模型，最佳品質",
        "icon": "👑",
        "type": "旗艦版本",
        "test_prompt": "Professional portrait of a person in business attire",
        "expected_size": "1024x1024",
        "priority": 3,
        "source": "base",
        "auth_required": False
    },
    "flux.1-kontext-pro": {
        "name": "FLUX.1 Kontext Pro",
        "description": "支持圖像編輯和上下文理解（需認證）",
        "icon": "🎯",
        "type": "編輯專用",
        "test_prompt": "Abstract geometric shapes in vibrant colors",
        "expected_size": "1024x1024",
        "priority": 4,
        "source": "base",
        "auth_required": True
    },
    "krea-flux-aesthetic": {
        "name": "Krea FLUX Aesthetic",
        "description": "Krea AI Studio 專業美學模型，藝術級品質",
        "icon": "🎭",
        "type": "美學專業",
        "test_prompt": "Aesthetic portrait with cinematic lighting and artistic composition",
        "expected_size": "1024x1024",
        "priority": 5,
        "source": "base",
        "auth_required": False,
        "provider_specific": "FLUX Krea AI Studio"
    },
    "krea-flux-artistic": {
        "name": "Krea FLUX Artistic",
        "description": "Krea AI Studio 藝術創作模型，風格化強",
        "icon": "🖼️",
        "type": "藝術創作",
        "test_prompt": "Artistic interpretation with unique style and creative elements",
        "expected_size": "1024x1024",
        "priority": 6,
        "source": "base",
        "auth_required": False,
        "provider_specific": "FLUX Krea AI Studio"
    }
}

# 模型自動發現規則 (新增 Krea 相關模式)
FLUX_MODEL_PATTERNS = {
    r'flux[\.\-]?1[\.\-]?schnell': {
        "name_template": "FLUX.1 Schnell",
        "icon": "⚡",
        "type": "快速生成",
        "priority_base": 100,
        "auth_required": False
    },
    r'flux[\.\-]?1[\.\-]?dev': {
        "name_template": "FLUX.1 Dev",
        "icon": "🔧",
        "type": "開發版本",
        "priority_base": 200,
        "auth_required": False
    },
    r'flux[\.\-]?1[\.\-]?pro': {
        "name_template": "FLUX.1 Pro",
        "icon": "👑",
        "type": "專業版本",
        "priority_base": 300,
        "auth_required": False
    },
    r'flux[\.\-]?1[\.\-]?kontext|kontext': {
        "name_template": "FLUX.1 Kontext",
        "icon": "🎯",
        "type": "上下文理解",
        "priority_base": 400,
        "auth_required": True
    },
    r'krea[\.\-]?flux[\.\-]?aesthetic|aesthetic': {
        "name_template": "Krea FLUX Aesthetic",
        "icon": "🎭",
        "type": "美學專業",
        "priority_base": 150,
        "auth_required": False
    },
    r'krea[\.\-]?flux[\.\-]?artistic|artistic': {
        "name_template": "Krea FLUX Artistic", 
        "icon": "🖼️",
        "type": "藝術創作",
        "priority_base": 160,
        "auth_required": False
    }
}

# 提供商特定的模型端點
HF_FLUX_ENDPOINTS = [
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1.1-pro",
]

# Krea AI Studio 專用端點
KREA_FLUX_ENDPOINTS = [
    "krea/flux-aesthetic-v1",
    "krea/flux-artistic-v1",
    "krea/flux-professional-v1"
]

def auto_discover_flux_models(client, provider: str, api_key: str, base_url: str) -> Dict[str, Dict]:
    """自動發現模型，現已支持 Pollinations.ai 和 FLUX Krea AI Studio"""
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

        elif provider == "FLUX Krea AI Studio":
            # Krea AI Studio 模型發現
            for endpoint in KREA_FLUX_ENDPOINTS:
                model_id = endpoint.split('/')[-1]
                model_info = analyze_model_name(model_id, endpoint)
                model_info['source'] = 'krea'
                model_info['endpoint'] = endpoint
                model_info['provider_specific'] = 'FLUX Krea AI Studio'
                discovered_models[model_id] = model_info
            
            # 嘗試從 API 獲取更多模型
            try:
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                models_url = f"{base_url}/models"
                response = requests.get(models_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    models_data = response.json()
                    if isinstance(models_data, dict) and 'data' in models_data:
                        for model in models_data['data']:
                            model_id = model.get('id', '')
                            if is_flux_model(model_id) or 'krea' in model_id.lower():
                                model_info = analyze_model_name(model_id)
                                model_info['source'] = 'krea_api'
                                model_info['provider_specific'] = 'FLUX Krea AI Studio'
                                discovered_models[model_id] = model_info
            except Exception as e:
                st.info(f"Krea AI Studio API 模型發現: {str(e)[:50]}...")

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
    flux_keywords = ['flux', 'black-forest-labs', 'kontext', 'krea', 'aesthetic', 'artistic']
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
                "auto_discovered": True,
                "auth_required": info.get("auth_required", False)
            }
            
            if full_path:
                analyzed_info["full_path"] = full_path
                if '/' in full_path:
                    author = full_path.split('/')[0]
                    analyzed_info["name"] += f" ({author})"
            
            return analyzed_info
    
    # 特殊處理 Krea 相關模型
    if 'krea' in model_lower or 'aesthetic' in model_lower or 'artistic' in model_lower:
        return {
            "name": model_id.replace('-', ' ').replace('_', ' ').title(),
            "icon": "🎭",
            "type": "美學專業",
            "description": f"Krea AI Studio 專業模型: {model_id}",
            "test_prompt": "Aesthetic portrait with professional lighting",
            "expected_size": "1024x1024",
            "priority": 150 + hash(model_id) % 50,
            "auto_discovered": True,
            "auth_required": False,
            "provider_specific": "FLUX Krea AI Studio",
            "full_path": full_path if full_path else model_id
        }
    
    return {
        "name": model_id.replace('-', ' ').replace('_', ' ').title(),
        "icon": "🤖",
        "type": "自動發現",
        "description": f"自動發現的模型: {model_id}",
        "test_prompt": "A detailed and beautiful image",
        "expected_size": "1024x1024",
        "priority": 999,
        "auto_discovered": True,
        "auth_required": 'kontext' in model_id.lower(),
        "full_path": full_path if full_path else model_id
    }

def merge_models() -> Dict[str, Dict]:
    """合併基礎模型和自動發現的模型"""
    discovered = st.session_state.get('discovered_models', {})
    merged_models = BASE_FLUX_MODELS.copy()
    
    # 根據提供商過濾模型
    current_provider = st.session_state.api_config.get('provider', '')
    
    # 如果不是 Krea AI Studio，移除 Krea 專用模型
    if current_provider != "FLUX Krea AI Studio":
        merged_models = {k: v for k, v in merged_models.items() 
                        if not v.get('provider_specific') == 'FLUX Krea AI Studio'}
    
    for model_id, model_info in discovered.items():
        if model_id not in merged_models:
            merged_models[model_id] = model_info
            
    # 按 'priority' 排序
    sorted_models = sorted(merged_models.items(), key=lambda item: item[1].get('priority', 999))
    return dict(sorted_models)


def validate_api_key(api_key: str, base_url: str, provider: str) -> Tuple[bool, str]:
    """驗證 API 密鑰是否有效，新增 FLUX Krea AI Studio 驗證"""
    try:
        if provider == "Pollinations.ai":
            test_url = f"{base_url}/models"
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                return True, "Pollinations.ai 服務連接成功"
            else:
                return False, f"HTTP {response.status_code}: Pollinations.ai 連接失敗"

        elif provider == "FLUX Krea AI Studio":
            # Krea AI Studio API 驗證
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            test_url = f"{base_url}/models"
            response = requests.get(test_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return True, "FLUX Krea AI Studio API 連接成功"
            elif response.status_code == 401:
                return False, "API 密鑰無效或已過期"
            else:
                return False, f"HTTP {response.status_code}: Krea AI Studio 連接失敗"

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
        
        if provider == "FLUX Krea AI Studio":
            # Krea AI Studio API 調用測試
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            data = {
                "prompt": test_prompt,
                "model": model_name,
                "width": 1024,
                "height": 1024,
                "steps": 20
            }
            
            # 發送測試請求 (不實際生成，只測試端點)
            test_url = f"{base_url}/generate"
            response = requests.post(test_url, headers=headers, json=data, timeout=10)
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code in [200, 202]:
                test_result.update({
                    'available': True,
                    'response_time': response_time,
                    'details': {
                        'status_code': response.status_code,
                        'test_prompt': test_prompt,
                        'provider': 'FLUX Krea AI Studio'
                    }
                })
            else:
                test_result.update({
                    'available': False,
                    'error': f"HTTP {response.status_code}",
                    'details': {'status_code': response.status_code}
                })
        
        elif provider == "Hugging Face":
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
    """帶重試機制的圖像生成，支援 Pollinations.ai 和 FLUX Krea AI Studio"""
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
                    "seed": random.randint(0, 1000000),
                    "nologo": "true"
                }
                
                # 清理 None 值
                query_params = {k: v for k, v in query_params.items() if v is not None}
                
                # 處理認證
                headers = {}
                config = st.session_state.get('api_config', {})
                auth_mode = config.get('pollinations_auth_mode', 'free')
                
                if auth_mode == 'token' and config.get('pollinations_token'):
                    headers['Authorization'] = f"Bearer {config['pollinations_token']}"
                elif auth_mode == 'referrer' and config.get('pollinations_referrer'):
                    headers['Referer'] = config['pollinations_referrer']
                
                encoded_prompt = quote(prompt)
                request_url = f"{base_url}/prompt/{encoded_prompt}?{urlencode(query_params)}"
                
                response = requests.get(request_url, headers=headers, timeout=120)

                if response.status_code == 200:
                    # 模擬 OpenAI 響應格式
                    class MockResponse:
                        def __init__(self, image_data):
                            self.data = [type('obj', (object,), {
                                'url': f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                            })()]
                    
                    return True, MockResponse(response.content)
                else:
                    error_text = response.text
                    if "Access to" in error_text and "is limited" in error_text:
                        return False, f"此模型需要認證。請在側邊欄配置 Pollinations.ai 認證信息。錯誤: {error_text}"
                    raise Exception(f"HTTP {response.status_code}: {error_text}")

            elif provider == "FLUX Krea AI Studio":
                # Krea AI Studio API 調用
                prompt = params.get("prompt", "")
                width, height = params.get("size", "1024x1024").split('x')
                
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                headers["Content-Type"] = "application/json"
                
                payload = {
                    "prompt": prompt,
                    "model": params.get("model", "krea-flux-aesthetic"),
                    "width": int(width),
                    "height": int(height),
                    "steps": 20,
                    "guidance_scale": 7.5,
                    "seed": random.randint(0, 1000000)
                }
                
                # Krea AI Studio 生成請求
                generate_url = f"{base_url}/generate"
                response = requests.post(generate_url, headers=headers, json=payload, timeout=120)
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # 處理響應格式 (根據實際 API 響應調整)
                    if 'images' in result_data:
                        images = result_data['images']
                    elif 'data' in result_data:
                        images = result_data['data']
                    else:
                        images = [result_data]
                    
                    # 模擬 OpenAI 響應格式
                    class MockResponse:
                        def __init__(self, images_data):
                            self.data = []
                            for img_data in images_data:
                                if isinstance(img_data, str):
                                    # 如果是 base64 字符串
                                    self.data.append(type('obj', (object,), {
                                        'url': f"data:image/png;base64,{img_data}"
                                    })())
                                elif isinstance(img_data, dict) and 'url' in img_data:
                                    # 如果是包含 URL 的字典
                                    self.data.append(type('obj', (object,), {
                                        'url': img_data['url']
                                    })())
                    
                    return True, MockResponse(images)
                else:
                    error_text = response.text
                    raise Exception(f"HTTP {response.status_code}: {error_text}")
            
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
            'validated': False,
            'pollinations_auth_mode': 'free',
            'pollinations_token': '',
            'pollinations_referrer': ''
        }
    
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'favorite_images' not in st.session_state:
        st.session_state.favorite_images = []
    
    if 'model_test_results' not in st.session_state:
        st.session_state.model_test_results = {}
    
    if 'discovered_models' not in st.session_state:
        st.session_state.discovered_models = {}
    
    # 新增：初始化模型更新標誌
    if 'models_updated' not in st.session_state:
        st.session_state.models_updated = False

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
    config = st.session_state.api_config
    if config.get('provider') in ["Hugging Face", "Pollinations.ai", "FLUX Krea AI Studio"]:
        return None
    if config.get('api_key'):
        try:
            return OpenAI(
                api_key=config['api_key'],
                base_url=config['base_url']
            )
        except Exception:
            return None
    return None

def show_api_settings():
    """顯示 API 設置界面 (增強版：包含註冊鏈接和 FLUX Krea AI Studio)"""
    st.subheader("🔑 API 設置")
    
    provider_options = list(API_PROVIDERS.keys())
    current_provider = st.session_state.api_config.get('provider', 'Navy')
    
    # 確保當前 provider 在選項中
    provider_index = provider_options.index(current_provider) if current_provider in provider_options else 0
    
    selected_provider = st.selectbox(
        "選擇 API 提供商",
        options=provider_options,
        index=provider_index,
        format_func=lambda x: f"{API_PROVIDERS[x]['icon']} {API_PROVIDERS[x]['name']}"
    )
    
    provider_info = API_PROVIDERS[selected_provider]
    st.info(f"📋 {provider_info['description']}")
    
    # 顯示 FLUX Krea AI Studio 特色功能
    if selected_provider == "FLUX Krea AI Studio":
        st.markdown("### 🎭 FLUX Krea AI Studio 特色")
        
        features = provider_info.get('features', [])
        if features:
            for feature in features:
                st.markdown(f"✨ **{feature}**")
        
        st.markdown("""
        **專業美學優勢：**
        - 🎨 **藝術級品質**: 專注於美學和藝術表現
        - 🖼️ **風格化控制**: 豐富的藝術風格選項
        - 🎯 **專業參數**: 精細的生成參數控制
        - 📐 **構圖優化**: 智能的構圖和色彩平衡
        """)
    
    # 顯示註冊和文檔鏈接
    if selected_provider != "Custom" and provider_info.get('register_url'):
        st.markdown("### 🔗 相關鏈接")
        
        col_links1, col_links2, col_links3 = st.columns(3)
        
        with col_links1:
            if provider_info.get('register_url'):
                st.markdown(f"[📝 註冊帳號]({provider_info['register_url']})")
        
        with col_links2:
            if provider_info.get('api_docs'):
                st.markdown(f"[📚 API 文檔]({provider_info['api_docs']})")
        
        with col_links3:
            if provider_info.get('pricing_url'):
                st.markdown(f"[💰 價格方案]({provider_info['pricing_url']})")
    
    # Pollinations.ai 特殊認證設置
    if selected_provider == "Pollinations.ai":
        st.markdown("### 🌸 Pollinations.ai 認證設置")
        
        # 特別說明
        st.info("💡 Pollinations.ai 提供三種使用方式：免費（基礎模型）、域名認證（推薦）、Token認證（高級功能）")
        
        auth_mode = st.radio(
            "選擇認證模式",
            options=["free", "referrer", "token"],
            format_func=lambda x: {
                "free": "🆓 免費模式（基礎模型）",
                "referrer": "🌐 域名認證（推薦）", 
                "token": "🔑 Token 認證（高級）"
            }[x],
            index=["free", "referrer", "token"].index(
                st.session_state.api_config.get('pollinations_auth_mode', 'free')
            )
        )
        
        if auth_mode == "referrer":
            st.info("✨ 輸入您的應用域名以存取更多模型（如 kontext）")
            referrer_input = st.text_input(
                "應用域名",
                value=st.session_state.api_config.get('pollinations_referrer', ''),
                placeholder="例如：myapp.vercel.app 或 username.github.io",
                help="輸入您部署應用的域名"
            )
            st.caption("💡 Koyeb 部署示例：yourapp-yourname.koyeb.app")
            
        elif auth_mode == "token":
            st.info("🔐 使用 Token 進行後端認證，適合服務端整合")
            st.markdown(f"➡️ [在此獲取您的 Token]({provider_info['register_url']})")
            
            token_input = st.text_input(
                "Pollinations Token",
                value="",
                type="password",
                placeholder="請從 https://auth.pollinations.ai 獲取您的 token",
                help="獲取 token 後可使用所有高級模型"
            )
            current_token = st.session_state.api_config.get('pollinations_token', '')
            if current_token and not token_input:
                st.caption(f"🔐 當前 Token: {current_token[:10]}...{current_token[-8:] if len(current_token) > 18 else ''}")
        else:
            st.success("🆓 免費模式：無需註冊，但只能使用基礎模型")
            st.caption("如需使用高級模型（如 kontext），請選擇域名認證或 Token 認證")
    
    # API 密鑰設置
    is_key_required = selected_provider not in ["Pollinations.ai"]
    
    api_key_input = ""
    current_key = st.session_state.api_config.get('api_key', '')
    
    if is_key_required:
        st.markdown("### 🔑 API 密鑰")
        
        # 如果沒有 API 密鑰，顯示獲取提示
        if not current_key:
            st.warning(f"⚠️ 需要 {provider_info['name']} API 密鑰")
            if provider_info.get('register_url'):
                st.markdown(f"👉 [點此註冊並獲取 API 密鑰]({provider_info['register_url']})")
        
        # 特別提示 FLUX Krea AI Studio
        if selected_provider == "FLUX Krea AI Studio":
            st.info("🎭 FLUX Krea AI Studio 需要專業帳戶來存取高品質美學模型")
        
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
    else:
        if selected_provider != "Pollinations.ai":
            st.success("✅ 此提供商無需 API 密鑰。")
        current_key = "N/A"

    # 處理 Base URL 變化
    if selected_provider != current_provider:
        base_url_value = provider_info['base_url_default']
    else:
        base_url_value = st.session_state.api_config.get('base_url', provider_info['base_url_default'])

    base_url_input = st.text_input(
        "API 端點 URL",
        value=base_url_value,
        placeholder=provider_info['base_url_default'],
        help="API 服務的基礎 URL"
    )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        save_btn = st.button("💾 保存設置", type="primary")
    
    with col2:
        test_btn = st.button("🧪 測試連接")
    
    with col3:
        clear_btn = st.button("🗑️ 清除設置", type="secondary")
    
    if save_btn:
        final_api_key = api_key_input if api_key_input else current_key
        if is_key_required and not final_api_key:
            st.error("❌ 請輸入 API 密鑰")
            if provider_info.get('register_url'):
                st.info(f"💡 尚未註冊？[點此註冊]({provider_info['register_url']})")
        elif not base_url_input:
            st.error("❌ 請輸入 API 端點 URL")
        else:
            config_update = {
                'provider': selected_provider,
                'api_key': final_api_key,
                'base_url': base_url_input,
                'validated': False
            }
            
            # Pollinations.ai 特殊設置
            if selected_provider == "Pollinations.ai":
                config_update['pollinations_auth_mode'] = auth_mode
                if auth_mode == "referrer":
                    config_update['pollinations_referrer'] = referrer_input
                elif auth_mode == "token":
                    config_update['pollinations_token'] = token_input if token_input else st.session_state.api_config.get('pollinations_token', '')
            
            st.session_state.api_config.update(config_update)
            # 清除舊的發現模型和選擇的模型
            st.session_state.discovered_models = {}
            if 'selected_model' in st.session_state:
                del st.session_state.selected_model
            st.session_state.models_updated = True
            st.success("✅ API 設置已保存，模型列表已重置。")
            time.sleep(0.5)  # 給用戶時間看到成功消息
            rerun_app()
    
    if test_btn:
        test_api_key = api_key_input if api_key_input else current_key
        if is_key_required and not test_api_key:
            st.error("❌ 請先輸入 API 密鑰")
            if provider_info.get('register_url'):
                st.info(f"💡 [點此獲取 API 密鑰]({provider_info['register_url']})")
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
                    if "密鑰" in message and provider_info.get('register_url'):
                        st.info(f"💡 需要有效的API密鑰？[點此註冊]({provider_info['register_url']})")
    
    if clear_btn:
        st.session_state.api_config = {
            'provider': 'Navy',
            'api_key': '',
            'base_url': 'https://api.navy/v1',
            'validated': False,
            'pollinations_auth_mode': 'free',
            'pollinations_token': '',
            'pollinations_referrer': ''
        }
        st.session_state.discovered_models = {}
        if 'selected_model' in st.session_state:
            del st.session_state.selected_model
        st.session_state.models_updated = True
        st.success("🗑️ API 設置已清除，模型列表已重置。")
        time.sleep(0.5)  # 給用戶時間看到成功消息
        rerun_app()


def auto_discover_models():
    """執行自動模型發現 (不更新頁面)"""
    config = st.session_state.api_config
    provider = config.get('provider')
    is_key_required = provider not in ["Pollinations.ai"]
    
    if is_key_required and not config.get('api_key'):
        st.error("❌ 請先配置 API 密鑰")
        # 顯示註冊鏈接
        provider_info = API_PROVIDERS.get(provider, {})
        if provider_info.get('register_url'):
            st.info(f"💡 [點此註冊並獲取 API 密鑰]({provider_info['register_url']})")
        return
    
    #
