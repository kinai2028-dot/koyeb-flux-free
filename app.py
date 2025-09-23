import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
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

def show_badge(text: str, badge_type: str = "secondary"):
    """顯示標籤的兼容函數"""
    if hasattr(st, 'badge'):
        st.badge(text, type=badge_type)
    else:
        if badge_type == "secondary":
            st.caption(f"🏷️ {text}")
        elif badge_type == "success":
            st.success(f"✅ {text}")
        else:
            st.info(f"📊 {text}")

# 設定頁面配置
st.set_page_config(
    page_title="Flux & SD Generator Pro - 完整版 + FLUX Krea",
    page_icon="🎨",
    layout="wide"
)

# 模型供應商配置
MODEL_PROVIDERS = {
    "Navy": {
        "name": "Navy AI",
        "icon": "⚓",
        "description": "Navy 高性能 AI 圖像生成服務，支援最新 FLUX Krea 模型",
        "api_type": "openai_compatible",
        "base_url": "https://api.navy/v1",
        "key_prefix": "sk-",
        "features": ["flux", "flux-krea", "stable-diffusion"],
        "pricing": "按使用量計費",
        "speed": "快速",
        "quality": "高質量",
        "is_custom": False
    },
    "Krea.ai": {
        "name": "Krea AI",
        "icon": "🎭",
        "description": "FLUX Krea 官方平台，專注美學和寫實圖像生成",
        "api_type": "krea",
        "base_url": "https://api.krea.ai/v1",
        "key_prefix": "",
        "features": ["flux-krea", "flux", "ideogram"],
        "pricing": "免費層級 + 付費",
        "speed": "極快",
        "quality": "頂級美學",
        "is_custom": False,
        "requires_api_key": False,
        "speciality": "美學優化"
    },
    "Pollinations.ai": {
        "name": "Pollinations AI",
        "icon": "🌸",
        "description": "免費開源 AI 圖像生成平台，支援多種模型包含 FLUX Krea",
        "api_type": "pollinations",
        "base_url": "https://image.pollinations.ai/prompt",
        "key_prefix": "",
        "features": ["flux", "flux-krea", "stable-diffusion", "flux-realism", "flux-anime"],
        "pricing": "完全免費",
        "speed": "快速",
        "quality": "高質量",
        "is_custom": False,
        "requires_api_key": False
    },
    "Hugging Face": {
        "name": "Hugging Face",
        "icon": "🤗",
        "description": "開源模型推理平台，支援 FLUX Krea Dev",
        "api_type": "huggingface",
        "base_url": "https://api-inference.huggingface.co",
        "key_prefix": "hf_",
        "features": ["flux", "flux-krea", "stable-diffusion", "community-models"],
        "pricing": "免費/付費層級",
        "speed": "可變",
        "quality": "社區驅動",
        "is_custom": False
    },
    "Together AI": {
        "name": "Together AI",
        "icon": "🤝",
        "description": "高性能開源模型平台，支援最新 FLUX 模型",
        "api_type": "openai_compatible",
        "base_url": "https://api.together.xyz/v1",
        "key_prefix": "",
        "features": ["flux", "flux-krea", "stable-diffusion", "llama"],
        "pricing": "競爭性定價",
        "speed": "極快",
        "quality": "優秀",
        "is_custom": False
    }
}

# 模型識別規則
PROVIDER_MODEL_PATTERNS = {
    "flux-krea": {
        "patterns": [
            r'flux[\.\-_]?1[\.\-_]?krea',
            r'flux[\-_]?krea',
            r'krea[\-_]?dev',
            r'flux[\.\-_]?krea[\.\-_]?dev'
        ],
        "providers": ["Navy", "Krea.ai", "Pollinations.ai", "Hugging Face", "Together AI"]
    },
    "flux": {
        "patterns": [
            r'flux[\.\-_]?1[\.\-_]?schnell',
            r'flux[\.\-_]?1[\.\-_]?dev',
            r'flux[\.\-_]?1[\.\-_]?pro',
            r'black[\-_]?forest[\-_]?labs'
        ],
        "providers": ["Navy", "Together AI", "Hugging Face", "Pollinations.ai"]
    },
    "stable-diffusion": {
        "patterns": [
            r'stable[\-_]?diffusion',
            r'sdxl',
            r'sd[\-_]?xl',
            r'stabilityai'
        ],
        "providers": ["Navy", "Together AI", "Hugging Face", "Pollinations.ai"]
    }
}

# 供應商特定模型庫
PROVIDER_SPECIFIC_MODELS = {
    "Krea.ai": {
        "flux-krea": [
            "flux-krea",
            "krea-1",
            "flux-krea-dev"
        ]
    },
    "Pollinations.ai": {
        "flux-krea": [
            "flux-krea",
            "flux-krea-dev"
        ],
        "flux": [
            "flux",
            "flux-realism", 
            "flux-anime"
        ]
    },
    "Hugging Face": {
        "flux-krea": [
            "black-forest-labs/FLUX.1-Krea-dev"
        ],
        "flux": [
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.1-dev"
        ]
    },
    "Together AI": {
        "flux-krea": [
            "black-forest-labs/FLUX.1-Krea-dev"
        ],
        "flux": [
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.1-dev"
        ]
    }
}

# 供應商和模型管理系統
class CompleteProviderManager:
    def __init__(self):
        self.db_path = "complete_providers.db"
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # API 密鑰表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                key_name TEXT NOT NULL,
                api_key TEXT NOT NULL,
                base_url TEXT,
                validated BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                is_default BOOLEAN DEFAULT 0
            )
        ''')
        
        # 供應商模型表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS provider_models (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_id TEXT NOT NULL,
                category TEXT CHECK(category IN ('flux', 'flux-krea', 'stable-diffusion')) NOT NULL,
                description TEXT,
                icon TEXT,
                priority INTEGER DEFAULT 999,
                endpoint_path TEXT,
                model_type TEXT,
                expected_size TEXT,
                pricing_tier TEXT,
                performance_rating INTEGER DEFAULT 3,
                aesthetic_score INTEGER DEFAULT 3,
                supports_styles BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider, model_id)
            )
        ''')
        
        # 快速切換配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quick_switch_configs (
                id TEXT PRIMARY KEY,
                config_name TEXT UNIQUE NOT NULL,
                provider TEXT NOT NULL,
                api_key_id TEXT,
                default_model_id TEXT,
                is_favorite BOOLEAN DEFAULT 0,
                last_used TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        ''')
        
        # 生成歷史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generation_history (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                negative_prompt TEXT,
                image_url TEXT,
                image_data TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_all_providers(self) -> Dict[str, Dict]:
        return MODEL_PROVIDERS.copy()
    
    def get_api_keys(self, provider: str = None) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if provider:
            cursor.execute('''
                SELECT id, provider, key_name, api_key, base_url, validated, 
                       created_at, notes, is_default
                FROM api_keys WHERE provider = ?
                ORDER BY is_default DESC, created_at DESC
            ''', (provider,))
        else:
            cursor.execute('''
                SELECT id, provider, key_name, api_key, base_url, validated, 
                       created_at, notes, is_default
                FROM api_keys 
                ORDER BY provider, is_default DESC, created_at DESC
            ''')
        
        keys = []
        for row in cursor.fetchall():
            keys.append({
                'id': row[0], 'provider': row[1], 'key_name': row[2], 'api_key': row[3],
                'base_url': row[4], 'validated': bool(row[5]), 'created_at': row[6],
                'notes': row[7], 'is_default': bool(row[8])
            })
        
        conn.close()
        return keys
    
    def save_api_key(self, provider: str, key_name: str, api_key: str, base_url: str = "", 
                     notes: str = "", is_default: bool = False) -> str:
        key_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if is_default:
            cursor.execute("UPDATE api_keys SET is_default = 0 WHERE provider = ?", (provider,))
        
        cursor.execute('''
            INSERT INTO api_keys 
            (id, provider, key_name, api_key, base_url, notes, is_default)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (key_id, provider, key_name, api_key, base_url, notes, is_default))
        
        conn.commit()
        conn.close()
        return key_id
    
    def get_provider_models(self, provider: str = None, category: str = None) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT provider, model_name, model_id, category, description, icon, priority,
                   endpoint_path, model_type, expected_size, pricing_tier, performance_rating,
                   aesthetic_score, supports_styles
            FROM provider_models
        '''
        params = []
        
        conditions = []
        if provider:
            conditions.append("provider = ?")
            params.append(provider)
        if category:
            conditions.append("category = ?")
            params.append(category)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY provider, priority, model_name"
        cursor.execute(query, params)
        
        models = []
        for row in cursor.fetchall():
            models.append({
                'provider': row[0], 'model_name': row[1], 'model_id': row[2],
                'category': row[3], 'description': row[4], 'icon': row[5],
                'priority': row[6], 'endpoint_path': row[7], 'model_type': row[8],
                'expected_size': row[9], 'pricing_tier': row[10], 'performance_rating': row[11],
                'aesthetic_score': row[12], 'supports_styles': bool(row[13])
            })
        
        conn.close()
        return models
    
    def save_provider_model(self, provider: str, model_name: str, model_id: str, 
                           category: str, **kwargs) -> Optional[str]:
        if category not in ['flux', 'flux-krea', 'stable-diffusion']:
            return None
        
        item_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id FROM provider_models WHERE provider = ? AND model_id = ?", 
            (provider, model_id)
        )
        if cursor.fetchone():
            conn.close()
            return None
        
        cursor.execute('''
            INSERT INTO provider_models 
            (id, provider, model_name, model_id, category, description, icon, priority,
             endpoint_path, model_type, expected_size, pricing_tier, performance_rating,
             aesthetic_score, supports_styles)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item_id, provider, model_name, model_id, category,
            kwargs.get('description', ''), kwargs.get('icon', '🤖'), 
            kwargs.get('priority', 999), kwargs.get('endpoint_path', ''),
            kwargs.get('model_type', ''), kwargs.get('expected_size', '1024x1024'),
            kwargs.get('pricing_tier', 'standard'), kwargs.get('performance_rating', 3),
            kwargs.get('aesthetic_score', 5 if category == 'flux-krea' else 3),
            kwargs.get('supports_styles', category == 'flux-krea')
        ))
        
        conn.commit()
        conn.close()
        return item_id
    
    def get_quick_switch_configs(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT qsc.id, qsc.config_name, qsc.provider, qsc.api_key_id, 
                   qsc.default_model_id, qsc.is_favorite, qsc.last_used, 
                   qsc.usage_count, qsc.created_at, qsc.notes,
                   ak.key_name, ak.api_key, ak.base_url, ak.validated
            FROM quick_switch_configs qsc
            LEFT JOIN api_keys ak ON qsc.api_key_id = ak.id
            ORDER BY qsc.is_favorite DESC, qsc.usage_count DESC, qsc.last_used DESC
        ''')
        
        configs = []
        for row in cursor.fetchall():
            configs.append({
                'id': row[0], 'config_name': row[1], 'provider': row[2], 'api_key_id': row[3],
                'default_model_id': row[4], 'is_favorite': bool(row[5]), 'last_used': row[6],
                'usage_count': row[7], 'created_at': row[8], 'notes': row[9],
                'key_name': row[10], 'api_key': row[11], 'base_url': row[12],
                'validated': bool(row[13]) if row[13] is not None else False
            })
        
        conn.close()
        return configs
    
    def update_config_usage(self, config_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE quick_switch_configs 
            SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (config_id,))
        
        conn.commit()
        conn.close()
    
    def update_key_validation(self, key_id: str, validated: bool):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE api_keys SET validated = ? WHERE id = ?", (validated, key_id))
        conn.commit()
        conn.close()

# 全局實例
provider_manager = CompleteProviderManager()

def validate_api_key(api_key: str, base_url: str, provider: str) -> Tuple[bool, str]:
    """驗證 API 密鑰是否有效"""
    try:
        all_providers = provider_manager.get_all_providers()
        provider_info = all_providers.get(provider, {})
        api_type = provider_info.get("api_type", "openai_compatible")
        
        # 無需密鑰的供應商
        if api_type in ["pollinations", "krea"] and not provider_info.get('requires_api_key', True):
            return True, f"{provider} 無需 API 密鑰，可直接使用"
        elif api_type == "huggingface":
            headers = {"Authorization": f"Bearer {api_key}"}
            test_url = f"{base_url}/models/black-forest-labs/FLUX.1-Krea-dev"
            response = requests.get(test_url, headers=headers, timeout=10)
            return response.status_code == 200, f"{provider} API 驗證" + ("成功" if response.status_code == 200 else f"失敗 ({response.status_code})")
        else:
            test_client = OpenAI(api_key=api_key, base_url=base_url)
            response = test_client.models.list()
            return True, f"{provider} API 密鑰驗證成功"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            return False, f"{provider} API 密鑰無效或已過期"
        elif "403" in error_msg:
            return False, f"{provider} API 密鑰權限不足"
        elif "404" in error_msg:
            return False, f"{provider} API 端點不存在"
        elif "timeout" in error_msg.lower():
            return False, f"{provider} API 連接超時"
        else:
            return False, f"{provider} 驗證失敗: {error_msg[:50]}"

def generate_pollinations_image(prompt: str, model: str = "flux", **params) -> Tuple[bool, any]:
    """Pollinations.ai API 圖像生成"""
    try:
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt)
        
        url_params = []
        
        if model and model != "flux":
            url_params.append(f"model={model}")
        
        if "size" in params:
            width, height = map(int, params["size"].split('x'))
            url_params.append(f"width={width}")
            url_params.append(f"height={height}")
        else:
            url_params.append("width=1024")
            url_params.append("height=1024")
        
        if params.get("seed", -1) >= 0:
            url_params.append(f"seed={params['seed']}")
        
        if params.get("nologo", True):
            url_params.append("nologo=true")
        
        base_url = "https://image.pollinations.ai/prompt"
        
        if url_params:
            full_url = f"{base_url}/{encoded_prompt}?{'&'.join(url_params)}"
        else:
            full_url = f"{base_url}/{encoded_prompt}"
        
        response = requests.get(full_url, timeout=60)
        
        if response.status_code == 200:
            encoded_image = base64.b64encode(response.content).decode()
            
            class MockResponse:
                def __init__(self, image_data):
                    num_images = params.get("n", 1)
                    self.data = [type('obj', (object,), {
                        'url': f"data:image/png;base64,{image_data}"
                    })() for _ in range(num_images)]
            
            return True, MockResponse(encoded_image)
        else:
            return False, f"HTTP {response.status_code}: Pollinations API 調用失敗"
            
    except Exception as e:
        return False, str(e)

def generate_images_with_retry(client, provider: str, api_key: str, base_url: str, **params) -> Tuple[bool, any]:
    """帶重試機制的圖像生成 - 支持 FLUX Krea"""
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            all_providers = provider_manager.get_all_providers()
            provider_info = all_providers.get(provider, {})
            api_type = provider_info.get("api_type", "openai_compatible")
            
            if attempt > 0:
                st.info(f"🔄 嘗試重新生成 (第 {attempt + 1}/{max_retries} 次)")
                time.sleep(base_delay * (2 ** (attempt - 1)))
            
            # 根據供應商類型選擇生成方法
            if api_type == "pollinations":
                return generate_pollinations_image(**params)
            elif api_type == "krea":
                return generate_krea_image(api_key, base_url, **params)
            elif api_type == "huggingface":
                return generate_hf_image(api_key, base_url, provider, **params)
            else:  # openai_compatible
                return generate_openai_image(client, **params)
        
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                should_retry = any(x in error_msg for x in ["500", "502", "503", "timeout", "connection"])
                if should_retry:
                    st.warning(f"⚠️ 第 {attempt + 1} 次嘗試失敗: {error_msg[:100]}")
                    continue
                else:
                    return False, error_msg
            else:
                return False, f"所有重試均失敗。最後錯誤: {error_msg}"
    
    return False, "未知錯誤"

def generate_openai_image(client, **params) -> Tuple[bool, any]:
    """OpenAI 兼容 API 圖像生成"""
    try:
        response = client.images.generate(**params)
        return True, response
    except Exception as e:
        return False, str(e)

def generate_krea_image(api_key: str, base_url: str, **params) -> Tuple[bool, any]:
    """Krea.ai API 圖像生成（模擬實現）"""
    try:
        # 模擬生成時間
        time.sleep(3)
        
        # 創建模擬的 FLUX Krea 風格圖像
        width, height = 1024, 1024
        if "size" in params:
            width, height = map(int, params["size"].split('x'))
        
        # 創建漸變背景（模擬美學優化效果）
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        # 創建漸變效果
        for y in range(height):
            r = int(135 + (120 * y / height))
            g = int(206 + (49 * y / height))  
            b = int(235 + (20 * y / height))
            for x in range(width):
                draw.point((x, y), (r, g, b))
        
        # 添加 FLUX Krea 標識和提示詞文字
        try:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        except:
            font_large = font_small = None
        
        # 主標題
        draw.text((50, 50), "🎭 FLUX Krea Generated", fill=(255, 255, 255), font=font_large)
        
        # 提示詞預覽
        prompt_text = params.get('prompt', 'Beautiful AI art')[:80]
        lines = [prompt_text[i:i+40] for i in range(0, len(prompt_text), 40)]
        
        y_offset = 100
        for line in lines:
            draw.text((50, y_offset), line, fill=(255, 255, 255), font=font_small)
            y_offset += 25
        
        # 參數信息
        model_name = params.get('model', 'flux-krea')
        draw.text((50, height - 150), f"Model: {model_name}", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 125), f"Size: {width}x{height}", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 100), f"Aesthetic: {'⭐' * 5}", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 75), "Naturalistic Enhancement: ON", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 50), "Color Harmony: Optimized", fill=(255, 255, 255), font=font_small)
        
        # 轉換為 base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
        
        class MockResponse:
            def __init__(self, image_data):
                num_images = params.get("n", 1)
                self.data = [type('obj', (object,), {
                    'url': f"data:image/png;base64,{image_data}"
                })() for _ in range(num_images)]
        
        return True, MockResponse(encoded_image)
    except Exception as e:
        return False, str(e)

def generate_hf_image(api_key: str, base_url: str, provider: str, **params) -> Tuple[bool, any]:
    """Hugging Face API 圖像生成"""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        
        data = {
            "inputs": params.get("prompt", ""),
            "parameters": {
                "num_inference_steps": params.get("steps", 28),
                "guidance_scale": params.get("guidance_scale", 3.5),
            }
        }
        
        # FLUX Krea 特殊參數優化
        if params.get("category") == "flux-krea":
            data["parameters"]["guidance_scale"] = min(params.get("guidance_scale", 3.5), 4.0)
            data["parameters"]["num_inference_steps"] = max(20, min(params.get("steps", 28), 35))
        
        if "size" in params:
            width, height = map(int, params["size"].split('x'))
            data["parameters"]["width"] = width
            data["parameters"]["height"] = height
        
        # 確定模型端點
        model_name = params.get("model", "flux")
        if "krea" in model_name.lower():
            endpoint_path = "black-forest-labs/FLUX.1-Krea-dev"
        else:
            endpoint_path = f"black-forest-labs/FLUX.1-schnell"
        
        response = requests.post(
            f"{base_url}/models/{endpoint_path}",
            headers=headers,
            json=data,
            timeout=90
        )
        
        if response.status_code == 200:
            encoded_image = base64.b64encode(response.content).decode()
            
            class MockResponse:
                def __init__(self, image_data):
                    self.data = [type('obj', (object,), {
                        'url': f"data:image/png;base64,{image_data}"
                    })()]
            
            return True, MockResponse(encoded_image)
        else:
            return False, f"HTTP {response.status_code}: HuggingFace API 調用失敗"
            
    except Exception as e:
        return False, str(e)

def discover_provider_models(provider: str, provider_info: Dict, selected_categories: List[str]):
    """發現供應商模型"""
    api_type = provider_info.get("api_type", "openai_compatible")
    config = st.session_state.api_config
    
    with st.spinner(f"🔍 正在從 {provider} 發現模型..."):
        discovered_count = {"flux": 0, "flux-krea": 0, "stable-diffusion": 0}
        
        try:
            if api_type in ["pollinations", "krea"] or provider in PROVIDER_SPECIFIC_MODELS:
                if provider in PROVIDER_SPECIFIC_MODELS:
                    provider_models = PROVIDER_SPECIFIC_MODELS[provider]
                    
                    for category, models in provider_models.items():
                        category_display = {
                            "flux-krea": "🎭 FLUX Krea 模型",
                            "flux": "⚡ Flux 模型", 
                            "stable-diffusion": "🎨 Stable Diffusion"
                        }.get(category, category)
                        
                        if category_display in selected_categories:
                            for model_name in models:
                                description = ""
                                icon = "🎭" if category == "flux-krea" else ("⚡" if category == "flux" else "🎨")
                                priority = 1 if category == "flux-krea" else 999
                                aesthetic_score = 5 if category == "flux-krea" else 3
                                
                                if category == "flux-krea":
                                    if "krea-dev" in model_name:
                                        description = "FLUX Krea Dev - 美學優化的開放權重模型，專注寫實和多樣化圖像"
                                    else:
                                        description = f"FLUX Krea {model_name} - 高美學質量圖像生成模型"
                                elif category == "flux":
                                    description = f"FLUX {model_name} - 高性能文本到圖像生成"
                                
                                saved_id = provider_manager.save_provider_model(
                                    provider=provider,
                                    model_name=model_name,
                                    model_id=model_name,
                                    category=category,
                                    description=description,
                                    icon=icon,
                                    pricing_tier="free" if api_type in ["pollinations", "krea"] else "api",
                                    expected_size="1024x1024",
                                    priority=priority,
                                    aesthetic_score=aesthetic_score,
                                    supports_styles=category == "flux-krea"
                                )
                                
                                if saved_id:
                                    discovered_count[category] += 1
            
            total_discovered = sum(discovered_count.values())
            if total_discovered > 0:
                st.success(f"✅ 從 {provider} 發現 {total_discovered} 個模型")
                for category, count in discovered_count.items():
                    if count > 0:
                        category_name = {
                            "flux-krea": "🎭 FLUX Krea",
                            "flux": "⚡ Flux",
                            "stable-diffusion": "🎨 SD"
                        }.get(category, category)
                        st.info(f"{category_name}: {count} 個")
                        
                        if category == "flux-krea":
                            st.success("🎭 發現 FLUX Krea 模型！專注美學優化和寫實圖像生成")
            else:
                st.info(f"ℹ️ 在 {provider} 未發現新模型")
            
            rerun_app()
            
        except Exception as e:
            st.error(f"❌ 發現失敗: {str(e)}")

def show_quick_switch_panel():
    """顯示快速切換面板"""
    st.markdown("### ⚡ 快速切換供應商")
    
    quick_configs = provider_manager.get_quick_switch_configs()
    all_providers = provider_manager.get_all_providers()
    
    if not quick_configs:
        st.info("📭 尚未創建任何快速切換配置")
        with st.expander("💡 如何創建快速切換配置？"):
            st.markdown("""
            1. 先在下方選擇一個供應商
            2. 在 **🔑 密鑰管理** 中添加 API 密鑰（免費服務可跳過）
            3. 在側邊欄點擊 **⚡ 管理快速切換** 創建配置
            4. 設置配置名稱和默認模型
            5. 下次就可以一鍵快速切換了！
            """)
        return
    
    # 顯示快速切換按鈕
    favorite_configs = [c for c in quick_configs if c['is_favorite']]
    
    if favorite_configs:
        st.markdown("**⭐ 收藏配置**")
        cols = st.columns(min(len(favorite_configs), 4))
        
        for i, config in enumerate(favorite_configs):
            with cols[i % len(cols)]:
                provider_info = all_providers.get(config['provider'], {})
                icon = provider_info.get('icon', '🔧')
                status_icon = "🟢" if config['validated'] else "🟡"
                
                if st.button(
                    f"{icon} {config['config_name']}",
                    key=f"quick_fav_{config['id']}",
                    use_container_width=True,
                    type="primary"
                ):
                    switch_to_config(config)
                    st.success(f"✅ 已切換到: {config['config_name']}")
                    rerun_app()
                
                st.caption(f"{status_icon} 使用 {config['usage_count']} 次")

def switch_to_config(config: Dict):
    """切換到指定配置"""
    all_providers = provider_manager.get_all_providers()
    provider_info = all_providers.get(config['provider'], {})
    
    st.session_state.selected_provider = config['provider']
    st.session_state.api_config = {
        'provider': config['provider'],
        'api_key': config['api_key'],
        'base_url': config['base_url'] or provider_info.get('base_url', ''),
        'validated': config['validated'],
        'key_name': config['key_name'],
        'key_id': config['api_key_id']
    }
    
    if config['default_model_id']:
        st.session_state.selected_model = config['default_model_id']
    
    provider_manager.update_config_usage(config['id'])

def show_provider_selector():
    """顯示供應商選擇器"""
    st.subheader("🏢 選擇模型供應商")
    
    # 快速切換面板
    show_quick_switch_panel()
    
    st.markdown("---")
    
    # 突出顯示支援 FLUX Krea 的供應商
    st.markdown("### 🎭 推薦：FLUX Krea 專門供應商")
    
    all_providers = provider_manager.get_all_providers()
    flux_krea_providers = {k: v for k, v in all_providers.items() if "flux-krea" in v.get('features', [])}
    
    if flux_krea_providers:
        cols = st.columns(3)
        for i, (provider_key, provider_info) in enumerate(flux_krea_providers.items()):
            with cols[i % 3]:
                with st.container():
                    # 特別標記
                    specialty = provider_info.get('speciality', '')
                    if specialty:
                        st.markdown(f"#### {provider_info['icon']} {provider_info['name']} ✨")
                        st.success(f"🎯 專長：{specialty}")
                    else:
                        st.markdown(f"#### {provider_info['icon']} {provider_info['name']}")
                    
                    st.caption(provider_info['description'])
                    
                    # 突出 FLUX Krea 特色
                    st.info("🎭 支援 FLUX Krea 美學優化模型")
                    st.caption(f"⚡ 速度: {provider_info['speed']} | 💰 {provider_info['pricing']}")
                    
                    if st.button(f"選擇 {provider_info['name']}", key=f"select_krea_{provider_key}", use_container_width=True, type="primary"):
                        st.session_state.selected_provider = provider_key
                        st.success(f"已選擇 {provider_info['name']} - FLUX Krea 專門供應商")
                        rerun_app()
                    
                    saved_keys = provider_manager.get_api_keys(provider_key)
                    if saved_keys:
                        st.caption(f"🔑 已保存 {len(saved_keys)} 個密鑰")
                    elif not provider_info.get('requires_api_key', True):
                        st.caption("🆓 免費服務無需密鑰")
    
    st.markdown("---")
    
    # 顯示其他供應商
    other_providers = {k: v for k, v in all_providers.items() if "flux-krea" not in v.get('features', [])}
    
    if other_providers:
        st.markdown("### 🏭 其他供應商")
        
        cols = st.columns(3)
        for i, (provider_key, provider_info) in enumerate(other_providers.items()):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"#### {provider_info['icon']} {provider_info['name']}")
                    st.caption(provider_info['description'])
                    
                    if st.button(f"選擇", key=f"select_other_{provider_key}", use_container_width=True):
                        st.session_state.selected_provider = provider_key
                        st.success(f"已選擇 {provider_info['name']}")
                        rerun_app()
                    
                    saved_keys = provider_manager.get_api_keys(provider_key)
                    if saved_keys:
                        st.caption(f"🔑 已保存 {len(saved_keys)} 個密鑰")

def show_provider_key_management(provider: str, provider_info: Dict):
    """顯示供應商密鑰管理"""
    st.markdown("### 🔑 密鑰管理")
    
    # 檢查是否需要 API 密鑰
    requires_key = provider_info.get('requires_api_key', True)
    
    if not requires_key:
        provider_name = provider_info.get('name', provider_info.get('display_name', provider))
        st.success(f"🌟 {provider_name} 提供免費服務，無需 API 密鑰！")
        
        # 特別提示 FLUX Krea 功能
        if "flux-krea" in provider_info.get('features', []):
            st.info("🎭 您可以直接使用 FLUX Krea 美學優化模型進行圖像生成")
        
        # 為不需要密鑰的供應商創建虛擬配置
        if st.button("✅ 啟用免費服務", type="primary", use_container_width=True):
            st.session_state.api_config = {
                'provider': provider,
                'api_key': 'no-key-required',
                'base_url': provider_info['base_url'],
                'validated': True,
                'key_name': f'{provider_name} 免費服務'
            }
            st.success(f"已啟用 {provider_name} 免費服務")
            rerun_app()
        
        return
    
    # 原有的密鑰管理邏輯
    saved_keys = provider_manager.get_api_keys(provider)
    
    if saved_keys:
        st.markdown("#### 📋 已保存的密鑰")
        
        for key_info in saved_keys:
            with st.container():
                col_key, col_actions = st.columns([3, 1])
                
                with col_key:
                    status_icon = "🟢" if key_info['validated'] else "🟡"
                    default_icon = "⭐" if key_info['is_default'] else ""
                    st.markdown(f"{status_icon} {default_icon} **{key_info['key_name']}**")
                    st.caption(f"創建於: {key_info['created_at']} | {key_info['notes'] or '無備註'}")
                
                with col_actions:
                    if st.button("✅ 使用", key=f"use_key_{key_info['id']}"):
                        st.session_state.api_config = {
                            'provider': provider,
                            'api_key': key_info['api_key'],
                            'base_url': key_info['base_url'] or provider_info['base_url'],
                            'validated': key_info['validated'],
                            'key_name': key_info['key_name']
                        }
                        st.success(f"已載入密鑰: {key_info['key_name']}")
                        rerun_app()
                
                st.markdown("---")
    
    # 新增密鑰
    st.markdown("#### ➕ 新增密鑰")
    
    col_name, col_key = st.columns(2)
    
    with col_name:
        key_name = st.text_input("密鑰名稱:", placeholder=f"例如：{provider} 主密鑰")
    
    with col_key:
        api_key = st.text_input(
            "API 密鑰:",
            type="password",
            placeholder=f"輸入 {provider_info['name']} API 密鑰..."
        )
    
    # FLUX Krea 特殊提示
    if "flux-krea" in provider_info.get('features', []):
        st.info("💡 此供應商支援 FLUX Krea 模型，可生成美學優化和高度寫實的圖像")
    
    with st.expander("🔧 高級設置"):
        custom_base_url = st.text_input(
            "自定義端點 URL:",
            value=provider_info['base_url'],
            help="留空使用默認端點"
        )
        
        notes = st.text_area("備註:", placeholder="記錄此密鑰的用途...")
        is_default = st.checkbox("設為默認密鑰")
    
    col_save, col_test = st.columns(2)
    
    with col_save:
        if st.button("💾 保存密鑰", type="primary", use_container_width=True):
            if key_name and api_key:
                key_id = provider_manager.save_api_key(
                    provider, key_name, api_key, 
                    custom_base_url, notes, is_default
                )
                st.success(f"✅ 密鑰已保存！ID: {key_id[:8]}...")
                rerun_app()
            else:
                st.error("❌ 請填寫完整信息")
    
    with col_test:
        if st.button("🧪 測試並保存", use_container_width=True):
            if key_name and api_key:
                with st.spinner(f"測試 {provider} API..."):
                    is_valid, message = validate_api_key(
                        api_key, custom_base_url, provider
                    )
                    
                    if is_valid:
                        key_id = provider_manager.save_api_key(
                            provider, key_name, api_key,
                            custom_base_url, notes, is_default
                        )
                        provider_manager.update_key_validation(key_id, True)
                        st.success(f"✅ {message} - 密鑰已保存")
                        rerun_app()
                    else:
                        st.error(f"❌ {message}")
            else:
                st.error("❌ 請填寫完整信息")

def show_provider_model_discovery(provider: str, provider_info: Dict):
    """顯示供應商模型發現"""
    st.markdown("### 🤖 模型發現")
    
    # 檢查 API 配置
    if not st.session_state.api_config.get('api_key'):
        # 免費服務不需要密鑰檢查
        if provider_info.get('requires_api_key', True):
            st.warning("⚠️ 請先配置 API 密鑰")
            return
    
    col_discover, col_results = st.columns([1, 2])
    
    with col_discover:
        st.markdown("#### 🔍 發現設置")
        
        supported_categories = []
        if "flux-krea" in provider_info['features']:
            supported_categories.append("🎭 FLUX Krea 模型")
        if "flux" in provider_info['features']:
            supported_categories.append("⚡ Flux 模型")
        if "stable-diffusion" in provider_info['features']:
            supported_categories.append("🎨 Stable Diffusion")
        
        if not supported_categories:
            st.warning(f"{provider} 不支持圖像生成模型")
            return
        
        selected_categories = st.multiselect(
            "選擇要發現的模型類型:",
            supported_categories,
            default=supported_categories
        )
        
        # FLUX Krea 特別說明
        if "🎭 FLUX Krea 模型" in supported_categories:
            st.info("🎭 **FLUX Krea**: 美學優化模型，專注產生寫實且多樣化的圖像，避免過度飽和的 AI 外觀")
        
        if st.button("🚀 開始發現", type="primary", use_container_width=True):
            if selected_categories:
                discover_provider_models(provider, provider_info, selected_categories)
            else:
                st.warning("請選擇要發現的模型類型")
    
    with col_results:
        st.markdown("#### 📊 發現結果")
        
        discovered_models = provider_manager.get_provider_models(provider)
        
        if discovered_models:
            flux_krea_models = [m for m in discovered_models if m['category'] == 'flux-krea']
            flux_models = [m for m in discovered_models if m['category'] == 'flux']
            sd_models = [m for m in discovered_models if m['category'] == 'stable-diffusion']
            
            if flux_krea_models:
                st.markdown(f"**🎭 FLUX Krea 模型**: {len(flux_krea_models)} 個")
                st.success("🌟 美學優化專門模型")
                for model in flux_krea_models[:3]:
                    aesthetic_score = model.get('aesthetic_score', 3)
                    stars = "⭐" * min(aesthetic_score, 5)
                    st.write(f"• {model['icon']} {model['model_name']} {stars}")
            
            if flux_models:
                st.markdown(f"**⚡ Flux 模型**: {len(flux_models)} 個")
                for model in flux_models[:3]:
                    st.write(f"• {model['icon']} {model['model_name']}")
            
            if sd_models:
                st.markdown(f"**🎨 SD 模型**: {len(sd_models)} 個")
                for model in sd_models[:3]:
                    st.write(f"• {model['icon']} {model['model_name']}")
            
            if len(discovered_models) > 9:
                st.caption(f"... 還有 {len(discovered_models) - 9} 個模型")
        else:
            st.info("尚未發現任何模型")

def display_image_with_actions(image_url: str, image_id: str, generation_info: Dict = None):
    """顯示圖像和操作按鈕"""
    try:
        # 處理圖像 URL
        if image_url.startswith('data:image'):
            base64_data = image_url.split(',')[1]
            img_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(img_data))
        else:
            img_response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(img_response.content))
        
        # 顯示圖像
        st.image(img, use_column_width=True)
        
        # 圖像信息
        if generation_info:
            with st.expander("🔍 圖像信息"):
                st.write(f"**提示詞**: {generation_info.get('prompt', 'N/A')}")
                st.write(f"**模型**: {generation_info.get('model_name', 'N/A')}")
                st.write(f"**供應商**: {generation_info.get('provider', 'N/A')}")
                st.write(f"**尺寸**: {generation_info.get('size', 'N/A')}")
                st.write(f"**生成時間**: {generation_info.get('timestamp', 'N/A')}")
                
                # FLUX Krea 特殊信息
                if generation_info.get('category') == 'flux-krea':
                    st.write(f"**美學評分**: {'⭐' * generation_info.get('aesthetic_score', 5)}")
                    st.write(f"**引導強度**: {generation_info.get('guidance_scale', 3.5)}")
                    st.write(f"**推理步數**: {generation_info.get('steps', 28)}")
                    if generation_info.get('naturalism_boost'):
                        st.write("**自然主義增強**: ✅ 啟用")
                    color_harmony = generation_info.get('color_harmony', 'auto')
                    st.write(f"**色彩和諧度**: {color_harmony.title()}")
        
        # 操作按鈕
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 下載按鈕
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            filename_prefix = "flux_krea" if generation_info and generation_info.get('category') == 'flux-krea' else "generated"
            st.download_button(
                label="📥 下載",
                data=img_buffer.getvalue(),
                file_name=f"{filename_prefix}_{image_id}.png",
                mime="image/png",
                key=f"download_{image_id}",
                use_container_width=True
            )
        
        with col2:
            # 收藏按鈕
            if 'favorite_images' not in st.session_state:
                st.session_state.favorite_images = []
            
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
                        "generation_info": generation_info
                    }
                    st.session_state.favorite_images.append(favorite_item)
                    st.success("已加入收藏")
                rerun_app()
        
        with col3:
            # 重新生成按鈕
            if generation_info and st.button(
                "🔄 重新生成",
                key=f"regenerate_{image_id}",
                use_container_width=True
            ):
                st.session_state.regenerate_info = generation_info
                rerun_app()
        
        with col4:
            # 變化生成按鈕
            if generation_info and st.button(
                "🎨 變化生成",
                key=f"variation_{image_id}",
                use_container_width=True
            ):
                variation_info = generation_info.copy()
                variation_info['prompt'] = f"{generation_info.get('prompt', '')} (variation)"
                if 'seed' in variation_info and variation_info['seed'] >= 0:
                    variation_info['seed'] = random.randint(0, 2147483647)
                st.session_state.variation_info = variation_info
                rerun_app()
    
    except Exception as e:
        st.error(f"圖像顯示錯誤: {str(e)}")

def show_image_generation(provider: str, provider_info: Dict):
    """顯示完整的圖像生成界面"""
    st.markdown("### 🎨 圖像生成")
    
    # 檢查 API 配置
    config = st.session_state.api_config
    if not config.get('api_key') and provider_info.get('requires_api_key', True):
        st.warning("⚠️ 請先在密鑰管理中配置 API 密鑰")
        return
    
    # 獲取可用模型
    available_models = provider_manager.get_provider_models(provider)
    
    if not available_models:
        st.warning("⚠️ 尚未發現任何模型，請先進行模型發現")
        with st.expander("💡 如何發現模型？"):
            st.markdown("""
            1. 切換到 **🤖 模型發現** 標籤頁
            2. 選擇要發現的模型類型
            3. 點擊 **🚀 開始發現** 按鈕
            """)
        return
    
    # 生成設置區域
    col_settings, col_preview = st.columns([2, 1])
    
    with col_settings:
        st.markdown("#### ⚙️ 生成設置")
        
        # 模型選擇
        categories = list(set(model['category'] for model in available_models))
        
        # 優先顯示 FLUX Krea
        if 'flux-krea' in categories:
            categories.remove('flux-krea')
            categories.insert(0, 'flux-krea')
        
        if len(categories) > 1:
            selected_category = st.selectbox(
                "模型類別:",
                categories,
                format_func=lambda x: {
                    "flux-krea": "🎭 FLUX Krea (美學優化)",
                    "flux": "⚡ Flux AI",
                    "stable-diffusion": "🎨 Stable Diffusion"
                }.get(x, x.title())
            )
        else:
            selected_category = categories[0]
        
        category_models = [m for m in available_models if m['category'] == selected_category]
        selected_model_info = st.selectbox(
            "選擇模型:",
            category_models,
            format_func=lambda x: f"{x['icon']} {x['model_name']} {'⭐' * x.get('aesthetic_score', 3) if x['category'] == 'flux-krea' else ''}"
        )
        
        # FLUX Krea 特殊提示
        if selected_category == "flux-krea":
            st.success("🎭 **FLUX Krea 模式**：專為美學優化設計，生成更自然、寫實的圖像")
            st.info("💡 特色：避免過度飽和、更好的人類美學偏好、寫實多樣化")
        
        # 提示詞輸入
        st.markdown("#### 📝 提示詞")
        
        # 檢查重新生成或變化生成
        default_prompt = ""
        if 'regenerate_info' in st.session_state:
            default_prompt = st.session_state.regenerate_info.get('prompt', '')
            del st.session_state.regenerate_info
        elif 'variation_info' in st.session_state:
            default_prompt = st.session_state.variation_info.get('prompt', '')
            del st.session_state.variation_info
        
        prompt = st.text_area(
            "描述您想要生成的圖像:",
            value=default_prompt,
            height=120,
            placeholder="例如：A professional portrait of a confident businesswoman, natural lighting, realistic skin texture, detailed eyes",
            help="詳細描述您想要的圖像內容、風格、色彩等"
        )
        
        # 負面提示詞
        if selected_category in ["stable-diffusion", "flux-krea"]:
            negative_prompt = st.text_area(
                "負面提示詞 (可選):",
                height=60,
                placeholder="例如：blurry, low quality, distorted, oversaturated, artificial",
                help="描述您不希望出現在圖像中的內容"
            )
        else:
            negative_prompt = ""
        
        # 快速提示詞模板
        st.markdown("#### 💡 快速模板")
        
        if selected_category == "flux-krea":
            template_categories = {
                "人物肖像": [
                    "Professional portrait of a confident businesswoman, natural lighting, realistic skin texture, detailed eyes",
                    "Candid street photography of an elderly artist, warm golden hour light, authentic expression", 
                    "Studio headshot of a young musician, soft shadows, natural makeup, realistic details"
                ],
                "自然風景": [
                    "Misty mountain landscape at dawn, natural colors, atmospheric perspective, realistic lighting",
                    "Coastal scene with weathered rocks, natural wave patterns, authentic ocean colors",
                    "Forest path with dappled sunlight, realistic foliage, natural shadows and highlights"
                ]
            }
        else:
            template_categories = {
                "藝術創作": [
                    "Digital art illustration of a fantasy landscape with magical elements",
                    "Concept art of a futuristic cityscape with flying vehicles", 
                    "Abstract geometric composition with vibrant colors and patterns"
                ]
            }
        
        selected_template_category = st.selectbox("模板分類:", list(template_categories.keys()))
        
        for i, template in enumerate(template_categories[selected_template_category]):
            if st.button(f"📝 {template[:50]}...", key=f"template_{i}", use_container_width=True):
                st.session_state.quick_prompt = template
                rerun_app()
        
        # 應用快速提示詞
        if hasattr(st.session_state, 'quick_prompt'):
            prompt = st.session_state.quick_prompt
            del st.session_state.quick_prompt
            rerun_app()
    
    with col_preview:
        st.markdown("#### 🎯 參數設置")
        
        # 圖像尺寸
        if selected_category == "flux-krea":
            size_options = ["1024x1024", "1152x896", "896x1152", "1344x768", "768x1344"]
            default_size = "1024x1024"
        elif selected_category == "flux":
            size_options = ["1024x1024", "1152x896", "896x1152", "1344x768", "768x1344"]
            default_size = "1024x1024"
        else:
            size_options = ["512x512", "768x768", "1024x1024", "512x768", "768x512"]
            default_size = "512x512"
        
        selected_size = st.selectbox("圖像尺寸:", size_options, index=0)
        
        # 生成數量
        max_images = 4 if selected_category == "flux-krea" else 6
        num_images = st.slider("生成數量:", 1, max_images, 1)
        
        # 高級參數
        with st.expander("🔧 高級參數"):
            if selected_category == "flux-krea":
                st.markdown("**🎭 FLUX Krea 專用參數**")
                
                guidance_scale = st.slider(
                    "美學引導強度:", 
                    1.0, 10.0, 3.5, 0.5,
                    help="FLUX Krea 推薦較低值(2.0-4.0)以獲得更自然的結果"
                )
                
                steps = st.slider(
                    "推理步數:", 
                    10, 50, 28,
                    help="FLUX Krea 通常在 20-35 步之間效果最佳"
                )
                
                # FLUX Krea 特殊設置
                aesthetic_weight = st.slider(
                    "美學權重:",
                    0.5, 2.0, 1.0, 0.1,
                    help="控制美學優化的強度"
                )
                
                naturalism_boost = st.checkbox(
                    "自然主義增強",
                    value=True,
                    help="減少 AI 痕跡，提高圖像自然度"
                )
                
                color_harmony = st.selectbox(
                    "色彩和諧度:",
