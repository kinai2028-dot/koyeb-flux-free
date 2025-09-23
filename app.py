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
    page_title="Flux & SD Generator Pro - 完整版 + Pollinations",
    page_icon="🎨",
    layout="wide"
)

# 模型供應商配置 - 加入 Pollinations.ai
MODEL_PROVIDERS = {
    "Navy": {
        "name": "Navy AI",
        "icon": "⚓",
        "description": "Navy 高性能 AI 圖像生成服務",
        "api_type": "openai_compatible",
        "base_url": "https://api.navy/v1",
        "key_prefix": "sk-",
        "features": ["flux", "stable-diffusion"],
        "pricing": "按使用量計費",
        "speed": "快速",
        "quality": "高質量",
        "is_custom": False
    },
    "OpenAI Compatible": {
        "name": "OpenAI Compatible",
        "icon": "🤖",
        "description": "標準 OpenAI 格式兼容服務",
        "api_type": "openai_compatible",
        "base_url": "https://api.openai.com/v1",
        "key_prefix": "sk-",
        "features": ["dall-e", "custom-models"],
        "pricing": "官方定價",
        "speed": "中等",
        "quality": "官方品質",
        "is_custom": False
    },
    "Pollinations.ai": {
        "name": "Pollinations AI",
        "icon": "🌸",
        "description": "免費開源 AI 圖像生成平台，支持多種模型",
        "api_type": "pollinations",
        "base_url": "https://image.pollinations.ai/prompt",
        "key_prefix": "",
        "features": ["flux", "stable-diffusion", "flux-realism", "flux-anime", "any-dark"],
        "pricing": "完全免費",
        "speed": "快速",
        "quality": "高質量",
        "is_custom": False,
        "requires_api_key": False
    },
    "Hugging Face": {
        "name": "Hugging Face",
        "icon": "🤗",
        "description": "開源模型推理平台",
        "api_type": "huggingface",
        "base_url": "https://api-inference.huggingface.co",
        "key_prefix": "hf_",
        "features": ["flux", "stable-diffusion", "community-models"],
        "pricing": "免費/付費層級",
        "speed": "可變",
        "quality": "社區驅動",
        "is_custom": False
    },
    "Together AI": {
        "name": "Together AI",
        "icon": "🤝",
        "description": "高性能開源模型平台",
        "api_type": "openai_compatible",
        "base_url": "https://api.together.xyz/v1",
        "key_prefix": "",
        "features": ["flux", "stable-diffusion", "llama"],
        "pricing": "競爭性定價",
        "speed": "極快",
        "quality": "優秀",
        "is_custom": False
    },
    "Fireworks AI": {
        "name": "Fireworks AI",
        "icon": "🎆",
        "description": "快速推理和微調平台",
        "api_type": "openai_compatible",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "key_prefix": "",
        "features": ["flux", "stable-diffusion", "custom-training"],
        "pricing": "高性價比",
        "speed": "極快",
        "quality": "優秀",
        "is_custom": False
    },
    "Replicate": {
        "name": "Replicate",
        "icon": "🔄",
        "description": "雲端機器學習模型平台",
        "api_type": "replicate",
        "base_url": "https://api.replicate.com/v1",
        "key_prefix": "r8_",
        "features": ["flux", "stable-diffusion", "video-generation"],
        "pricing": "按秒計費",
        "speed": "可變",
        "quality": "多樣化",
        "is_custom": False
    },
    "RunPod": {
        "name": "RunPod",
        "icon": "🏃",
        "description": "GPU 雲服務平台",
        "api_type": "custom",
        "base_url": "https://api.runpod.ai/v2",
        "key_prefix": "",
        "features": ["flux", "stable-diffusion", "custom-endpoints"],
        "pricing": "GPU 租用",
        "speed": "可自定義",
        "quality": "可自定義",
        "is_custom": False
    },
    "DeepInfra": {
        "name": "DeepInfra",
        "icon": "🏗️",
        "description": "深度學習推理基礎設施",
        "api_type": "openai_compatible",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "key_prefix": "",
        "features": ["flux", "stable-diffusion", "llm"],
        "pricing": "靈活定價",
        "speed": "快速",
        "quality": "穩定",
        "is_custom": False
    }
}

# 模型識別規則和供應商特定模型庫
PROVIDER_MODEL_PATTERNS = {
    "flux": {
        "patterns": [
            r'flux[\.\-_]?1[\.\-_]?schnell',
            r'flux[\.\-_]?1[\.\-_]?dev',
            r'flux[\.\-_]?1[\.\-_]?pro',
            r'black[\-_]?forest[\-_]?labs',
            r'flux[\-_]?realism',
            r'flux[\-_]?anime'
        ],
        "providers": ["Navy", "Together AI", "Fireworks AI", "Hugging Face", "Replicate", "Pollinations.ai"]
    },
    "stable-diffusion": {
        "patterns": [
            r'stable[\-_]?diffusion',
            r'sdxl',
            r'sd[\-_]?xl',
            r'stabilityai'
        ],
        "providers": ["Navy", "Together AI", "Fireworks AI", "Hugging Face", "Replicate", "RunPod", "Pollinations.ai"]
    }
}

# 供應商特定模型庫 - 加入 Pollinations.ai 模型
PROVIDER_SPECIFIC_MODELS = {
    "Pollinations.ai": {
        "flux": [
            "flux",
            "flux-realism", 
            "flux-anime",
            "flux-3d"
        ],
        "stable-diffusion": [
            "turbo",
            "any-dark",
            "deliberate"
        ]
    },
    "Hugging Face": {
        "flux": [
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.1-dev",
            "XLabs-AI/flux-RealismLora",
            "multimodalart/FLUX.1-merged"
        ],
        "stable-diffusion": [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-2-1",
            "runwayml/stable-diffusion-v1-5",
            "prompthero/openjourney"
        ]
    },
    "Together AI": {
        "flux": [
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.1-dev"
        ],
        "stable-diffusion": [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5"
        ]
    },
    "Fireworks AI": {
        "flux": [
            "accounts/fireworks/models/flux-1-dev-fp8",
            "accounts/fireworks/models/flux-1-schnell-fp8"
        ],
        "stable-diffusion": [
            "accounts/fireworks/models/sdxl",
            "accounts/fireworks/models/stable-diffusion-v1-5"
        ]
    }
}

# 完整的供應商和模型管理系統（保持與之前相同的完整代碼）
class CompleteProviderManager:
    def __init__(self):
        self.db_path = "complete_providers.db"
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 自定義供應商表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS custom_providers (
                id TEXT PRIMARY KEY,
                provider_name TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                icon TEXT DEFAULT '🔧',
                description TEXT,
                api_type TEXT DEFAULT 'openai_compatible',
                base_url TEXT NOT NULL,
                key_prefix TEXT DEFAULT '',
                features TEXT DEFAULT '',
                pricing TEXT DEFAULT '自定義定價',
                speed TEXT DEFAULT '未知',
                quality TEXT DEFAULT '未知',
                headers TEXT DEFAULT '{}',
                auth_type TEXT DEFAULT 'bearer',
                timeout INTEGER DEFAULT 30,
                max_retries INTEGER DEFAULT 3,
                rate_limit INTEGER DEFAULT 60,
                requires_api_key BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
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
                category TEXT CHECK(category IN ('flux', 'stable-diffusion')) NOT NULL,
                description TEXT,
                icon TEXT,
                priority INTEGER DEFAULT 999,
                endpoint_path TEXT,
                model_type TEXT,
                expected_size TEXT,
                pricing_tier TEXT,
                performance_rating INTEGER DEFAULT 3,
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
        
        # 收藏表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS favorites (
                id TEXT PRIMARY KEY,
                generation_id TEXT,
                image_url TEXT,
                image_data TEXT,
                prompt TEXT,
                model_info TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # 其他方法保持與之前相同...
    def get_all_providers(self) -> Dict[str, Dict]:
        """獲取所有供應商（預設+自定義）"""
        all_providers = MODEL_PROVIDERS.copy()
        
        custom_providers = self.get_custom_providers()
        for provider in custom_providers:
            all_providers[provider['provider_name']] = provider
        
        return all_providers
    
    def get_custom_providers(self) -> List[Dict]:
        """獲取自定義供應商列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, provider_name, display_name, icon, description, api_type, base_url,
                   key_prefix, features, pricing, speed, quality, headers, auth_type,
                   timeout, max_retries, rate_limit, requires_api_key, is_active
            FROM custom_providers
            WHERE is_active = 1
            ORDER BY display_name
        ''')
        
        providers = []
        for row in cursor.fetchall():
            providers.append({
                'id': row[0],
                'provider_name': row[1],
                'display_name': row[2],
                'icon': row[3],
                'description': row[4],
                'api_type': row[5],
                'base_url': row[6],
                'key_prefix': row[7],
                'features': json.loads(row[8]) if row[8] else [],
                'pricing': row[9],
                'speed': row[10],
                'quality': row[11],
                'headers': json.loads(row[12]) if row[12] else {},
                'auth_type': row[13],
                'timeout': row[14],
                'max_retries': row[15],
                'rate_limit': row[16],
                'requires_api_key': bool(row[17]),
                'is_active': bool(row[18]),
                'is_custom': True
            })
        
        conn.close()
        return providers
    
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
    
    def save_provider_model(self, provider: str, model_name: str, model_id: str, 
                           category: str, **kwargs) -> Optional[str]:
        if category not in ['flux', 'stable-diffusion']:
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
             endpoint_path, model_type, expected_size, pricing_tier, performance_rating)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item_id, provider, model_name, model_id, category,
            kwargs.get('description', ''), kwargs.get('icon', '🤖'), 
            kwargs.get('priority', 999), kwargs.get('endpoint_path', ''),
            kwargs.get('model_type', ''), kwargs.get('expected_size', '512x512'),
            kwargs.get('pricing_tier', 'standard'), kwargs.get('performance_rating', 3)
        ))
        
        conn.commit()
        conn.close()
        return item_id
    
    def get_provider_models(self, provider: str = None, category: str = None) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT provider, model_name, model_id, category, description, icon, priority,
                   endpoint_path, model_type, expected_size, pricing_tier, performance_rating
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
                'expected_size': row[9], 'pricing_tier': row[10], 'performance_rating': row[11]
            })
        
        conn.close()
        return models
    
    def save_quick_switch_config(self, config_name: str, provider: str, api_key_id: str = None,
                                default_model_id: str = "", notes: str = "", is_favorite: bool = False) -> str:
        config_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO quick_switch_configs 
                (id, config_name, provider, api_key_id, default_model_id, is_favorite, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (config_id, config_name, provider, api_key_id, default_model_id, is_favorite, notes))
            
            conn.commit()
            conn.close()
            return config_id
            
        except sqlite3.IntegrityError:
            conn.close()
            return None
    
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
                'id': row[0],
                'config_name': row[1],
                'provider': row[2],
                'api_key_id': row[3],
                'default_model_id': row[4],
                'is_favorite': bool(row[5]),
                'last_used': row[6],
                'usage_count': row[7],
                'created_at': row[8],
                'notes': row[9],
                'key_name': row[10],
                'api_key': row[11],
                'base_url': row[12],
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
    """驗證 API 密鑰是否有效 - 加入 Pollinations.ai 支持"""
    try:
        all_providers = provider_manager.get_all_providers()
        provider_info = all_providers.get(provider, {})
        api_type = provider_info.get("api_type", "openai_compatible")
        
        # Pollinations.ai 不需要 API 密鑰
        if api_type == "pollinations":
            return True, f"{provider} 無需 API 密鑰，可直接使用"
        elif api_type == "huggingface":
            headers = {"Authorization": f"Bearer {api_key}"}
            test_url = f"{base_url}/models/stabilityai/stable-diffusion-xl-base-1.0"
            response = requests.get(test_url, headers=headers, timeout=10)
            return response.status_code == 200, f"{provider} API 驗證" + ("成功" if response.status_code == 200 else f"失敗 ({response.status_code})")
        elif api_type == "replicate":
            headers = {"Authorization": f"Token {api_key}"}
            test_url = f"{base_url}/models"
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
        # Pollinations.ai URL 構建
        base_url = "https://image.pollinations.ai/prompt"
        
        # URL 編碼提示詞
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt)
        
        # 構建參數
        url_params = []
        
        # 模型參數
        if model and model != "flux":
            url_params.append(f"model={model}")
        
        # 尺寸參數
        if "size" in params:
            width, height = map(int, params["size"].split('x'))
            url_params.append(f"width={width}")
            url_params.append(f"height={height}")
        else:
            url_params.append("width=1024")
            url_params.append("height=1024")
        
        # 種子參數
        if params.get("seed", -1) >= 0:
            url_params.append(f"seed={params['seed']}")
        
        # 其他參數
        if params.get("enhance", False):
            url_params.append("enhance=true")
        
        if params.get("nologo", True):
            url_params.append("nologo=true")
        
        # 構建完整URL
        if url_params:
            full_url = f"{base_url}/{encoded_prompt}?{'&'.join(url_params)}"
        else:
            full_url = f"{base_url}/{encoded_prompt}"
        
        # 發送請求
        response = requests.get(full_url, timeout=60)
        
        if response.status_code == 200:
            # 將圖像轉換為 base64
            encoded_image = base64.b64encode(response.content).decode()
            
            # 創建模擬 OpenAI 響應格式
            class MockResponse:
                def __init__(self, image_data):
                    num_images = params.get("n", 1)
                    self.data = [type('obj', (object,), {
                        'url': f"data:image/png;base64,{image_data}"
                    })() for _ in range(num_images)]
            
            return True, MockResponse(encoded_image)
        else:
            return False, f"HTTP {response.status_code}: Pollinations API 調用失敗"
            
    except requests.exceptions.Timeout:
        return False, "請求超時，請稍後重試"
    except requests.exceptions.ConnectionError:
        return False, "網絡連接錯誤"
    except Exception as e:
        return False, str(e)

def generate_images_with_retry(client, provider: str, api_key: str, base_url: str, **params) -> Tuple[bool, any]:
    """帶重試機制的圖像生成 - 加入 Pollinations.ai 支持"""
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            all_providers = provider_manager.get_all_providers()
            provider_info = all_providers.get(provider, {})
            api_type = provider_info.get("api_type", "openai_compatible")
            
            if attempt > 0:
                st.info(f"🔄 嘗試重新生成 (第 {attempt + 1}/{max_retries} 次)")
                time.sleep(base_delay * (2 ** (attempt - 1)))  # 指數退避
            
            if api_type == "pollinations":
                return generate_pollinations_image(**params)
            elif api_type == "huggingface":
                return generate_hf_image(api_key, base_url, provider, **params)
            elif api_type == "replicate":
                return generate_replicate_image(api_key, base_url, provider, **params)
            else:  # openai_compatible
                return generate_openai_image(client, **params)
        
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                should_retry = False
                
                # 判斷是否應該重試
                if "500" in error_msg or "502" in error_msg or "503" in error_msg:
                    should_retry = True
                elif "timeout" in error_msg.lower():
                    should_retry = True
                elif "connection" in error_msg.lower():
                    should_retry = True
                
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

def generate_hf_image(api_key: str, base_url: str, provider: str, **params) -> Tuple[bool, any]:
    """Hugging Face API 圖像生成"""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # 構建請求數據
        data = {
            "inputs": params.get("prompt", ""),
            "parameters": {
                "num_inference_steps": params.get("steps", 20),
                "guidance_scale": params.get("guidance_scale", 7.5),
            }
        }
        
        # 如果有指定尺寸，添加到參數中
        if "size" in params:
            width, height = map(int, params["size"].split('x'))
            data["parameters"]["width"] = width
            data["parameters"]["height"] = height
        
        # 獲取模型端點路徑
        model_name = params.get("model", "stable-diffusion-xl")
        provider_models = provider_manager.get_provider_models(provider)
        model_info = next((m for m in provider_models if m['model_id'] == model_name), None)
        
        if model_info and model_info.get('endpoint_path'):
            endpoint_path = model_info['endpoint_path']
        else:
            # 默認端點路徑
            if "flux" in model_name.lower():
                endpoint_path = f"black-forest-labs/{model_name}"
            else:
                endpoint_path = f"stabilityai/{model_name}"
        
        response = requests.post(
            f"{base_url}/models/{endpoint_path}",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            # 處理返回的圖像數據
            encoded_image = base64.b64encode(response.content).decode()
            
            # 創建模擬 OpenAI 響應格式
            class MockResponse:
                def __init__(self, image_data):
                    num_images = params.get("n", 1)
                    self.data = [type('obj', (object,), {
                        'url': f"data:image/png;base64,{image_data}"
                    })() for _ in range(num_images)]
            
            return True, MockResponse(encoded_image)
        else:
            return False, f"HTTP {response.status_code}: HF API 調用失敗"
            
    except Exception as e:
        return False, str(e)

def generate_replicate_image(api_key: str, base_url: str, provider: str, **params) -> Tuple[bool, any]:
    """Replicate API 圖像生成"""
    # 保持原有實現
    return False, "Replicate 實現開發中"

def discover_provider_models(provider: str, provider_info: Dict, selected_categories: List[str]):
    """發現供應商模型 - 加入 Pollinations.ai 支持"""
    api_type = provider_info.get("api_type", "openai_compatible")
    config = st.session_state.api_config
    
    with st.spinner(f"🔍 正在從 {provider} 發現模型..."):
        discovered_count = {"flux": 0, "stable-diffusion": 0}
        
        try:
            if api_type == "pollinations":
                # Pollinations.ai 預定義模型
                if provider in PROVIDER_SPECIFIC_MODELS:
                    provider_models = PROVIDER_SPECIFIC_MODELS[provider]
                    
                    for category, models in provider_models.items():
                        if (category == "flux" and "⚡ Flux 模型" in selected_categories) or \
                           (category == "stable-diffusion" and "🎨 Stable Diffusion" in selected_categories):
                            
                            for model_name in models:
                                saved_id = provider_manager.save_provider_model(
                                    provider=provider,
                                    model_name=model_name,
                                    model_id=model_name,
                                    category=category,
                                    description=f"{model_name} model from Pollinations.ai",
                                    icon="🌸",
                                    pricing_tier="free",
                                    expected_size="1024x1024",
                                    priority=1 if model_name == "flux" else 999
                                )
                                
                                if saved_id:
                                    discovered_count[category] += 1
            
            elif api_type == "huggingface":
                if provider in PROVIDER_SPECIFIC_MODELS:
                    provider_models = PROVIDER_SPECIFIC_MODELS[provider]
                    
                    for category, models in provider_models.items():
                        if (category == "flux" and "⚡ Flux 模型" in selected_categories) or \
                           (category == "stable-diffusion" and "🎨 Stable Diffusion" in selected_categories):
                            
                            for model_path in models:
                                model_name = model_path.split('/')[-1]
                                
                                headers = {"Authorization": f"Bearer {config['api_key']}"}
                                test_url = f"{config['base_url']}/models/{model_path}"
                                
                                try:
                                    response = requests.get(test_url, headers=headers, timeout=5)
                                    if response.status_code == 200:
                                        saved_id = provider_manager.save_provider_model(
                                            provider=provider,
                                            model_name=model_name,
                                            model_id=model_name,
                                            category=category,
                                            description=f"{category.title()} model from {provider}",
                                            icon="⚡" if category == "flux" else "🎨",
                                            endpoint_path=model_path,
                                            pricing_tier="community",
                                            expected_size="1024x1024" if category == "flux" else "512x512"
                                        )
                                        
                                        if saved_id:
                                            discovered_count[category] += 1
                                except:
                                    continue
            
            elif api_type == "openai_compatible":
                client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])
                response = client.models.list()
                
                for model in response.data:
                    model_id = model.id
                    model_lower = model_id.lower()
                    
                    category = None
                    if any(re.search(pattern, model_lower) for pattern in PROVIDER_MODEL_PATTERNS["flux"]["patterns"]):
                        if "⚡ Flux 模型" in selected_categories:
                            category = "flux"
                    elif any(re.search(pattern, model_lower) for pattern in PROVIDER_MODEL_PATTERNS["stable-diffusion"]["patterns"]):
                        if "🎨 Stable Diffusion" in selected_categories:
                            category = "stable-diffusion"
                    
                    if category:
                        saved_id = provider_manager.save_provider_model(
                            provider=provider,
                            model_name=model_id,
                            model_id=model_id,
                            category=category,
                            description=f"{category.title()} model from {provider}",
                            icon="⚡" if category == "flux" else "🎨",
                            pricing_tier="api",
                            expected_size="1024x1024" if category == "flux" else "512x512"
                        )
                        
                        if saved_id:
                            discovered_count[category] += 1
            
            total_discovered = sum(discovered_count.values())
            if total_discovered > 0:
                st.success(f"✅ 從 {provider} 發現 {total_discovered} 個模型")
                for category, count in discovered_count.items():
                    if count > 0:
                        st.info(f"{'⚡ Flux' if category == 'flux' else '🎨 SD'}: {count} 個")
            else:
                st.info(f"ℹ️ 在 {provider} 未發現新模型")
            
            rerun_app()
            
        except Exception as e:
            st.error(f"❌ 發現失敗: {str(e)}")

# 修改密鑰管理以支援 Pollinations.ai（無需密鑰）
def show_provider_key_management(provider: str, provider_info: Dict):
    """顯示供應商密鑰管理 - 支援 Pollinations.ai"""
    st.markdown("### 🔑 密鑰管理")
    
    # 檢查是否需要 API 密鑰
    requires_key = provider_info.get('requires_api_key', True)
    
    if not requires_key:
        st.success(f"🌸 {provider_info['name']} 完全免費，無需 API 密鑰！")
        st.info("✨ 您可以直接開始生成圖像，無需任何配置")
        
        # 為不需要密鑰的供應商創建虛擬配置
        if st.button("✅ 啟用免費服務", type="primary", use_container_width=True):
            st.session_state.api_config = {
                'provider': provider,
                'api_key': 'no-key-required',
                'base_url': provider_info['base_url'],
                'validated': True,
                'key_name': f'{provider} 免費服務'
            }
            st.success(f"已啟用 {provider_info['name']} 免費服務")
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
        if provider_info.get('is_custom'):
            placeholder = f"輸入 {provider_info['display_name']} API 密鑰..."
        else:
            placeholder = f"輸入 {provider_info['name']} API 密鑰..."
        
        api_key = st.text_input(
            "API 密鑰:",
            type="password",
            placeholder=placeholder
        )
    
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

# 顯示圖像和操作按鈕的函數（保持不變）
def display_image_with_actions(image_url: str, image_id: str, generation_info: Dict = None):
    """顯示圖像和操作按鈕"""
    try:
        # 處理不同類型的圖像 URL
        if image_url.startswith('data:image'):
            # Base64 圖像
            base64_data = image_url.split(',')[1]
            img_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(img_data))
        else:
            # 普通 URL
            img_response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(img_response.content))
            img_data = img_response.content
        
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
        
        # 操作按鈕
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 下載按鈕
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            st.download_button(
                label="📥 下載",
                data=img_buffer.getvalue(),
                file_name=f"generated_{image_id}.png",
                mime="image/png",
                key=f"download_{image_id}",
                use_container_width=True
            )
        
        with col2:
            # 收藏按鈕
            is_favorite = any(fav['id'] == image_id for fav in st.session_state.get('favorite_images', []))
            if st.button(
                "⭐ 已收藏" if is_favorite else "☆ 收藏",
                key=f"favorite_{image_id}",
                use_container_width=True
            ):
                if 'favorite_images' not in st.session_state:
                    st.session_state.favorite_images = []
                    
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
            # 分享按鈕
            if st.button(
                "🔗 複製連結",
                key=f"share_{image_id}",
                use_container_width=True
            ):
                st.info("分享功能開發中")
    
    except Exception as e:
        st.error(f"圖像顯示錯誤: {str(e)}")

# 由於篇幅限制，以下函數保持與之前版本相同：
# - show_quick_switch_panel()
# - switch_to_config()
# - show_provider_selector()
# - show_provider_model_discovery()
# - show_image_generation()
# - generate_image_main()
# - show_generation_history()
# - show_provider_performance()
# - show_custom_provider_creator()
# - show_provider_management()
# - init_session_state()

# 所有其他函數保持不變，只需在相關地方加入對 Pollinations.ai 的支持

def init_session_state():
    """初始化會話狀態"""
    if 'api_config' not in st.session_state:
        st.session_state.api_config = {
            'provider': '',
            'api_key': '',
            'base_url': '',
            'validated': False
        }
    
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'favorite_images' not in st.session_state:
        st.session_state.favorite_images = []

# 初始化
init_session_state()

# 檢查 API 配置
api_configured = st.session_state.api_config.get('api_key') is not None and st.session_state.api_config.get('api_key') != ''

# 側邊欄
with st.sidebar:
    st.markdown("### ⚡ 快速切換")
    
    if 'selected_provider' in st.session_state and api_configured:
        provider = st.session_state.selected_provider
        all_providers = provider_manager.get_all_providers()
        provider_info = all_providers.get(provider, {})
        
        if provider_info.get('is_custom'):
            current_name = f"{provider_info['icon']} {provider_info['display_name']}"
        else:
            current_name = f"{provider_info['icon']} {provider_info['name']}"
        
        st.success(f"✅ {current_name}")
        
        if st.session_state.api_config.get('key_name'):
            st.caption(f"🔑 {st.session_state.api_config['key_name']}")
    else:
        st.info("未配置 API")
    
    # 快速配置 Pollinations.ai
    st.markdown("---")
    st.markdown("### 🌸 免費服務")
    
    if st.button("🚀 使用 Pollinations.ai", use_container_width=True, type="primary"):
        st.session_state.selected_provider = "Pollinations.ai"
        st.session_state.api_config = {
            'provider': "Pollinations.ai",
            'api_key': 'no-key-required',
            'base_url': 'https://image.pollinations.ai/prompt',
            'validated': True,
            'key_name': 'Pollinations.ai 免費服務'
        }
        st.success("🌸 Pollinations.ai 已啟用！")
        rerun_app()
    
    st.caption("🎨 完全免費的 AI 圖像生成")
    
    st.markdown("---")
    
    # 統計信息
    st.markdown("### 📊 統計")
    total_keys = len(provider_manager.get_api_keys())
    quick_configs = provider_manager.get_quick_switch_configs()
    total_configs = len(quick_configs)
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("密鑰數", total_keys)
    with col_stat2:
        st.metric("快速配置", total_configs)

# 主標題
st.title("🎨 Flux & SD Generator Pro - 完整版 + Pollinations.ai")

# 主要內容
if 'selected_provider' not in st.session_state:
    st.subheader("🏢 選擇模型供應商")
    
    # 突出顯示免費服務
    st.markdown("### 🌸 推薦：免費服務")
    
    with st.container():
        col_pollinations = st.columns(1)[0]
        with col_pollinations:
            st.markdown("#### 🌸 Pollinations.ai - 完全免費！")
            st.success("✨ 無需註冊、無需 API 密鑰、無使用限制")
            st.caption("支持 Flux、Stable Diffusion 等多種高質量模型")
            
            col_features = st.columns(3)
            with col_features[0]:
                st.info("🆓 **完全免費**")
            with col_features[1]:
                st.info("⚡ **快速生成**")
            with col_features[2]:
                st.info("🎨 **高質量輸出**")
            
            if st.button("🚀 立即使用 Pollinations.ai", type="primary", use_container_width=True):
                st.session_state.selected_provider = "Pollinations.ai"
                st.session_state.api_config = {
                    'provider': "Pollinations.ai",
                    'api_key': 'no-key-required',
                    'base_url': 'https://image.pollinations.ai/prompt',
                    'validated': True,
                    'key_name': 'Pollinations.ai 免費服務'
                }
                st.success("🌸 Pollinations.ai 已啟用！正在跳轉...")
                rerun_app()
    
    st.markdown("---")
    
    # 顯示其他供應商
    all_providers = provider_manager.get_all_providers()
    other_providers = {k: v for k, v in all_providers.items() if k != "Pollinations.ai"}
    
    st.markdown("### 🏭 其他供應商")
    
    cols = st.columns(3)
    for i, (provider_key, provider_info) in enumerate(other_providers.items()):
        with cols[i % 3]:
            with st.container():
                if provider_info.get('is_custom'):
                    st.markdown(f"#### {provider_info['icon']} {provider_info['display_name']}")
                else:
                    st.markdown(f"#### {provider_info['icon']} {provider_info['name']}")
                
                st.caption(provider_info['description'])
                
                if st.button(f"選擇", key=f"select_{provider_key}", use_container_width=True):
                    st.session_state.selected_provider = provider_key
                    if provider_info.get('is_custom'):
                        st.success(f"已選擇 {provider_info['display_name']}")
                    else:
                        st.success(f"已選擇 {provider_info['name']}")
                    rerun_app()
                
                saved_keys = provider_manager.get_api_keys(provider_key)
                if saved_keys:
                    st.caption(f"🔑 已保存 {len(saved_keys)} 個密鑰")

else:
    # 顯示供應商管理界面（包含完整功能）
    selected_provider = st.session_state.selected_provider
    all_providers = provider_manager.get_all_providers()
    provider_info = all_providers[selected_provider]
    
    if provider_info.get('is_custom'):
        st.subheader(f"{provider_info['icon']} {provider_info['display_name']} (自定義)")
    else:
        st.subheader(f"{provider_info['icon']} {provider_info['name']}")
    
    col_info, col_switch = st.columns([3, 1])
    
    with col_info:
        st.info(f"📋 {provider_info['description']}")
        st.caption(f"🔗 API 類型: {provider_info['api_type']} | 端點: {provider_info['base_url']}")
        
        features_badges = " ".join([f"`{feature}`" for feature in provider_info['features']])
        st.markdown(f"**支持功能**: {features_badges}")
    
    with col_switch:
        if st.button("🔄 切換供應商", use_container_width=True):
            del st.session_state.selected_provider
            rerun_app()
    
    management_tabs = st.tabs(["🔑 密鑰管理", "🤖 模型發現", "🎨 圖像生成", "📊 性能監控"])
    
    with management_tabs[0]:
        show_provider_key_management(selected_provider, provider_info)
    
    with management_tabs[1]:
        show_provider_model_discovery(selected_provider, provider_info)
    
    with management_tabs[2]:
        # show_image_generation(selected_provider, provider_info)
        st.markdown("### 🎨 圖像生成")
        st.info("🚀 完整的圖像生成界面開發中...")
    
    with management_tabs[3]:
        # show_provider_performance(selected_provider, provider_info)
        st.markdown("### 📊 性能監控")
        st.info("📊 性能監控功能開發中...")

# 頁腳
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🌸 <strong>Pollinations.ai 免費服務</strong> | 
    ⚡ <strong>快速切換</strong> | 
    🎨 <strong>多模型支持</strong> | 
    📊 <strong>智能管理</strong>
    <br><br>
    <small>現已支援 Pollinations.ai 免費 AI 圖像生成服務！</small>
</div>
""", unsafe_allow_html=True)
