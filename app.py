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
        # 使用替代的顯示方式
        if badge_type == "secondary":
            st.caption(f"🏷️ {text}")
        elif badge_type == "success":
            st.success(f"✅ {text}")
        else:
            st.info(f"📊 {text}")

# 設定頁面配置
st.set_page_config(
    page_title="Flux & SD Generator Pro - 自設供應商版",
    page_icon="🎨",
    layout="wide"
)

# 模型供應商配置（與之前相同）
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

# 模型識別規則 - 按供應商分類
PROVIDER_MODEL_PATTERNS = {
    "flux": {
        "patterns": [
            r'flux[\.\-_]?1[\.\-_]?schnell',
            r'flux[\.\-_]?1[\.\-_]?dev',
            r'flux[\.\-_]?1[\.\-_]?pro',
            r'black[\-_]?forest[\-_]?labs'
        ],
        "providers": ["Navy", "Together AI", "Fireworks AI", "Hugging Face", "Replicate"]
    },
    "stable-diffusion": {
        "patterns": [
            r'stable[\-_]?diffusion',
            r'sdxl',
            r'sd[\-_]?xl',
            r'stabilityai'
        ],
        "providers": ["Navy", "Together AI", "Fireworks AI", "Hugging Face", "Replicate", "RunPod"]
    }
}

# 供應商特定模型庫
PROVIDER_SPECIFIC_MODELS = {
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

# 擴展的供應商和模型管理系統
class CustomProviderModelManager:
    def __init__(self):
        self.db_path = "custom_provider_models.db"
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
        
        # 生成歷史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generation_history (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
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
    
    def save_custom_provider(self, **kwargs) -> str:
        """保存自定義供應商"""
        provider_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO custom_providers 
                (id, provider_name, display_name, icon, description, api_type, base_url, 
                 key_prefix, features, pricing, speed, quality, headers, auth_type, 
                 timeout, max_retries, rate_limit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                provider_id,
                kwargs.get('provider_name', ''),
                kwargs.get('display_name', ''),
                kwargs.get('icon', '🔧'),
                kwargs.get('description', ''),
                kwargs.get('api_type', 'openai_compatible'),
                kwargs.get('base_url', ''),
                kwargs.get('key_prefix', ''),
                json.dumps(kwargs.get('features', [])),
                kwargs.get('pricing', '自定義定價'),
                kwargs.get('speed', '未知'),
                kwargs.get('quality', '未知'),
                json.dumps(kwargs.get('headers', {})),
                kwargs.get('auth_type', 'bearer'),
                kwargs.get('timeout', 30),
                kwargs.get('max_retries', 3),
                kwargs.get('rate_limit', 60)
            ))
            
            conn.commit()
            conn.close()
            return provider_id
            
        except sqlite3.IntegrityError:
            conn.close()
            return None
    
    def get_custom_providers(self) -> List[Dict]:
        """獲取自定義供應商列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, provider_name, display_name, icon, description, api_type, base_url,
                   key_prefix, features, pricing, speed, quality, headers, auth_type,
                   timeout, max_retries, rate_limit, is_active
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
                'is_active': bool(row[17]),
                'is_custom': True
            })
        
        conn.close()
        return providers
    
    def get_all_providers(self) -> Dict[str, Dict]:
        """獲取所有供應商（預設+自定義）"""
        all_providers = MODEL_PROVIDERS.copy()
        
        custom_providers = self.get_custom_providers()
        for provider in custom_providers:
            all_providers[provider['provider_name']] = provider
        
        return all_providers
    
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
        """保存供應商模型"""
        if category not in ['flux', 'stable-diffusion']:
            return None
        
        item_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 檢查是否已存在
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
        """獲取供應商模型"""
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
    
    def save_generation_history(self, provider: str, model_id: str, prompt: str, 
                               image_url: str = "", image_data: str = "", metadata: Dict = {}) -> str:
        """保存生成歷史"""
        history_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO generation_history
            (id, provider, model_id, prompt, image_url, image_data, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (history_id, provider, model_id, prompt, image_url, image_data, json.dumps(metadata)))
        
        conn.commit()
        conn.close()
        return history_id
    
    def get_generation_history(self, limit: int = 50) -> List[Dict]:
        """獲取生成歷史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, provider, model_id, prompt, image_url, image_data, metadata, created_at
            FROM generation_history
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'id': row[0],
                'provider': row[1],
                'model_id': row[2],
                'prompt': row[3],
                'image_url': row[4],
                'image_data': row[5],
                'metadata': json.loads(row[6]) if row[6] else {},
                'created_at': row[7]
            })
        
        conn.close()
        return history
    
    def update_key_validation(self, key_id: str, validated: bool):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE api_keys SET validated = ? WHERE id = ?", (validated, key_id))
        conn.commit()
        conn.close()
    
    def delete_api_key(self, key_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
        conn.commit()
        conn.close()

# 全局實例
provider_manager = CustomProviderModelManager()

def validate_api_key(api_key: str, base_url: str, provider: str) -> Tuple[bool, str]:
    """驗證 API 密鑰是否有效"""
    try:
        all_providers = provider_manager.get_all_providers()
        provider_info = all_providers.get(provider, {})
        api_type = provider_info.get("api_type", "openai_compatible")
        
        if api_type == "huggingface":
            headers = {"Authorization": f"Bearer {api_key}"}
            test_url = f"{base_url}/models/stabilityai/stable-diffusion-xl-base-1.0"
            response = requests.get(test_url, headers=headers, timeout=10)
            return response.status_code == 200, f"{provider} API 驗證" + ("成功" if response.status_code == 200 else f"失敗 ({response.status_code})")
        elif api_type == "replicate":
            headers = {"Authorization": f"Token {api_key}"}
            test_url = f"{base_url}/models"
            response = requests.get(test_url, headers=headers, timeout=10)
            return response.status_code == 200, f"{provider} API 驗證" + ("成功" if response.status_code == 200 else f"失敗 ({response.status_code})")
        else:  # openai_compatible or custom
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

def generate_images_with_retry(client, provider: str, api_key: str, base_url: str, **params) -> Tuple[bool, any]:
    """帶重試機制的圖像生成"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            all_providers = provider_manager.get_all_providers()
            provider_info = all_providers.get(provider, {})
            api_type = provider_info.get("api_type", "openai_compatible")
            
            if api_type == "huggingface":
                # HF API 調用
                headers = {"Authorization": f"Bearer {api_key}"}
                data = {"inputs": params.get("prompt", "")}
                
                model_name = params.get("model", "stable-diffusion-xl")
                provider_models = provider_manager.get_provider_models(provider)
                model_info = next((m for m in provider_models if m['model_id'] == model_name), {})
                endpoint_path = model_info.get('endpoint_path', f"stabilityai/{model_name}")
                
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
                            encoded_image = base64.b64encode(image_data).decode()
                            self.data = [type('obj', (object,), {
                                'url': f"data:image/png;base64,{encoded_image}"
                            })()]
                    
                    return True, MockResponse(response.content)
                else:
                    raise Exception(f"HTTP {response.status_code}: HF API 調用失敗")
            else:
                # OpenAI Compatible API 調用
                response = client.images.generate(**params)
                return True, response
        
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"第 {attempt + 1} 次嘗試失敗，正在重試...")
                time.sleep(2)
                continue
            else:
                return False, str(e)
    
    return False, "所有重試均失敗"

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
        
        # 操作按鈕
        col1, col2, col3 = st.columns(3)
        
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
    
    except Exception as e:
        st.error(f"圖像顯示錯誤: {str(e)}")

def show_custom_provider_creator():
    """顯示自定義供應商創建器"""
    st.subheader("🛠️ 創建自定義 API 供應商")
    
    with st.form("custom_provider_form"):
        st.markdown("### 📋 基本信息")
        
        col_name, col_display = st.columns(2)
        
        with col_name:
            provider_name = st.text_input(
                "供應商 ID *",
                placeholder="例如：my-custom-api",
                help="唯一標識符，只能包含字母、數字、連字符"
            )
        
        with col_display:
            display_name = st.text_input(
                "顯示名稱 *",
                placeholder="例如：My Custom API",
                help="在界面中顯示的名稱"
            )
        
        col_icon, col_desc = st.columns([1, 3])
        
        with col_icon:
            icon = st.text_input("圖標", value="🔧", help="單個 Emoji 表情")
        
        with col_desc:
            description = st.text_area(
                "描述",
                placeholder="描述此 API 供應商的特色和用途...",
                height=100
            )
        
        st.markdown("### 🔧 API 配置")
        
        col_type, col_url = st.columns(2)
        
        with col_type:
            api_type = st.selectbox(
                "API 類型 *",
                ["openai_compatible", "huggingface", "replicate", "custom"],
                format_func=lambda x: {
                    "openai_compatible": "OpenAI 兼容格式",
                    "huggingface": "Hugging Face 格式",
                    "replicate": "Replicate 格式",
                    "custom": "自定義格式"
                }[x]
            )
        
        with col_url:
            base_url = st.text_input(
                "API 端點 URL *",
                placeholder="https://api.example.com/v1",
                help="完整的 API 基礎 URL"
            )
        
        col_prefix, col_auth = st.columns(2)
        
        with col_prefix:
            key_prefix = st.text_input(
                "密鑰前綴",
                placeholder="sk-",
                help="API 密鑰的前綴格式"
            )
        
        with col_auth:
            auth_type = st.selectbox(
                "認證方式",
                ["bearer", "api_key", "custom"],
                format_func=lambda x: {
                    "bearer": "Bearer Token",
                    "api_key": "API Key Header",
                    "custom": "自定義認證"
                }[x]
            )
        
        st.markdown("### 🎯 功能支持")
        
        features = st.multiselect(
            "支持的功能",
            ["flux", "stable-diffusion", "dall-e", "midjourney", "video-generation", "audio-generation", "custom-models"],
            format_func=lambda x: {
                "flux": "⚡ Flux AI 模型",
                "stable-diffusion": "🎨 Stable Diffusion",
                "dall-e": "🖼️ DALL-E",
                "midjourney": "🎭 Midjourney 風格",
                "video-generation": "🎬 視頻生成",
                "audio-generation": "🎵 音頻生成",
                "custom-models": "🔧 自定義模型"
            }[x]
        )
        
        st.markdown("### 📊 性能指標")
        
        col_pricing, col_speed, col_quality = st.columns(3)
        
        with col_pricing:
            pricing = st.text_input("定價模式", placeholder="例如：$0.01/請求")
        
        with col_speed:
            speed = st.selectbox("速度等級", ["極慢", "慢", "中等", "快速", "極快", "未知"])
        
        with col_quality:
            quality = st.selectbox("品質等級", ["低", "中", "高", "優秀", "頂級", "未知"])
        
        # 提交按鈕
        submit_button = st.form_submit_button("💾 創建供應商", type="primary", use_container_width=True)
        
        if submit_button:
            # 驗證必填字段
            if not provider_name or not display_name or not base_url:
                st.error("❌ 請填寫所有必填字段 (*)")
            elif not re.match(r'^[a-zA-Z0-9-_]+$', provider_name):
                st.error("❌ 供應商 ID 只能包含字母、數字、連字符和下劃線")
            else:
                # 保存自定義供應商
                provider_data = {
                    'provider_name': provider_name,
                    'display_name': display_name,
                    'icon': icon,
                    'description': description,
                    'api_type': api_type,
                    'base_url': base_url,
                    'key_prefix': key_prefix,
                    'features': features,
                    'pricing': pricing,
                    'speed': speed,
                    'quality': quality,
                    'auth_type': auth_type,
                    'timeout': 30,
                    'max_retries': 3,
                    'rate_limit': 60
                }
                
                provider_id = provider_manager.save_custom_provider(**provider_data)
                
                if provider_id:
                    st.success(f"✅ 自定義供應商 '{display_name}' 創建成功！")
                    st.info(f"🆔 供應商 ID: {provider_id[:8]}...")
                    time.sleep(1)
                    rerun_app()
                else:
                    st.error(f"❌ 創建失敗：供應商 ID '{provider_name}' 已存在")

def show_provider_selector():
    """顯示供應商選擇器（包含自定義供應商）"""
    st.subheader("🏢 選擇模型供應商")
    
    # 獲取所有供應商
    all_providers = provider_manager.get_all_providers()
    
    # 分類顯示
    default_providers = {k: v for k, v in all_providers.items() if not v.get('is_custom', False)}
    custom_providers = {k: v for k, v in all_providers.items() if v.get('is_custom', False)}
    
    # 預設供應商
    if default_providers:
        st.markdown("### 🏭 預設供應商")
        
        # 創建比較表格
        provider_data = []
        for provider_key, provider_info in default_providers.items():
            provider_data.append({
                "供應商": f"{provider_info['icon']} {provider_info['name']}",
                "特色": ", ".join(provider_info['features']),
                "定價": provider_info['pricing'],
                "速度": provider_info['speed'],
                "品質": provider_info['quality']
            })
        
        st.dataframe(provider_data, use_container_width=True)
        
        cols = st.columns(3)
        for i, (provider_key, provider_info) in enumerate(default_providers.items()):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"#### {provider_info['icon']} {provider_info['name']}")
                    st.caption(provider_info['description'])
                    
                    # 特色標籤
                    if 'features' in provider_info:
                        features_text = " | ".join([f"🏷️ {feature}" for feature in provider_info['features']])
                        st.markdown(f"**特色**: {features_text}")
                    
                    # 選擇按鈕
                    if st.button(f"選擇 {provider_info['name']}", key=f"select_default_{provider_key}", use_container_width=True):
                        st.session_state.selected_provider = provider_key
                        st.success(f"已選擇 {provider_info['name']}")
                        rerun_app()
                    
                    # 顯示已保存的密鑰數量 - 修復後的版本
                    saved_keys = provider_manager.get_api_keys(provider_key)
                    if saved_keys:
                        # 使用兼容的方式顯示標籤
                        st.caption(f"🔑 已保存 {len(saved_keys)} 個密鑰")
    
    # 自定義供應商
    if custom_providers:
        st.markdown("### 🔧 自定義供應商")
        
        cols = st.columns(3)
        for i, (provider_key, provider_info) in enumerate(custom_providers.items()):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"#### {provider_info['icon']} {provider_info['display_name']}")
                    st.caption(provider_info['description'] or "自定義 API 供應商")
                    
                    # API 類型和端點
                    st.caption(f"**類型**: {provider_info['api_type']} | **端點**: {provider_info['base_url'][:30]}...")
                    
                    # 特色標籤
                    if provider_info['features']:
                        features_text = " | ".join([f"🏷️ {feature}" for feature in provider_info['features']])
                        st.markdown(f"**功能**: {features_text}")
                    
                    # 選擇按鈕
                    if st.button(f"選擇 {provider_info['display_name']}", key=f"select_custom_{provider_key}", use_container_width=True):
                        st.session_state.selected_provider = provider_key
                        st.success(f"已選擇 {provider_info['display_name']}")
                        rerun_app()
                    
                    # 顯示已保存的密鑰數量 - 修復後的版本
                    saved_keys = provider_manager.get_api_keys(provider_key)
                    if saved_keys:
                        st.caption(f"🔑 已保存 {len(saved_keys)} 個密鑰")
    else:
        st.markdown("### 🔧 自定義供應商")
        st.info("尚未創建任何自定義供應商")
    
    # 管理按鈕
    st.markdown("---")
    col_create, col_manage = st.columns(2)
    
    with col_create:
        if st.button("➕ 創建自定義供應商", use_container_width=True, type="primary"):
            st.session_state.show_custom_creator = True
            rerun_app()
    
    with col_manage:
        if st.button("🔧 管理自定義供應商", use_container_width=True):
            st.session_state.show_custom_manager = True
            rerun_app()

def show_provider_management():
    """顯示供應商管理界面"""
    if 'selected_provider' not in st.session_state:
        show_provider_selector()
        return
    
    selected_provider = st.session_state.selected_provider
    all_providers = provider_manager.get_all_providers()
    provider_info = all_providers[selected_provider]
    
    # 顯示供應商信息
    if provider_info.get('is_custom'):
        st.subheader(f"{provider_info['icon']} {provider_info['display_name']} (自定義)")
    else:
        st.subheader(f"{provider_info['icon']} {provider_info['name']}")
    
    # 供應商信息
    col_info, col_switch = st.columns([3, 1])
    
    with col_info:
        st.info(f"📋 {provider_info['description']}")
        st.caption(f"🔗 API 類型: {provider_info['api_type']} | 端點: {provider_info['base_url']}")
        
        # 支持的功能
        features_badges = " ".join([f"`{feature}`" for feature in provider_info['features']])
        st.markdown(f"**支持功能**: {features_badges}")
    
    with col_switch:
        if st.button("🔄 切換供應商", use_container_width=True):
            del st.session_state.selected_provider
            rerun_app()
    
    # 管理模式選擇
    management_tabs = st.tabs(["🔑 密鑰管理", "🤖 模型發現", "🎨 圖像生成", "📊 性能監控"])
    
    with management_tabs[0]:
        show_provider_key_management(selected_provider, provider_info)
    
    with management_tabs[1]:
        show_provider_model_discovery(selected_provider, provider_info)
    
    with management_tabs[2]:
        show_image_generation(selected_provider, provider_info)
    
    with management_tabs[3]:
        show_provider_performance(selected_provider, provider_info)

# 這裡需要加入所有其他函數，為了節省空間，我只展示修復的關鍵部分

def show_provider_key_management(provider: str, provider_info: Dict):
    """顯示供應商密鑰管理"""
    st.markdown("### 🔑 密鑰管理")
    
    # 現有密鑰列表
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
    
    # 高級設置
    with st.expander("🔧 高級設置"):
        custom_base_url = st.text_input(
            "自定義端點 URL:",
            value=provider_info['base_url'],
            help="留空使用默認端點"
        )
        
        notes = st.text_area("備註:", placeholder="記錄此密鑰的用途...")
        is_default = st.checkbox("設為默認密鑰")
    
    # 保存按鈕
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

# 在這裡加入之前所有的其他函數，如 show_provider_model_discovery, show_image_generation, show_provider_performance 等

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
    
    if 'provider_selection_mode' not in st.session_state:
        st.session_state.provider_selection_mode = False

# 初始化
init_session_state()

# 檢查 API 配置
api_configured = st.session_state.api_config.get('api_key') is not None and st.session_state.api_config.get('api_key') != ''

# 側邊欄
with st.sidebar:
    st.markdown("### 🏢 供應商狀態")
    
    if 'selected_provider' in st.session_state:
        provider = st.session_state.selected_provider
        all_providers = provider_manager.get_all_providers()
        provider_info = all_providers.get(provider, {})
        
        if provider_info.get('is_custom'):
            st.success(f"{provider_info['icon']} {provider_info['display_name']} (自定義)")
        else:
            st.success(f"{provider_info['icon']} {provider_info['name']}")
        
        if api_configured:
            st.success("🟢 API 已配置")
            if st.session_state.api_config.get('key_name'):
                st.caption(f"🔑 {st.session_state.api_config['key_name']}")
        else:
            st.error("🔴 API 未配置")
    else:
        st.info("未選擇供應商")
    
    st.markdown("---")
    
    # 統計信息
    st.markdown("### 📊 統計")
    total_keys = len(provider_manager.get_api_keys())
    total_models = len(provider_manager.get_provider_models())
    custom_providers_count = len(provider_manager.get_custom_providers())
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("密鑰數", total_keys)
        st.metric("自定義供應商", custom_providers_count)
    with col_stat2:
        st.metric("模型數", total_models)
        history = provider_manager.get_generation_history(10)
        st.metric("生成歷史", len(history))

# 主標題
st.title("🎨 Flux & SD Generator Pro - 自設供應商版")

# 主要內容
if 'show_custom_creator' in st.session_state and st.session_state.show_custom_creator:
    show_custom_provider_creator()
    if st.button("⬅️ 返回", key="back_from_creator"):
        del st.session_state.show_custom_creator
        rerun_app()

elif 'selected_provider' not in st.session_state:
    show_provider_selector()
else:
    show_provider_management()

# 頁腳
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🛠️ <strong>自設供應商</strong> | 
    🎨 <strong>完整圖像生成</strong> | 
    📊 <strong>智能管理</strong> | 
    🔄 <strong>靈活切換</strong>
    <br><br>
    <small>支援自定義 API 供應商、完整的圖像生成功能和歷史管理</small>
</div>
""", unsafe_allow_html=True)
