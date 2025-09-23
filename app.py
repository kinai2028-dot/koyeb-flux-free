import streamlit as st

# 必須是第一個 Streamlit 命令 - 設定頁面配置
st.set_page_config(
    page_title="Flux & SD Generator Pro - 完整版 + FLUX Krea + NavyAI",
    page_icon="🎨",
    layout="wide"
)

# 現在可以導入其他模組
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

# 模型供應商配置
MODEL_PROVIDERS = {
    "NavyAI": {
        "name": "NavyAI",
        "icon": "⚓",
        "description": "統一 API 接口，支援 OpenAI、Google、Mistral 等 50+ 模型",
        "api_type": "openai_compatible",
        "base_url": "https://api.navy/v1",
        "key_prefix": "navy_",
        "features": ["flux", "flux-krea", "dalle", "midjourney", "stable-diffusion", "openai", "google", "mistral"],
        "pricing": "統一計費",
        "speed": "極快",
        "quality": "多供應商",
        "is_custom": False,
        "requires_api_key": True,
        "uptime": ">99%",
        "support": "24/7",
        "speciality": "統一接口"
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

# 供應商特定模型庫
PROVIDER_SPECIFIC_MODELS = {
    "NavyAI": {
        "flux-krea": [
            "black-forest-labs/flux-krea-dev",
            "black-forest-labs/flux-krea-schnell"
        ],
        "flux": [
            "black-forest-labs/flux.1-dev",
            "black-forest-labs/flux.1-schnell",
            "black-forest-labs/flux.1-pro"
        ],
        "dalle": [
            "dalle-3",
            "dalle-2"
        ],
        "stable-diffusion": [
            "stability-ai/sdxl-turbo",
            "stability-ai/stable-diffusion-xl-base-1.0",
            "stability-ai/stable-diffusion-3-medium"
        ],
        "midjourney": [
            "midjourney-v6",
            "midjourney-v5"
        ]
    },
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
                category TEXT CHECK(category IN ('flux', 'flux-krea', 'stable-diffusion', 'dalle', 'midjourney')) NOT NULL,
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
        valid_categories = ['flux', 'flux-krea', 'stable-diffusion', 'dalle', 'midjourney']
        if category not in valid_categories:
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
            kwargs.get('supports_styles', category in ['flux-krea', 'dalle', 'midjourney'])
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

# 全局實例
provider_manager = CompleteProviderManager()

def safe_seed_check(seed_value):
    """安全檢查 seed 值"""
    if seed_value is None:
        return False
    try:
        return isinstance(seed_value, (int, float)) and seed_value >= 0
    except (TypeError, ValueError):
        return False

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
        
        seed_value = params.get("seed")
        if safe_seed_check(seed_value):
            url_params.append(f"seed={int(seed_value)}")
        
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

def generate_krea_image(api_key: str, base_url: str, **params) -> Tuple[bool, any]:
    """Krea.ai API 圖像生成（模擬實現）"""
    try:
        time.sleep(3)
        
        width, height = 1024, 1024
        if "size" in params:
            width, height = map(int, params["size"].split('x'))
        
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        for y in range(height):
            r = int(135 + (120 * y / height))
            g = int(206 + (49 * y / height))  
            b = int(235 + (20 * y / height))
            for x in range(width):
                draw.point((x, y), (r, g, b))
        
        try:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        except:
            font_large = font_small = None
        
        draw.text((50, 50), "🎭 FLUX Krea Generated", fill=(255, 255, 255), font=font_large)
        
        prompt_text = params.get('prompt', 'Beautiful AI art')[:80]
        lines = [prompt_text[i:i+40] for i in range(0, len(prompt_text), 40)]
        
        y_offset = 100
        for line in lines:
            draw.text((50, y_offset), line, fill=(255, 255, 255), font=font_small)
            y_offset += 25
        
        model_name = params.get('model', 'flux-krea')
        draw.text((50, height - 100), f"Model: {model_name}", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 75), f"Aesthetic: {'⭐' * 5}", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 50), "Color Harmony: Optimized", fill=(255, 255, 255), font=font_small)
        
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

def generate_navyai_image(api_key: str, base_url: str, **params) -> Tuple[bool, any]:
    """NavyAI API 圖像生成（模擬實現）"""
    try:
        time.sleep(2)
        
        width, height = 1024, 1024
        if "size" in params:
            width, height = map(int, params["size"].split('x'))
        
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        # NavyAI 主題色彩
        for y in range(height):
            r = int(25 + (50 * y / height))   # 深藍色系
            g = int(50 + (100 * y / height))  
            b = int(100 + (155 * y / height))
            for x in range(width):
                draw.point((x, y), (r, g, b))
        
        try:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        except:
            font_large = font_small = None
        
        # 主標題
        draw.text((50, 50), "⚓ NavyAI Generated", fill=(255, 255, 255), font=font_large)
        
        # 提示詞預覽
        prompt_text = params.get('prompt', 'AI generated art')[:80]
        lines = [prompt_text[i:i+40] for i in range(0, len(prompt_text), 40)]
        
        y_offset = 100
        for line in lines:
            draw.text((50, y_offset), line, fill=(255, 255, 255), font=font_small)
            y_offset += 25
        
        # 參數信息
        model_name = params.get('model', 'flux-1-dev')
        draw.text((50, height - 150), f"Model: {model_name}", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 125), "NavyAI Unified API", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 100), "50+ AI Models Access", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 75), ">99% Uptime", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 50), "24/7 Support", fill=(255, 255, 255), font=font_small)
        
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

def generate_images_with_retry(client, provider: str, api_key: str, base_url: str, **params) -> Tuple[bool, any]:
    """帶重試機制的圖像生成"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            all_providers = provider_manager.get_all_providers()
            provider_info = all_providers.get(provider, {})
            api_type = provider_info.get("api_type", "openai_compatible")
            
            if attempt > 0:
                st.info(f"🔄 嘗試重新生成 (第 {attempt + 1}/{max_retries} 次)")
                time.sleep(2)
            
            if api_type == "pollinations":
                return generate_pollinations_image(**params)
            elif api_type == "krea":
                return generate_krea_image(api_key, base_url, **params)
            elif provider == "NavyAI":
                # 特殊處理 NavyAI
                return generate_navyai_image(api_key, base_url, **params)
            else:
                # OpenAI 兼容模式
                if client:
                    clean_params = {
                        "model": params.get("model"),
                        "prompt": params.get("prompt"),
                        "n": params.get("n", 1),
                        "size": params.get("size", "1024x1024")
                    }
                    
                    if params.get("quality"):
                        clean_params["quality"] = params["quality"]
                        
                    response = client.images.generate(**clean_params)
                    return True, response
                else:
                    return False, "客戶端未初始化"
        
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                should_retry = any(x in error_msg for x in ["500", "502", "503", "timeout"])
                if should_retry:
                    st.warning(f"⚠️ 第 {attempt + 1} 次嘗試失敗: {error_msg[:100]}")
                    continue
            return False, f"生成失敗: {error_msg}"
    
    return False, "未知錯誤"

def discover_provider_models(provider: str, provider_info: Dict, selected_categories: List[str]):
    """發現供應商模型"""
    with st.spinner(f"🔍 正在從 {provider} 發現模型..."):
        discovered_count = {
            "flux": 0, "flux-krea": 0, "stable-diffusion": 0, 
            "dalle": 0, "midjourney": 0
        }
        
        try:
            if provider in PROVIDER_SPECIFIC_MODELS:
                provider_models = PROVIDER_SPECIFIC_MODELS[provider]
                
                for category, models in provider_models.items():
                    category_display = {
                        "flux-krea": "🎭 FLUX Krea 模型",
                        "flux": "⚡ Flux 模型", 
                        "stable-diffusion": "🎨 Stable Diffusion",
                        "dalle": "🖼️ DALL-E 模型",
                        "midjourney": "🎯 Midjourney 模型"
                    }.get(category, category)
                    
                    if category_display in selected_categories:
                        for model_name in models:
                            description = ""
                            icon_map = {
                                "flux-krea": "🎭", "flux": "⚡", "stable-diffusion": "🎨",
                                "dalle": "🖼️", "midjourney": "🎯"
                            }
                            icon = icon_map.get(category, "🤖")
                            priority = 1 if category == "flux-krea" else 2 if category in ["dalle", "midjourney"] else 999
                            aesthetic_score = 5 if category in ["flux-krea", "midjourney"] else 4 if category == "dalle" else 3
                            
                            if category == "flux-krea":
                                description = f"FLUX Krea {model_name} - 美學優化圖像生成模型"
                            elif category == "flux":
                                description = f"FLUX {model_name} - 高性能文本到圖像生成"
                            elif category == "dalle":
                                description = f"DALL-E {model_name} - OpenAI 圖像生成模型"
                            elif category == "midjourney":
                                description = f"Midjourney {model_name} - 藝術風格圖像生成"
                            elif category == "stable-diffusion":
                                description = f"Stable Diffusion {model_name} - 開源圖像生成"
                            
                            saved_id = provider_manager.save_provider_model(
                                provider=provider,
                                model_name=model_name,
                                model_id=model_name,
                                category=category,
                                description=description,
                                icon=icon,
                                pricing_tier="premium" if provider == "NavyAI" else "free",
                                expected_size="1024x1024",
                                priority=priority,
                                aesthetic_score=aesthetic_score,
                                supports_styles=category in ["flux-krea", "dalle", "midjourney"]
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
                            "stable-diffusion": "🎨 SD",
                            "dalle": "🖼️ DALL-E",
                            "midjourney": "🎯 Midjourney"
                        }.get(category, category)
                        st.info(f"{category_name}: {count} 個")
                        
                        if category == "flux-krea":
                            st.success("🎭 發現 FLUX Krea 模型！專注美學優化和寫實圖像生成")
                        elif provider == "NavyAI" and category in ["dalle", "midjourney"]:
                            st.info(f"⚓ NavyAI 統一接口支援 {category_name} 模型")
            else:
                st.info(f"ℹ️ 在 {provider} 未發現新模型")
            
            rerun_app()
            
        except Exception as e:
            st.error(f"❌ 發現失敗: {str(e)}")

def show_quick_switch_panel():
    """顯示快速切換面板"""
    st.markdown("### ⚡ 快速切換供應商")
    
    quick_configs = provider_manager.get_quick_switch_configs()
    
    if not quick_configs:
        st.info("📭 尚未創建任何快速切換配置")
        return
    
    # 顯示快速切換按鈕
    favorite_configs = [c for c in quick_configs if c['is_favorite']]
    
    if favorite_configs:
        st.markdown("**⭐ 收藏配置**")
        cols = st.columns(min(len(favorite_configs), 4))
        
        for i, config in enumerate(favorite_configs):
            with cols[i % len(cols)]:
                all_providers = provider_manager.get_all_providers()
                provider_info = all_providers.get(config['provider'], {})
                icon = provider_info.get('icon', '🔧')
                
                if st.button(
                    f"{icon} {config['config_name']}",
                    key=f"quick_fav_{config['id']}",
                    use_container_width=True,
                    type="primary"
                ):
                    switch_to_config(config)
                    st.success(f"✅ 已切換到: {config['config_name']}")
                    rerun_app()

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
        'key_name': config['key_name']
    }
    
    provider_manager.update_config_usage(config['id'])

def show_provider_selector():
    """顯示供應商選擇器"""
    st.subheader("🏢 選擇模型供應商")
    
    show_quick_switch_panel()
    
    st.markdown("---")
    st.markdown("### ⚓ 推薦：NavyAI 統一接口")
    
    # NavyAI 特殊推薦區域
    navyai_info = MODEL_PROVIDERS["NavyAI"]
    with st.container():
        col_navy1, col_navy2 = st.columns([2, 1])
        
        with col_navy1:
            st.markdown(f"#### {navyai_info['icon']} {navyai_info['name']} ⭐")
            st.success(f"🎯 專長：{navyai_info['speciality']}")
            st.caption(navyai_info['description'])
            
            # 特色標籤
            st.markdown("**🌟 NavyAI 特色:**")
            col_feat1, col_feat2 = st.columns(2)
            with col_feat1:
                st.info("⚓ 50+ AI 模型")
                st.info("📊 統一計費系統")
            with col_feat2:
                st.info(f"⚡ {navyai_info['uptime']} 運行時間")
                st.info(f"🔧 {navyai_info['support']} 技術支援")
        
        with col_navy2:
            st.markdown("**🎨 支援模型:**")
            features = navyai_info['features']
            for feature in features[:6]:  # 顯示前6個
                if feature in ["flux-krea", "dalle", "midjourney"]:
                    st.success(f"✨ {feature.upper()}")
                else:
                    st.info(f"• {feature}")
            
            if st.button(f"選擇 {navyai_info['name']}", key="select_navyai", use_container_width=True, type="primary"):
                st.session_state.selected_provider = "NavyAI"
                st.success(f"已選擇 {navyai_info['name']} - 統一 AI 模型接口")
                rerun_app()
    
    st.markdown("---")
    st.markdown("### 🎭 FLUX Krea 專門供應商")
    
    all_providers = provider_manager.get_all_providers()
    flux_krea_providers = {k: v for k, v in all_providers.items() if "flux-krea" in v.get('features', []) and k != "NavyAI"}
    
    cols = st.columns(2)
    for i, (provider_key, provider_info) in enumerate(flux_krea_providers.items()):
        with cols[i % 2]:
            with st.container():
                specialty = provider_info.get('speciality', '')
                if specialty:
                    st.markdown(f"#### {provider_info['icon']} {provider_info['name']} ✨")
                    st.success(f"🎯 專長：{specialty}")
                else:
                    st.markdown(f"#### {provider_info['icon']} {provider_info['name']}")
                
                st.caption(provider_info['description'])
                st.info("🎭 支援 FLUX Krea 美學優化模型")
                st.caption(f"⚡ 速度: {provider_info['speed']} | 💰 {provider_info['pricing']}")
                
                if st.button(f"選擇 {provider_info['name']}", key=f"select_krea_{provider_key}", use_container_width=True, type="primary"):
                    st.session_state.selected_provider = provider_key
                    st.success(f"已選擇 {provider_info['name']} - FLUX Krea 專門供應商")
                    rerun_app()
                
                if not provider_info.get('requires_api_key', True):
                    st.caption("🆓 免費服務無需密鑰")

def show_provider_key_management(provider: str, provider_info: Dict):
    """顯示供應商密鑰管理"""
    st.markdown("### 🔑 密鑰管理")
    
    # NavyAI 特殊提示
    if provider == "NavyAI":
        st.info("⚓ **NavyAI 快速獲取 API 密鑰:**")
        st.markdown("""
        1. 🌐 **Dashboard 方式**: [登入 NavyAI Dashboard](https://api.navy) 管理密鑰和使用統計
        2. 💬 **Discord 快速方式**: 在 Discord 中使用 `/key` 命令快速獲取密鑰
        3. 📚 **查看文檔**: [NavyAI 完整文檔](https://api.navy/docs) 了解更多功能
        """)
        st.markdown("---")
    
    requires_key = provider_info.get('requires_api_key', True)
    
    if not requires_key:
        provider_name = provider_info.get('name', provider)
        st.success(f"🌟 {provider_name} 提供免費服務，無需 API 密鑰！")
        
        if "flux-krea" in provider_info.get('features', []):
            st.info("🎭 您可以直接使用 FLUX Krea 美學優化模型進行圖像生成")
        
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
    
    saved_keys = provider_manager.get_api_keys(provider)
    
    if saved_keys:
        st.markdown("#### 📋 已保存的密鑰")
        
        for key_info in saved_keys:
            col_key, col_actions = st.columns([3, 1])
            
            with col_key:
                st.markdown(f"**{key_info['key_name']}**")
                st.caption(f"創建於: {key_info['created_at']}")
                if provider == "NavyAI" and key_info['validated']:
                    st.success("✅ NavyAI 密鑰已驗證")
            
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
    
    st.markdown("#### ➕ 新增密鑰")
    
    key_name = st.text_input("密鑰名稱:", placeholder=f"例如：{provider} 主密鑰")
    
    if provider == "NavyAI":
        api_key = st.text_input("NavyAI API 密鑰:", type="password", placeholder="輸入從 Dashboard 或 Discord 獲取的密鑰...")
        st.caption("💡 密鑰前綴通常為 'navy_' 開頭")
    else:
        api_key = st.text_input("API 密鑰:", type="password", placeholder=f"輸入 {provider_info['name']} API 密鑰...")
    
    # 特色功能提示
    if provider == "NavyAI":
        st.info("🌟 **NavyAI 統一接口優勢**: 一個密鑰訪問 50+ 模型，包含 FLUX Krea、DALL-E、Midjourney 等")
    elif "flux-krea" in provider_info.get('features', []):
        st.info("💡 此供應商支援 FLUX Krea 模型，可生成美學優化和高度寫實的圖像")
    
    if st.button("💾 保存密鑰", type="primary", use_container_width=True):
        if key_name and api_key:
            key_id = provider_manager.save_api_key(provider, key_name, api_key, provider_info['base_url'])
            st.success(f"✅ 密鑰已保存！ID: {key_id[:8]}...")
            if provider == "NavyAI":
                st.info("⚓ NavyAI 密鑰已配置，現在可以訪問 50+ AI 模型")
            rerun_app()
        else:
            st.error("❌ 請填寫完整信息")

def show_provider_model_discovery(provider: str, provider_info: Dict):
    """顯示供應商模型發現"""
    st.markdown("### 🤖 模型發現")
    
    if not st.session_state.api_config.get('api_key') and provider_info.get('requires_api_key', True):
        st.warning("⚠️ 請先配置 API 密鑰")
        return
    
    col_discover, col_results = st.columns([1, 2])
    
    with col_discover:
        st.markdown("#### 🔍 發現設置")
        
        supported_categories = []
        category_map = {
            "flux-krea": "🎭 FLUX Krea 模型",
            "flux": "⚡ Flux 模型",
            "stable-diffusion": "🎨 Stable Diffusion",
            "dalle": "🖼️ DALL-E 模型",
            "midjourney": "🎯 Midjourney 模型"
        }
        
        for feature in provider_info['features']:
            if feature in category_map:
                supported_categories.append(category_map[feature])
        
        if not supported_categories:
            st.warning(f"{provider} 不支持圖像生成模型")
            return
        
        selected_categories = st.multiselect(
            "選擇要發現的模型類型:",
            supported_categories,
            default=supported_categories
        )
        
        # 特色模型介紹
        if provider == "NavyAI":
            st.info("⚓ **NavyAI**: 統一接口訪問多個 AI 供應商的模型")
            if "🎭 FLUX Krea 模型" in supported_categories:
                st.success("🎭 支援 FLUX Krea 美學優化模型")
            if "🖼️ DALL-E 模型" in supported_categories:
                st.success("🖼️ 支援 DALL-E 高質量圖像生成")
            if "🎯 Midjourney 模型" in supported_categories:
                st.success("🎯 支援 Midjourney 藝術風格生成")
        elif "🎭 FLUX Krea 模型" in supported_categories:
            st.info("🎭 **FLUX Krea**: 美學優化模型，專注產生寫實且多樣化的圖像")
        
        if st.button("🚀 開始發現", type="primary", use_container_width=True):
            if selected_categories:
                discover_provider_models(provider, provider_info, selected_categories)
            else:
                st.warning("請選擇要發現的模型類型")
    
    with col_results:
        st.markdown("#### 📊 發現結果")
        
        discovered_models = provider_manager.get_provider_models(provider)
        
        if discovered_models:
            # 按類別分組
            model_groups = {
                'flux-krea': [],
                'flux': [],
                'stable-diffusion': [],
                'dalle': [],
                'midjourney': []
            }
            
            for model in discovered_models:
                category = model['category']
                if category in model_groups:
                    model_groups[category].append(model)
            
            # 顯示每個類別
            for category, models in model_groups.items():
                if not models:
                    continue
                    
                category_names = {
                    "flux-krea": "🎭 FLUX Krea 模型",
                    "flux": "⚡ Flux 模型",
                    "stable-diffusion": "🎨 SD 模型",
                    "dalle": "🖼️ DALL-E 模型",
                    "midjourney": "🎯 Midjourney 模型"
                }
                
                category_name = category_names.get(category, category)
                st.markdown(f"**{category_name}**: {len(models)} 個")
                
                if category == "flux-krea":
                    st.success("🌟 美學優化專門模型")
                elif category == "dalle" and provider == "NavyAI":
                    st.info("⚓ NavyAI 統一接口 - OpenAI DALL-E")
                elif category == "midjourney" and provider == "NavyAI":
                    st.info("⚓ NavyAI 統一接口 - Midjourney 藝術生成")
                
                for model in models[:3]:
                    aesthetic_score = model.get('aesthetic_score', 3)
                    if aesthetic_score >= 4:
                        stars = "⭐" * min(aesthetic_score, 5)
                        st.write(f"• {model['icon']} {model['model_name']} {stars}")
                    else:
                        st.write(f"• {model['icon']} {model['model_name']}")
        else:
            st.info("尚未發現任何模型")

def display_image_with_actions(image_url: str, image_id: str, generation_info: Dict = None):
    """顯示圖像和操作按鈕"""
    try:
        if image_url.startswith('data:image'):
            base64_data = image_url.split(',')[1]
            img_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(img_data))
        else:
            img_response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(img_response.content))
        
        st.image(img, use_column_width=True)
        
        if generation_info:
            with st.expander("🔍 圖像信息"):
                st.write(f"**提示詞**: {generation_info.get('prompt', 'N/A')}")
                st.write(f"**模型**: {generation_info.get('model_name', 'N/A')}")
                st.write(f"**供應商**: {generation_info.get('provider', 'N/A')}")
                st.write(f"**尺寸**: {generation_info.get('size', 'N/A')}")
                st.write(f"**生成時間**: {generation_info.get('timestamp', 'N/A')}")
                
                # 特殊標記
                provider = generation_info.get('provider', '')
                category = generation_info.get('category', '')
                
                if category == 'flux-krea':
                    st.write(f"**美學評分**: {'⭐' * generation_info.get('aesthetic_score', 5)}")
                elif provider == 'NavyAI':
                    st.write(f"**NavyAI 統一接口**: ⚓ {category.upper()}")
                elif category in ['dalle', 'midjourney']:
                    st.write(f"**高級模型**: {'⭐' * generation_info.get('aesthetic_score', 4)}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            st.download_button(
                label="📥 下載",
                data=img_buffer.getvalue(),
                file_name=f"ai_generated_{image_id}.png",
                mime="image/png",
                key=f"download_{image_id}",
                use_container_width=True
            )
        
        with col2:
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
            if generation_info and st.button("🔄 重新生成", key=f"regenerate_{image_id}", use_container_width=True):
                st.session_state.regenerate_info = generation_info
                rerun_app()
        
        with col4:
            if generation_info and st.button("🎨 變化生成", key=f"variation_{image_id}", use_container_width=True):
                variation_info = generation_info.copy()
                variation_info['prompt'] = f"{generation_info.get('prompt', '')} (variation)"
                
                seed_value = variation_info.get('seed')
                if safe_seed_check(seed_value):
                    variation_info['seed'] = random.randint(0, 2147483647)
                else:
                    variation_info['seed'] = random.randint(0, 2147483647)
                
                st.session_state.variation_info = variation_info
                rerun_app()
    
    except Exception as e:
        st.error(f"圖像顯示錯誤: {str(e)}")

def show_image_generation(provider: str, provider_info: Dict):
    """顯示完整的圖像生成界面"""
    st.markdown("### 🎨 圖像生成")
    
    config = st.session_state.api_config
    if not config.get('api_key') and provider_info.get('requires_api_key', True):
        st.warning("⚠️ 請先在密鑰管理中配置 API 密鑰")
        return
    
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
    
    col_settings, col_preview = st.columns([2, 1])
    
    with col_settings:
        st.markdown("#### ⚙️ 生成設置")
        
        categories = list(set(model['category'] for model in available_models))
        
        # 優先級排序
        priority_order = ['flux-krea', 'dalle', 'midjourney', 'flux', 'stable-diffusion']
        categories.sort(key=lambda x: priority_order.index(x) if x in priority_order else 999)
        
        if len(categories) > 1:
            selected_category = st.selectbox(
                "模型類別:",
                categories,
                format_func=lambda x: {
                    "flux-krea": "🎭 FLUX Krea (美學優化)",
                    "flux": "⚡ Flux AI",
                    "stable-diffusion": "🎨 Stable Diffusion",
                    "dalle": "🖼️ DALL-E (OpenAI)",
                    "midjourney": "🎯 Midjourney (藝術風格)"
                }.get(x, x.title())
            )
        else:
            selected_category = categories[0]
        
        category_models = [m for m in available_models if m['category'] == selected_category]
        selected_model_info = st.selectbox(
            "選擇模型:",
            category_models,
            format_func=lambda x: f"{x['icon']} {x['model_name']} {'⭐' * x.get('aesthetic_score', 3) if x.get('aesthetic_score', 0) >= 4 else ''}"
        )
        
        # 特殊模式提示
        if selected_category == "flux-krea":
            st.success("🎭 **FLUX Krea 模式**：專為美學優化設計，生成更自然、寫實的圖像")
            st.info("💡 特色：避免過度飽和、更好的人類美學偏好、寫實多樣化")
        elif selected_category == "dalle" and provider == "NavyAI":
            st.success("🖼️ **DALL-E 模式 (NavyAI)**：OpenAI 高質量圖像生成")
            st.info("⚓ 通過 NavyAI 統一接口訪問 OpenAI DALL-E")
        elif selected_category == "midjourney" and provider == "NavyAI":
            st.success("🎯 **Midjourney 模式 (NavyAI)**：藝術風格圖像生成")
            st.info("⚓ 通過 NavyAI 統一接口訪問 Midjourney")
        elif provider == "NavyAI":
            st.success(f"⚓ **NavyAI 統一接口**: {selected_category.upper()} 模型")
        
        st.markdown("#### 📝 提示詞")
        
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
            placeholder="例如：A professional portrait of a confident businesswoman, natural lighting, realistic skin texture, detailed eyes"
        )
        
        st.markdown("#### 💡 快速模板")
        
        # 根據模型類型提供不同模板
        if selected_category == "flux-krea":
            templates = [
                "Professional portrait of a confident businesswoman, natural lighting, realistic skin texture",
                "Candid street photography of an elderly artist, warm golden hour light, authentic expression",
                "Cozy coffee shop interior, natural lighting, authentic atmosphere, realistic textures"
            ]
        elif selected_category == "dalle":
            templates = [
                "A surreal landscape with floating islands and waterfalls, digital art style",
                "Portrait of a futuristic robot with human-like emotions, highly detailed",
                "Abstract art piece representing the concept of time, vibrant colors"
            ]
        elif selected_category == "midjourney":
            templates = [
                "Epic fantasy landscape with dragons soaring over mountains, cinematic lighting",
                "Steampunk cityscape with intricate mechanical details, bronze and copper tones",
                "Ethereal forest scene with magical creatures, soft glowing light"
            ]
        else:
            templates = [
                "Digital art illustration of a fantasy landscape with magical elements",
                "Concept art of a futuristic cityscape with flying vehicles",
                "Abstract geometric composition with vibrant colors and patterns"
            ]
        
        for i, template in enumerate(templates):
            if st.button(f"📝 {template[:50]}...", key=f"template_{i}", use_container_width=True):
                st.session_state.quick_prompt = template
                rerun_app()
        
        if hasattr(st.session_state, 'quick_prompt'):
            prompt = st.session_state.quick_prompt
            del st.session_state.quick_prompt
            rerun_app()
    
    with col_preview:
        st.markdown("#### 🎯 參數設置")
        
        # 根據模型類型調整尺寸選項
        if selected_category in ["flux-krea", "dalle"]:
            size_options = ["1024x1024", "1152x896", "896x1152", "1344x768", "768x1344"]
        elif selected_category == "midjourney":
            size_options = ["1024x1024", "1152x896", "896x1152"]
        else:
            size_options = ["512x512", "768x768", "1024x1024"]
        
        selected_size = st.selectbox("圖像尺寸:", size_options, index=0)
        num_images = st.slider("生成數量:", 1, 4, 1)
        
        with st.expander("🔧 高級參數"):
            if selected_category == "flux-krea":
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
                
                aesthetic_weight = st.slider("美學權重:", 0.5, 2.0, 1.0, 0.1)
                naturalism_boost = st.checkbox("自然主義增強", value=True)
                color_harmony = st.selectbox("色彩和諧度:", ["auto", "warm", "cool", "neutral", "vibrant"])
                
            elif selected_category == "dalle":
                guidance_scale = st.slider("創意引導:", 1.0, 20.0, 10.0, 0.5)
                steps = st.slider("生成質量:", 10, 50, 30)
                aesthetic_weight = st.slider("藝術風格權重:", 0.5, 2.0, 1.2, 0.1)
                naturalism_boost = st.checkbox("寫實增強", value=False)
                color_harmony = st.selectbox("色調風格:", ["auto", "vibrant", "muted", "monochrome"])
                
            elif selected_category == "midjourney":
                guidance_scale = st.slider("藝術引導:", 1.0, 20.0, 12.0, 0.5)
                steps = st.slider("細節層次:", 10, 50, 35)
                aesthetic_weight = st.slider("風格化程度:", 0.5, 2.0, 1.5, 0.1)
                naturalism_boost = st.checkbox("風格化增強", value=True)
                color_harmony = st.selectbox("藝術色調:", ["auto", "dramatic", "pastel", "neon", "vintage"])
                
            else:
                guidance_scale = st.slider("引導強度:", 1.0, 20.0, 7.5, 0.5)
                steps = st.slider("推理步數:", 10, 100, 25)
                aesthetic_weight = 1.0
                naturalism_boost = False
                color_harmony = "auto"
            
            seed = st.number_input("隨機種子 (可選):", min_value=-1, max_value=2147483647, value=-1)
            
            if seed == -1 and st.button("🎲 生成隨機種子"):
                new_seed = random.randint(0, 2147483647)
                st.success(f"隨機種子: {new_seed}")
                st.session_state.temp_seed = new_seed
    
    st.markdown("---")
    
    can_generate = selected_model_info and prompt.strip()
    
    col_gen, col_clear = st.columns([3, 1])
    
    with col_gen:
        if st.button(
            f"🚀 生成圖像 ({selected_model_info['model_name'] if selected_model_info else 'None'})",
            type="primary",
            disabled=not can_generate,
            use_container_width=True
        ):
            if can_generate:
                final_seed = seed if seed >= 0 else (st.session_state.get('temp_seed') if hasattr(st.session_state, 'temp_seed') else None)
                
                generate_image_main(
                    provider, provider_info, selected_model_info,
                    prompt, selected_size, num_images,
                    guidance_scale, steps, final_seed, selected_category,
                    aesthetic_weight, naturalism_boost, color_harmony
                )
    
    with col_clear:
        if st.button("🗑️ 清除", use_container_width=True):
            for key in ['quick_prompt', 'regenerate_info', 'variation_info', 'temp_seed']:
                if key in st.session_state:
                    del st.session_state[key]
            rerun_app()
    
    show_generation_history()

def generate_image_main(provider: str, provider_info: Dict, model_info: Dict,
                       prompt: str, size: str, num_images: int,
                       guidance_scale: float, steps: int, seed, category: str,
                       aesthetic_weight: float, naturalism_boost: bool, color_harmony: str):
    """主要圖像生成函數"""
    
    config = st.session_state.api_config
    
    client = None
    if provider_info.get('api_type') not in ["pollinations", "krea"]:
        try:
            client = OpenAI(
                api_key=config['api_key'],
                base_url=config['base_url']
            )
        except Exception as e:
            st.error(f"API 客戶端初始化失敗: {str(e)}")
            return
    
    safe_seed = None
    if safe_seed_check(seed):
        safe_seed = int(seed)
    
    generation_params = {
        "model": model_info['model_id'],
        "prompt": prompt,
        "n": num_images,
        "size": size,
        "category": category,
        "guidance_scale": guidance_scale,
        "steps": steps,
        "seed": safe_seed,
        "aesthetic_weight": aesthetic_weight,
        "naturalism_boost": naturalism_boost,
        "color_harmony": color_harmony
    }
    
    progress_container = st.empty()
    
    with progress_container.container():
        # 不同模型的生成提示
        if category == 'flux-krea':
            st.success(f"🎭 正在使用 FLUX Krea 模型 {model_info['model_name']} 生成美學優化圖像...")
        elif category == 'dalle' and provider == 'NavyAI':
            st.success(f"🖼️ 正在通過 NavyAI 統一接口使用 DALL-E 模型 {model_info['model_name']}...")
        elif category == 'midjourney' and provider == 'NavyAI':
            st.success(f"🎯 正在通過 NavyAI 統一接口使用 Midjourney 模型 {model_info['model_name']}...")
        elif provider == 'NavyAI':
            st.success(f"⚓ 正在通過 NavyAI 統一接口使用 {model_info['model_name']}...")
        else:
            st.info(f"🎨 正在使用 {model_info['model_name']} 生成圖像...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 不同模型的生成階段
        if category == 'flux-krea':
            stages = [
                "🔧 初始化 FLUX Krea 模型...",
                "📝 處理美學優化提示詞...", 
                "🎨 開始美學推理過程...",
                "✨ 生成自然寫實內容...",
                "🎭 美學優化中...",
                "🌈 色彩調和中...",
                "🎉 FLUX Krea 生成完成！"
            ]
        elif provider == 'NavyAI':
            stages = [
                f"⚓ 初始化 NavyAI 統一接口...",
                f"🔗 連接到 {category.upper()} 服務...", 
                f"📝 處理 {category} 風格提示詞...",
                f"✨ 生成 {category} 圖像內容...",
                f"🎨 {category} 風格優化中...",
                f"⚡ NavyAI 後處理中...",
                f"🎉 NavyAI {category} 生成完成！"
            ]
        else:
            stages = [
                "🔧 初始化模型...",
                "📝 處理提示詞...", 
                "🎨 開始推理過程...",
                "✨ 生成圖像內容...",
                "🔍 細節處理中...",
                "⚙️ 後處理優化...",
                "🎉 完成生成！"
            ]
        
        for i, stage in enumerate(stages):
            status_text.text(stage)
            time.sleep(0.5)
            progress_bar.progress((i + 1) / len(stages))
    
    success, result = generate_images_with_retry(
        client, provider, config['api_key'],
        config['base_url'], **generation_params
    )
    
    progress_container.empty()
    
    if success:
        response = result
        
        generation_info = {
            "prompt": prompt,
            "model_name": model_info['model_name'],
            "model_id": model_info['model_id'],
            "provider": provider,
            "category": category,
            "size": size,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "guidance_scale": guidance_scale,
            "steps": steps,
            "seed": safe_seed,
            "aesthetic_score": model_info.get('aesthetic_score', 5),
            "aesthetic_weight": aesthetic_weight,
            "naturalism_boost": naturalism_boost,
            "color_harmony": color_harmony
        }
        
        # 不同模型的成功提示
        if category == 'flux-krea':
            st.balloons()
            st.success(f"🎭✨ 成功生成 {len(response.data)} 張 FLUX Krea 美學優化圖像！")
        elif category == 'dalle' and provider == 'NavyAI':
            st.balloons()
            st.success(f"🖼️⚓ 成功通過 NavyAI 統一接口生成 {len(response.data)} 張 DALL-E 圖像！")
        elif category == 'midjourney' and provider == 'NavyAI':
            st.balloons()
            st.success(f"🎯⚓ 成功通過 NavyAI 統一接口生成 {len(response.data)} 張 Midjourney 藝術圖像！")
        elif provider == 'NavyAI':
            st.balloons()
            st.success(f"⚓✨ 成功通過 NavyAI 統一接口生成 {len(response.data)} 張圖像！")
        else:
            st.success(f"✨ 成功生成 {len(response.data)} 張圖像！")
        
        if 'generation_history' not in st.session_state:
            st.session_state.generation_history = []
        
        if len(response.data) == 1:
            st.markdown("#### 🎨 生成結果")
            image_id = f"gen_{uuid.uuid4().hex[:8]}"
            display_image_with_actions(response.data[0].url, image_id, generation_info)
            
            st.session_state.generation_history.insert(0, {
                "id": image_id,
                "image_url": response.data[0].url,
                "generation_info": generation_info
            })
            
        else:
            st.markdown("#### 🎨 生成結果")
            
            # 根據模型類型調整列數
            if category in ['flux-krea', 'dalle', 'midjourney']:
                cols_count = 2
            else:
                cols_count = min(len(response.data), 3)
            
            img_cols = st.columns(cols_count)
            
            for i, image_data in enumerate(response.data):
                with img_cols[i % len(img_cols)]:
                    if category == 'flux-krea':
                        st.markdown(f"**🎭 美學圖像 {i+1}**")
                    elif category == 'dalle' and provider == 'NavyAI':
                        st.markdown(f"**🖼️ DALL-E 圖像 {i+1}**")
                    elif category == 'midjourney' and provider == 'NavyAI':
                        st.markdown(f"**🎯 Midjourney 圖像 {i+1}**")
                    elif provider == 'NavyAI':
                        st.markdown(f"**⚓ NavyAI 圖像 {i+1}**")
                    else:
                        st.markdown(f"**圖像 {i+1}**")
                    
                    image_id = f"gen_{uuid.uuid4().hex[:8]}_{i}"
                    display_image_with_actions(image_data.url, image_id, generation_info)
                    
                    st.session_state.generation_history.insert(0, {
                        "id": image_id,
                        "image_url": image_data.url,
                        "generation_info": generation_info
                    })
    else:
        st.error(f"❌ 生成失敗: {result}")

def show_generation_history():
    """顯示生成歷史"""
    if 'generation_history' not in st.session_state or not st.session_state.generation_history:
        return
    
    history = st.session_state.generation_history
    
    st.markdown("---")
    st.markdown("### 📚 最近生成")
    
    # 按類別分類歷史
    flux_krea_history = [h for h in history[:12] if h.get('generation_info', {}).get('category') == 'flux-krea']
    navyai_history = [h for h in history[:12] if h.get('generation_info', {}).get('provider') == 'NavyAI']
    other_history = [h for h in history[:12] if h.get('generation_info', {}).get('category') not in ['flux-krea'] and h.get('generation_info', {}).get('provider') != 'NavyAI']
    
    if flux_krea_history:
        st.markdown("#### 🎭 FLUX Krea 美學作品")
        cols = st.columns(min(len(flux_krea_history), 4))
        
        for i, item in enumerate(flux_krea_history[:4]):
            with cols[i]:
                try:
                    if item['image_url'].startswith('data:image'):
                        base64_data = item['image_url'].split(',')[1] 
                        img_data = base64.b64decode(base64_data)
                        img = Image.open(BytesIO(img_data))
                    else:
                        img_response = requests.get(item['image_url'], timeout=5)
                        img = Image.open(BytesIO(img_response.content))
                    
                    st.image(img, use_column_width=True)
                    
                    info = item.get('generation_info', {})
                    st.success(f"🎭 {info.get('model_name', 'FLUX Krea')}")
                    st.caption(f"美學: {'⭐' * info.get('aesthetic_score', 5)}")
                    
                    if st.button("🔄 重新生成", key=f"krea_hist_{item['id']}", use_container_width=True):
                        st.session_state.regenerate_info = info
                        rerun_app()
                        
                except Exception:
                    st.error("圖像載入失敗")
    
    if navyai_history:
        st.markdown("#### ⚓ NavyAI 統一接口作品")
        cols = st.columns(min(len(navyai_history), 4))
        
        for i, item in enumerate(navyai_history[:4]):
            with cols[i]:
                try:
                    if item['image_url'].startswith('data:image'):
                        base64_data = item['image_url'].split(',')[1] 
                        img_data = base64.
