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
    page_title="Flux & SD Generator Pro - 快速切換版",
    page_icon="🎨",
    layout="wide"
)

# 模型供應商配置
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
    }
}

# 模型識別規則和供應商特定模型庫
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
        "providers": ["Navy", "Together AI", "Fireworks AI", "Hugging Face", "Replicate"]
    }
}

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
class QuickSwitchProviderManager:
    def __init__(self):
        self.db_path = "quick_switch_providers.db"
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
        
        # 快速切換配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quick_switch_configs (
                id TEXT PRIMARY KEY,
                config_name TEXT UNIQUE NOT NULL,
                provider TEXT NOT NULL,
                api_key_id TEXT NOT NULL,
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
                image_url TEXT,
                image_data TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
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
    
    def save_quick_switch_config(self, config_name: str, provider: str, api_key_id: str,
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
    
    def delete_api_key(self, key_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
        conn.commit()
        conn.close()

# 全局實例
provider_manager = QuickSwitchProviderManager()

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

def show_quick_switch_panel():
    """顯示快速切換面板"""
    st.markdown("### ⚡ 快速切換供應商")
    
    quick_configs = provider_manager.get_quick_switch_configs()
    all_providers = provider_manager.get_all_providers()
    
    if not quick_configs:
        st.info("📭 尚未創建任何快速切換配置")
        return
    
    # 顯示快速切換按鈕
    favorite_configs = [c for c in quick_configs if c['is_favorite']]
    other_configs = [c for c in quick_configs if not c['is_favorite']]
    
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
    
    # 原有的供應商選擇界面
    all_providers = provider_manager.get_all_providers()
    default_providers = {k: v for k, v in all_providers.items() if not v.get('is_custom', False)}
    
    if default_providers:
        st.markdown("### 🏭 預設供應商")
        
        cols = st.columns(3)
        for i, (provider_key, provider_info) in enumerate(default_providers.items()):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"#### {provider_info['icon']} {provider_info['name']}")
                    st.caption(provider_info['description'])
                    
                    if st.button(f"選擇 {provider_info['name']}", key=f"select_default_{provider_key}", use_container_width=True):
                        st.session_state.selected_provider = provider_key
                        st.success(f"已選擇 {provider_info['name']}")
                        rerun_app()
                    
                    saved_keys = provider_manager.get_api_keys(provider_key)
                    if saved_keys:
                        st.caption(f"🔑 已保存 {len(saved_keys)} 個密鑰")

def show_provider_key_management(provider: str, provider_info: Dict):
    """顯示供應商密鑰管理"""
    st.markdown("### 🔑 密鑰管理")
    
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

def show_provider_model_discovery(provider: str, provider_info: Dict):
    """顯示供應商模型發現"""
    st.markdown("### 🤖 模型發現")
    
    if not st.session_state.api_config.get('api_key'):
        st.warning("⚠️ 請先配置 API 密鑰")
        return
    
    col_discover, col_results = st.columns([1, 2])
    
    with col_discover:
        st.markdown("#### 🔍 發現設置")
        
        supported_categories = []
        if "flux" in provider_info['features']:
            supported_categories.append("⚡ Flux 模型")
        if "stable-diffusion" in provider_info['features']:
            supported_categories.append("🎨 Stable Diffusion")
        
        if not supported_categories:
            st.warning(f"{provider} 不支持 Flux 或 SD 模型")
            return
        
        selected_categories = st.multiselect(
            "選擇要發現的模型類型:",
            supported_categories,
            default=supported_categories
        )
        
        if st.button("🚀 開始發現", type="primary", use_container_width=True):
            if selected_categories:
                discover_provider_models(provider, provider_info, selected_categories)
            else:
                st.warning("請選擇要發現的模型類型")
    
    with col_results:
        st.markdown("#### 📊 發現結果")
        
        discovered_models = provider_manager.get_provider_models(provider)
        
        if discovered_models:
            flux_models = [m for m in discovered_models if m['category'] == 'flux']
            sd_models = [m for m in discovered_models if m['category'] == 'stable-diffusion']
            
            if flux_models:
                st.markdown(f"**⚡ Flux 模型**: {len(flux_models)} 個")
                for model in flux_models[:3]:
                    st.write(f"• {model['icon']} {model['model_name']}")
            
            if sd_models:
                st.markdown(f"**🎨 SD 模型**: {len(sd_models)} 個")
                for model in sd_models[:3]:
                    st.write(f"• {model['icon']} {model['model_name']}")
            
            if len(discovered_models) > 6:
                st.caption(f"... 還有 {len(discovered_models) - 6} 個模型")
        else:
            st.info("尚未發現任何模型")

def discover_provider_models(provider: str, provider_info: Dict, selected_categories: List[str]):
    """發現供應商模型"""
    api_type = provider_info.get("api_type", "openai_compatible")
    config = st.session_state.api_config
    
    with st.spinner(f"🔍 正在從 {provider} 發現模型..."):
        discovered_count = {"flux": 0, "stable-diffusion": 0}
        
        try:
            if api_type == "huggingface":
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

def show_image_generation(provider: str, provider_info: Dict):
    """顯示圖像生成界面"""
    st.markdown("### 🎨 圖像生成")
    st.info("🚀 圖像生成功能開發中，敬請期待！")

def show_provider_performance(provider: str, provider_info: Dict):
    """顯示供應商性能監控"""
    st.markdown("### 📊 性能監控")
    st.info("📊 性能監控功能開發中，敬請期待！")

def show_provider_management():
    """顯示供應商管理界面"""
    if 'selected_provider' not in st.session_state:
        show_provider_selector()
        return
    
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
        show_image_generation(selected_provider, provider_info)
    
    with management_tabs[3]:
        show_provider_performance(selected_provider, provider_info)

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
    
    quick_configs = provider_manager.get_quick_switch_configs()
    favorite_configs = [c for c in quick_configs if c['is_favorite']]
    
    if favorite_configs:
        st.markdown("#### 🌟 收藏配置")
        for config in favorite_configs[:3]:
            all_providers = provider_manager.get_all_providers()
            provider_info = all_providers.get(config['provider'], {})
            icon = provider_info.get('icon', '🔧')
            
            if st.button(
                f"{icon} {config['config_name']}",
                key=f"sidebar_quick_{config['id']}",
                use_container_width=True
            ):
                switch_to_config(config)
                st.success(f"✅ 已切換到: {config['config_name']}")
                rerun_app()
    
    st.markdown("---")
    
    st.markdown("### 📊 統計")
    total_keys = len(provider_manager.get_api_keys())
    total_configs = len(quick_configs)
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("密鑰數", total_keys)
    with col_stat2:
        st.metric("快速配置", total_configs)

# 主標題
st.title("🎨 Flux & SD Generator Pro - 快速切換版")

# 主要內容
if 'selected_provider' not in st.session_state:
    show_provider_selector()
else:
    show_provider_management()

# 頁腳
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    ⚡ <strong>快速切換</strong> | 
    🎨 <strong>一鍵配置</strong> | 
    📊 <strong>使用統計</strong> | 
    ⭐ <strong>收藏管理</strong>
    <br><br>
    <small>支援快速切換不同供應商配置，提升工作效率</small>
</div>
""", unsafe_allow_html=True)
