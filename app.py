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

# 模型識別規則和供應商特定模型庫（與之前相同）
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
        
        # 快速切換配置表 - 新增
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
    
    def save_quick_switch_config(self, config_name: str, provider: str, api_key_id: str,
                                default_model_id: str = "", notes: str = "", is_favorite: bool = False) -> str:
        """保存快速切換配置"""
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
        """獲取快速切換配置"""
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
        """更新配置使用次數和時間"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE quick_switch_configs 
            SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (config_id,))
        
        conn.commit()
        conn.close()
    
    def delete_quick_switch_config(self, config_id: str):
        """刪除快速切換配置"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM quick_switch_configs WHERE id = ?", (config_id,))
        conn.commit()
        conn.close()
    
    # 其他方法保持不變...
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

def show_quick_switch_panel():
    """顯示快速切換面板"""
    st.markdown("### ⚡ 快速切換供應商")
    
    # 獲取快速切換配置
    quick_configs = provider_manager.get_quick_switch_configs()
    all_providers = provider_manager.get_all_providers()
    
    if not quick_configs:
        st.info("📭 尚未創建任何快速切換配置")
        st.markdown("💡 **提示**: 在密鑰管理中保存密鑰後，可以創建快速切換配置")
        return
    
    # 顯示快速切換按鈕
    st.markdown("#### 🚀 一鍵切換")
    
    # 收藏的配置優先顯示
    favorite_configs = [c for c in quick_configs if c['is_favorite']]
    other_configs = [c for c in quick_configs if not c['is_favorite']]
    
    if favorite_configs:
        st.markdown("**⭐ 收藏配置**")
        cols = st.columns(min(len(favorite_configs), 4))
        
        for i, config in enumerate(favorite_configs):
            with cols[i % len(cols)]:
                provider_info = all_providers.get(config['provider'], {})
                icon = provider_info.get('icon', '🔧')
                
                # 狀態指示器
                status_icon = "🟢" if config['validated'] else "🟡"
                
                button_text = f"{icon} {config['config_name']}"
                
                if st.button(
                    button_text,
                    key=f"quick_fav_{config['id']}",
                    use_container_width=True,
                    type="primary"
                ):
                    switch_to_config(config)
                    st.success(f"✅ 已切換到: {config['config_name']}")
                    rerun_app()
                
                # 顯示使用次數和狀態
                st.caption(f"{status_icon} 使用 {config['usage_count']} 次")
    
    if other_configs:
        st.markdown("**📋 所有配置**")
        cols = st.columns(min(len(other_configs), 3))
        
        for i, config in enumerate(other_configs):
            with cols[i % len(cols)]:
                provider_info = all_providers.get(config['provider'], {})
                icon = provider_info.get('icon', '🔧')
                status_icon = "🟢" if config['validated'] else "🟡"
                
                button_text = f"{icon} {config['config_name']}"
                
                if st.button(
                    button_text,
                    key=f"quick_other_{config['id']}",
                    use_container_width=True
                ):
                    switch_to_config(config)
                    st.success(f"✅ 已切換到: {config['config_name']}")
                    rerun_app()
                
                st.caption(f"{status_icon} 使用 {config['usage_count']} 次")

def switch_to_config(config: Dict):
    """切換到指定配置"""
    all_providers = provider_manager.get_all_providers()
    provider_info = all_providers.get(config['provider'], {})
    
    # 更新會話狀態
    st.session_state.selected_provider = config['provider']
    st.session_state.api_config = {
        'provider': config['provider'],
        'api_key': config['api_key'],
        'base_url': config['base_url'] or provider_info.get('base_url', ''),
        'validated': config['validated'],
        'key_name': config['key_name'],
        'key_id': config['api_key_id']
    }
    
    # 如果有默認模型，也一併設置
    if config['default_model_id']:
        st.session_state.selected_model = config['default_model_id']
    
    # 更新使用統計
    provider_manager.update_config_usage(config['id'])

def show_quick_switch_manager():
    """顯示快速切換配置管理"""
    st.markdown("### 🔧 快速切換配置管理")
    
    # 創建新配置
    with st.expander("➕ 創建新的快速切換配置"):
        with st.form("new_quick_config"):
            st.markdown("#### 📋 配置信息")
            
            config_name = st.text_input("配置名稱 *", placeholder="例如：工作用 Navy API")
            
            # 選擇供應商
            all_providers = provider_manager.get_all_providers()
            provider_options = list(all_providers.keys())
            selected_provider = st.selectbox(
                "選擇供應商 *",
                provider_options,
                format_func=lambda x: f"{all_providers[x]['icon']} {all_providers[x]['name'] if not all_providers[x].get('is_custom') else all_providers[x]['display_name']}"
            )
            
            # 選擇密鑰
            if selected_provider:
                provider_keys = provider_manager.get_api_keys(selected_provider)
                if provider_keys:
                    key_options = {key['id']: f"{key['key_name']} ({'✅' if key['validated'] else '⚠️'})" for key in provider_keys}
                    selected_key_id = st.selectbox("選擇密鑰 *", list(key_options.keys()), format_func=lambda x: key_options[x])
                else:
                    st.warning(f"⚠️ {selected_provider} 沒有保存的密鑰")
                    selected_key_id = None
            else:
                selected_key_id = None
            
            # 選擇默認模型
            if selected_provider:
                provider_models = provider_manager.get_provider_models(selected_provider)
                if provider_models:
                    model_options = [""] + [model['model_id'] for model in provider_models]
                    default_model = st.selectbox(
                        "默認模型（可選）",
                        model_options,
                        format_func=lambda x: "未選擇" if x == "" else next((m['model_name'] for m in provider_models if m['model_id'] == x), x)
                    )
                else:
                    default_model = ""
            else:
                default_model = ""
            
            notes = st.text_area("備註", placeholder="描述此配置的用途...")
            is_favorite = st.checkbox("設為收藏配置", help="收藏的配置會優先顯示")
            
            if st.form_submit_button("💾 創建配置", type="primary", use_container_width=True):
                if config_name and selected_provider and selected_key_id:
                    config_id = provider_manager.save_quick_switch_config(
                        config_name=config_name,
                        provider=selected_provider,
                        api_key_id=selected_key_id,
                        default_model_id=default_model,
                        notes=notes,
                        is_favorite=is_favorite
                    )
                    
                    if config_id:
                        st.success(f"✅ 快速切換配置 '{config_name}' 已創建！")
                        time.sleep(1)
                        rerun_app()
                    else:
                        st.error("❌ 創建失敗：配置名稱已存在")
                else:
                    st.error("❌ 請填寫所有必填字段")
    
    # 現有配置管理
    st.markdown("#### 📋 現有配置")
    
    quick_configs = provider_manager.get_quick_switch_configs()
    
    if quick_configs:
        for config in quick_configs:
            with st.container():
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
                    all_providers = provider_manager.get_all_providers()
                    provider_info = all_providers.get(config['provider'], {})
                    
                    # 配置標題
                    title_icons = []
                    if config['is_favorite']:
                        title_icons.append("⭐")
                    if config['validated']:
                        title_icons.append("🟢")
                    else:
                        title_icons.append("🟡")
                    
                    icon_text = " ".join(title_icons)
                    st.markdown(f"**{icon_text} {config['config_name']}**")
                    
                    # 詳細信息
                    provider_name = provider_info.get('name', provider_info.get('display_name', config['provider']))
                    st.caption(f"**供應商**: {provider_info.get('icon', '🔧')} {provider_name}")
                    st.caption(f"**密鑰**: {config['key_name']} | **使用次數**: {config['usage_count']}")
                    
                    if config['default_model_id']:
                        st.caption(f"**默認模型**: {config['default_model_id']}")
                    
                    if config['notes']:
                        st.caption(f"**備註**: {config['notes']}")
                    
                    if config['last_used']:
                        st.caption(f"**最後使用**: {config['last_used']}")
                
                with col_actions:
                    # 快速切換按鈕
                    if st.button("🚀 切換", key=f"switch_{config['id']}", use_container_width=True):
                        switch_to_config(config)
                        st.success(f"✅ 已切換到: {config['config_name']}")
                        rerun_app()
                    
                    # 測試按鈕
                    if st.button("🧪 測試", key=f"test_{config['id']}", use_container_width=True):
                        with st.spinner("測試連接..."):
                            is_valid, message = validate_api_key(config['api_key'], config['base_url'], config['provider'])
                            if is_valid:
                                st.success(f"✅ {message}")
                                provider_manager.update_key_validation(config['api_key_id'], True)
                            else:
                                st.error(f"❌ {message}")
                                provider_manager.update_key_validation(config['api_key_id'], False)
                    
                    # 刪除按鈕
                    if st.button("🗑️ 刪除", key=f"delete_{config['id']}", use_container_width=True):
                        if st.session_state.get(f"confirm_delete_{config['id']}", False):
                            provider_manager.delete_quick_switch_config(config['id'])
                            st.success("配置已刪除")
                            rerun_app()
                        else:
                            st.session_state[f"confirm_delete_{config['id']}"] = True
                            st.warning("再次點擊確認刪除")
                
                st.markdown("---")
    else:
        st.info("📭 尚未創建任何快速切換配置")

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

def show_provider_selector():
    """顯示供應商選擇器"""
    st.subheader("🏢 選擇模型供應商")
    
    # 快速切換面板
    show_quick_switch_panel()
    
    st.markdown("---")
    
    # 原有的供應商選擇界面
    all_providers = provider_manager.get_all_providers()
    default_providers = {k: v for k, v in all_providers.items() if not v.get('is_custom', False)}
    custom_providers = {k: v for k, v in all_providers.items() if v.get('is_custom', False)}
    
    # 預設供應商
    if default_providers:
        st.markdown("### 🏭 預設供應商")
        
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
                    
                    if 'features' in provider_info:
                        features_text = " | ".join([f"🏷️ {feature}" for feature in provider_info['features']])
                        st.markdown(f"**特色**: {features_text}")
                    
                    if st.button(f"選擇 {provider_info['name']}", key=f"select_default_{provider_key}", use_container_width=True):
                        st.session_state.selected_provider = provider_key
                        st.success(f"已選擇 {provider_info['name']}")
                        rerun_app()
                    
                    saved_keys = provider_manager.get_api_keys(provider_key)
                    if saved_keys:
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
                    
                    st.caption(f"**類型**: {provider_info['api_type']} | **端點**: {provider_info['base_url'][:30]}...")
                    
                    if provider_info['features']:
                        features_text = " | ".join([f"🏷️ {feature}" for feature in provider_info['features']])
                        st.markdown(f"**功能**: {features_text}")
                    
                    if st.button(f"選擇 {provider_info['display_name']}", key=f"select_custom_{provider_key}", use_container_width=True):
                        st.session_state.selected_provider = provider_key
                        st.success(f"已選擇 {provider_info['display_name']}")
                        rerun_app()
                    
                    saved_keys = provider_manager.get_api_keys(provider_key)
                    if saved_keys:
                        st.caption(f"🔑 已保存 {len(saved_keys)} 個密鑰")
    else:
        st.markdown("### 🔧 自定義供應商")
        st.info("尚未創建任何自定義供應商")

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

# 側邊欄 - 加入快速切換功能
with st.sidebar:
    st.markdown("### ⚡ 快速切換")
    
    # 顯示當前配置
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
    
    # 快速切換配置按鈕（僅顯示收藏的）
    quick_configs = provider_manager.get_quick_switch_configs()
    favorite_configs = [c for c in quick_configs if c['is_favorite']]
    
    if favorite_configs:
        st.markdown("#### 🌟 收藏配置")
        for config in favorite_configs[:3]:  # 最多顯示3個
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
    
    # 管理按鈕
    if st.button("⚡ 管理快速切換", use_container_width=True):
        st.session_state.show_quick_switch_manager = True
        rerun_app()
    
    st.markdown("---")
    
    # 統計信息
    st.markdown("### 📊 統計")
    total_keys = len(provider_manager.get_api_keys())
    total_configs = len(quick_configs)
    custom_providers_count = len(provider_manager.get_custom_providers())
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("密鑰數", total_keys)
        st.metric("快速配置", total_configs)
    with col_stat2:
        st.metric("自定義供應商", custom_providers_count)

# 主標題
st.title("🎨 Flux & SD Generator Pro - 快速切換版")

# 主要內容
if 'show_quick_switch_manager' in st.session_state and st.session_state.show_quick_switch_manager:
    show_quick_switch_manager()
    if st.button("⬅️ 返回", key="back_from_quick_manager"):
        del st.session_state.show_quick_switch_manager
        rerun_app()

elif 'selected_provider' not in st.session_state:
    show_provider_selector()
else:
    # 顯示當前供應商管理界面
    st.markdown("### 🚀 供應商管理界面")
    st.info("📝 這裡可以加入完整的供應商管理功能（密鑰管理、模型發現、圖像生成等）")
    
    if st.button("🔄 重新選擇供應商"):
        del st.session_state.selected_provider
        rerun_app()

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
