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

# 側邊欄
with st.sidebar:
    st.markdown("### 🎭 FLUX Krea 快速啟動")
    
    # 推薦 FLUX Krea 供應商
    krea_providers = ["Krea.ai", "Pollinations.ai", "Hugging Face"]
    available_krea = [p for p in krea_providers if p in MODEL_PROVIDERS]
    
    if available_krea:
        selected_krea = st.selectbox(
            "選擇 FLUX Krea 供應商:",
            [""] + available_krea,
            format_func=lambda x: "請選擇..." if x == "" else f"{MODEL_PROVIDERS[x]['icon']} {MODEL_PROVIDERS[x]['name']}"
        )
        
        if selected_krea and st.button("🚀 快速啟動 FLUX Krea", use_container_width=True, type="primary"):
            provider_info = MODEL_PROVIDERS[selected_krea]
            st.session_state.selected_provider = selected_krea
            
            # 如果不需要密鑰，直接配置
            if not provider_info.get('requires_api_key', True):
                st.session_state.api_config = {
                    'provider': selected_krea,
                    'api_key': 'no-key-required',
                    'base_url': provider_info['base_url'],
                    'validated': True,
                    'key_name': f'{provider_info["name"]} 免費服務'
                }
            
            st.success(f"🎭 {provider_info['name']} FLUX Krea 已啟動！")
            rerun_app()
    
    st.markdown("---")
    
    # 顯示當前狀態
    st.markdown("### ⚡ 當前狀態")
    
    api_configured = st.session_state.api_config.get('api_key') is not None and st.session_state.api_config.get('api_key') != ''
    
    if 'selected_provider' in st.session_state and api_configured:
        provider = st.session_state.selected_provider
        all_providers = provider_manager.get_all_providers()
        provider_info = all_providers.get(provider, {})
        
        current_name = f"{provider_info['icon']} {provider_info['name']}"
        st.success(f"✅ {current_name}")
        
        # 特別標注 FLUX Krea 支援
        if "flux-krea" in provider_info.get('features', []):
            st.info("🎭 支援 FLUX Krea")
        
        if st.session_state.api_config.get('key_name'):
            st.caption(f"🔑 {st.session_state.api_config['key_name']}")
    else:
        st.info("未配置 API")
    
    st.markdown("---")
    
    # 統計信息
    st.markdown("### 📊 統計")
    total_keys = len(provider_manager.get_api_keys())
    quick_configs = provider_manager.get_quick_switch_configs()
    total_configs = len(quick_configs)
    flux_krea_models = provider_manager.get_provider_models(category="flux-krea")
    total_krea_models = len(flux_krea_models)
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("密鑰數", total_keys)
        st.metric("FLUX Krea", total_krea_models)
    with col_stat2:
        st.metric("快速配置", total_configs)

# 主標題
st.title("🎨 Flux & SD Generator Pro - 完整版 + FLUX Krea")

# FLUX Krea 功能介紹
if 'selected_provider' not in st.session_state:
    st.markdown("### 🎭 什麼是 FLUX Krea？")
    
    col_intro1, col_intro2 = st.columns(2)
    
    with col_intro1:
        st.info("""
        **🎯 美學優化**
        
        FLUX Krea 是專門針對美學進行優化的 "Opinionated" 模型，致力於產生更真實、多樣化的圖像，避免過度飽和的紋理和典型的 "AI 外觀"。
        """)
        
        st.success("""
        **🌟 核心特色**
        
        • 寫實且多樣化的圖像輸出
        • 避免過度飽和的 AI 外觀  
        • 優秀的人類偏好評估表現
        • 與 FLUX.1 生態系統兼容
        """)
    
    with col_intro2:
        st.warning("""
        **🎨 適用場景**
        
        • 商業攝影和廣告
        • 藝術創作和概念設計
        • 電商產品圖像
        • 社交媒體內容
        """)
        
        st.info("""
        **⚡ 支援平台**
        
        • Krea.ai - 官方平台
        • Pollinations.ai - 完全免費
        • Hugging Face - 開源社區
        • Together AI - 高性能 API
        """)

# 主要內容
if 'selected_provider' not in st.session_state:
    show_provider_selector()
else:
    # 顯示供應商管理界面
    selected_provider = st.session_state.selected_provider
    all_providers = provider_manager.get_all_providers()
    provider_info = all_providers[selected_provider]
    
    st.subheader(f"{provider_info['icon']} {provider_info['name']}")
    
    # 特別標注 FLUX Krea 支援
    if "flux-krea" in provider_info.get('features', []):
        st.success("🎭 此供應商支援 FLUX Krea 美學優化模型！")
    
    col_info, col_switch = st.columns([3, 1])
    
    with col_info:
        st.info(f"📋 {provider_info['description']}")
        st.caption(f"🔗 API 類型: {provider_info['api_type']} | 端點: {provider_info['base_url']}")
        
        features_badges = " ".join([f"`{feature}`" for feature in provider_info['features']])
        st.markdown(f"**支持功能**: {features_badges}")
        
        # 特殊功能標注
        if provider_info.get('speciality'):
            st.success(f"🎯 專長: {provider_info['speciality']}")
    
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
        st.markdown("### 🎨 圖像生成")
        st.info("🚀 完整的圖像生成界面開發中，包含 FLUX Krea 特殊參數調節...")
        
        # 顯示可用模型
        discovered_models = provider_manager.get_provider_models(selected_provider)
        if discovered_models:
            st.markdown("#### 🤖 可用模型")
            for model in discovered_models:
                icon = "🎭" if model['category'] == 'flux-krea' else model['icon']
                st.write(f"{icon} **{model['model_name']}** - {model['description']}")
    
    with management_tabs[3]:
        st.markdown("### 📊 性能監控")
        st.info("📊 性能監控功能開發中，包含 FLUX Krea 美學評分...")
        
        # 顯示基本統計
        saved_keys = provider_manager.get_api_keys(selected_provider)
        discovered_models = provider_manager.get_provider_models(selected_provider)
        
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        
        with col_perf1:
            st.metric("🔑 密鑰數量", len(saved_keys))
        
        with col_perf2:
            flux_krea_models = [m for m in discovered_models if m['category'] == 'flux-krea']
            st.metric("🎭 FLUX Krea 模型", len(flux_krea_models))
        
        with col_perf3:
            all_models = len(discovered_models)
            st.metric("🤖 總模型數", all_models)

# 頁腳
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🎭 <strong>FLUX Krea 美學優化</strong> | 
    🌸 <strong>免費服務</strong> | 
    ⚡ <strong>快速切換</strong> | 
    📊 <strong>智能管理</strong>
    <br><br>
    <small>現已全面支援 FLUX Krea 美學優化模型，打造真正專業級的 AI 圖像生成體驗！</small>
</div>
""", unsafe_allow_html=True)
