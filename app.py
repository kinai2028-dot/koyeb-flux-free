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
    page_title="Flux AI & SD Generator Pro - 完整版",
    page_icon="🎨",
    layout="wide"
)

# 簡化的密鑰管理系統
class SimpleKeyManager:
    def __init__(self):
        self.db_path = "simple_keys.db"
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discovered_models (
                id TEXT PRIMARY KEY,
                model_name TEXT UNIQUE NOT NULL,
                provider TEXT NOT NULL,
                category TEXT,
                description TEXT,
                icon TEXT,
                priority INTEGER DEFAULT 999,
                endpoint_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
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
    
    def save_discovered_model(self, model_name: str, provider: str, category: str = "", 
                             description: str = "", icon: str = "", priority: int = 999,
                             endpoint_path: str = "") -> str:
        """保存自動發現的模型"""
        model_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 檢查是否已存在
        cursor.execute("SELECT id FROM discovered_models WHERE model_name = ?", (model_name,))
        if cursor.fetchone():
            conn.close()
            return None
        
        cursor.execute('''
            INSERT INTO discovered_models 
            (id, model_name, provider, category, description, icon, priority, endpoint_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_id, model_name, provider, category, description, icon, priority, endpoint_path))
        
        conn.commit()
        conn.close()
        return model_id
    
    def get_discovered_models(self, category: str = None) -> List[Dict]:
        """獲取發現的模型列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute('''
                SELECT model_name, provider, category, description, icon, priority, endpoint_path
                FROM discovered_models WHERE category = ?
                ORDER BY priority, model_name
            ''', (category,))
        else:
            cursor.execute('''
                SELECT model_name, provider, category, description, icon, priority, endpoint_path
                FROM discovered_models
                ORDER BY category, priority, model_name
            ''')
        
        models = []
        for row in cursor.fetchall():
            models.append({
                'model_name': row[0], 'provider': row[1], 'category': row[2],
                'description': row[3], 'icon': row[4], 'priority': row[5],
                'endpoint_path': row[6]
            })
        
        conn.close()
        return models
    
    def delete_api_key(self, key_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
        conn.commit()
        conn.close()
    
    def update_key_validation(self, key_id: str, validated: bool):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE api_keys SET validated = ? WHERE id = ?", (validated, key_id))
        conn.commit()
        conn.close()

# 全局實例
key_manager = SimpleKeyManager()

# API 提供商配置
API_PROVIDERS = {
    "Navy": {
        "name": "Navy API",
        "base_url_default": "https://api.navy/v1", 
        "key_prefix": "sk-",
        "description": "Navy AI 圖像生成服務",
        "icon": "⚓"
    },
    "OpenAI Compatible": {
        "name": "OpenAI Compatible API",
        "base_url_default": "https://api.openai.com/v1",
        "key_prefix": "sk-",
        "description": "OpenAI 官方或兼容的 API 服務",
        "icon": "🤖"
    },
    "Hugging Face": {
        "name": "Hugging Face API",
        "base_url_default": "https://api-inference.huggingface.co",
        "key_prefix": "hf_",
        "description": "Hugging Face 推理 API",
        "icon": "🤗"
    },
    "Together AI": {
        "name": "Together AI",
        "base_url_default": "https://api.together.xyz/v1",
        "key_prefix": "",
        "description": "Together AI 平台",
        "icon": "🤝"
    }
}

# 基礎模型庫
BASE_MODELS = {
    "flux-schnell": {
        "name": "FLUX.1 Schnell",
        "description": "最快的生成速度，開源模型",
        "icon": "⚡",
        "category": "flux",
        "test_prompt": "A simple cat sitting on a table",
        "priority": 1
    },
    "flux-dev": {
        "name": "FLUX.1 Dev", 
        "description": "開發版本，平衡速度與質量",
        "icon": "🔧",
        "category": "flux",
        "test_prompt": "A beautiful landscape with mountains",
        "priority": 2
    },
    "flux-pro": {
        "name": "FLUX.1 Pro",
        "description": "專業版本，最佳品質",
        "icon": "👑",
        "category": "flux",
        "test_prompt": "Professional portrait in natural lighting",
        "priority": 3
    },
    "stable-diffusion-xl": {
        "name": "Stable Diffusion XL",
        "description": "SDXL 基礎模型，高品質生成",
        "icon": "🎨",
        "category": "stable-diffusion",
        "test_prompt": "A beautiful sunset over mountains",
        "priority": 10
    },
    "stable-diffusion-2-1": {
        "name": "Stable Diffusion 2.1",
        "description": "SD 2.1 經典模型",
        "icon": "🖼️",
        "category": "stable-diffusion",
        "test_prompt": "Artistic portrait with dramatic lighting",
        "priority": 11
    }
}

# Hugging Face 模型端點
HF_MODEL_ENDPOINTS = {
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
        "prompthero/openjourney",
        "wavymulder/Analog-Diffusion"
    ]
}

def auto_discover_models_from_api(client, provider: str, api_key: str, base_url: str):
    """從 API 自動發現模型"""
    try:
        if provider == "Hugging Face":
            # HF 模型發現
            discovered = []
            for category, endpoints in HF_MODEL_ENDPOINTS.items():
                for endpoint in endpoints:
                    model_name = endpoint.split('/')[-1]
                    
                    # 分析模型信息
                    if "flux" in model_name.lower():
                        icon = "⚡"
                        category = "flux"
                        description = f"Flux model: {model_name}"
                    elif "stable" in model_name.lower() or "sd" in model_name.lower():
                        icon = "🎨"
                        category = "stable-diffusion"
                        description = f"Stable Diffusion model: {model_name}"
                    else:
                        icon = "🤖"
                        category = "other"
                        description = f"AI model: {model_name}"
                    
                    # 保存到數據庫
                    model_id = key_manager.save_discovered_model(
                        model_name=model_name,
                        provider=provider,
                        category=category,
                        description=description,
                        icon=icon,
                        priority=999,
                        endpoint_path=endpoint
                    )
                    
                    if model_id:
                        discovered.append(model_name)
            
            return discovered
        else:
            # OpenAI Compatible API 模型發現
            response = client.models.list()
            discovered = []
            
            for model in response.data:
                model_id = model.id
                
                # 智能分類
                model_lower = model_id.lower()
                if any(keyword in model_lower for keyword in ['flux', 'black-forest']):
                    category = "flux"
                    icon = "⚡"
                    description = f"Flux model: {model_id}"
                elif any(keyword in model_lower for keyword in ['stable', 'diffusion', 'sd']):
                    category = "stable-diffusion"
                    icon = "🎨"
                    description = f"Stable Diffusion model: {model_id}"
                else:
                    category = "other"
                    icon = "🤖"
                    description = f"AI model: {model_id}"
                
                # 保存模型
                saved_id = key_manager.save_discovered_model(
                    model_name=model_id,
                    provider=provider,
                    category=category,
                    description=description,
                    icon=icon,
                    priority=999
                )
                
                if saved_id:
                    discovered.append(model_id)
            
            return discovered
    
    except Exception as e:
        st.error(f"模型發現失敗: {str(e)}")
        return []

def get_all_available_models():
    """獲取所有可用模型（基礎+發現的）"""
    all_models = BASE_MODELS.copy()
    
    # 添加發現的模型
    discovered = key_manager.get_discovered_models()
    for model in discovered:
        model_key = model['model_name']
        all_models[model_key] = {
            "name": model['model_name'],
            "description": model['description'],
            "icon": model['icon'],
            "category": model['category'],
            "test_prompt": "A beautiful detailed image",
            "priority": model['priority'],
            "endpoint_path": model.get('endpoint_path', ''),
            "source": "discovered"
        }
    
    return all_models

def validate_api_key(api_key: str, base_url: str, provider: str) -> Tuple[bool, str]:
    """驗證 API 密鑰是否有效"""
    try:
        if provider == "Hugging Face":
            headers = {"Authorization": f"Bearer {api_key}"}
            test_url = f"{base_url}/models/stabilityai/stable-diffusion-xl-base-1.0"
            response = requests.get(test_url, headers=headers, timeout=10)
            return response.status_code == 200, "HF API 驗證" + ("成功" if response.status_code == 200 else f"失敗 ({response.status_code})")
        else:
            test_client = OpenAI(api_key=api_key, base_url=base_url)
            response = test_client.models.list()
            return True, "API 密鑰驗證成功"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            return False, "API 密鑰無效或已過期"
        elif "403" in error_msg:
            return False, "API 密鑰權限不足"
        elif "404" in error_msg:
            return False, "API 端點不存在"
        elif "timeout" in error_msg.lower():
            return False, "API 連接超時"
        else:
            return False, f"驗證失敗: {error_msg[:50]}"

def generate_images_with_retry(client, provider: str, api_key: str, base_url: str, **params) -> Tuple[bool, any]:
    """帶重試機制的圖像生成"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            if provider == "Hugging Face":
                # HF API 調用
                headers = {"Authorization": f"Bearer {api_key}"}
                data = {"inputs": params.get("prompt", "")}
                
                model_name = params.get("model", "stable-diffusion-xl")
                all_models = get_all_available_models()
                model_info = all_models.get(model_name, {})
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

def display_image_with_actions(image_url: str, image_id: str, history_item: Dict = None):
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
            # 重新生成按鈕
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

def show_key_manager():
    """密鑰管理界面"""
    st.subheader("🔐 API 密鑰管理")
    
    management_mode = st.radio(
        "操作模式:",
        ["💾 保存密鑰", "📋 管理密鑰", "🔍 發現模型"],
        horizontal=True
    )
    
    if management_mode == "💾 保存密鑰":
        col_provider, col_name = st.columns(2)
        
        with col_provider:
            save_provider = st.selectbox(
                "選擇提供商:",
                list(API_PROVIDERS.keys()),
                format_func=lambda x: f"{API_PROVIDERS[x]['icon']} {API_PROVIDERS[x]['name']}"
            )
        
        with col_name:
            key_name = st.text_input("密鑰名稱:", placeholder="例如：主要密鑰")
        
        provider_info = API_PROVIDERS[save_provider]
        new_api_key = st.text_input(
            "API 密鑰:",
            type="password",
            placeholder=f"輸入 {provider_info['name']} 的 API 密鑰..."
        )
        
        save_base_url = st.text_input(
            "API 端點:",
            value=provider_info['base_url_default']
        )
        
        notes = st.text_area("備註:", height=60)
        is_default = st.checkbox("設為默認密鑰")
        
        if st.button("💾 保存密鑰", type="primary"):
            if key_name and new_api_key:
                key_id = key_manager.save_api_key(
                    save_provider, key_name, new_api_key, 
                    save_base_url, notes, is_default
                )
                st.success(f"✅ 密鑰已保存！ID: {key_id[:8]}...")
                rerun_app()
            else:
                st.error("❌ 請填寫完整信息")
    
    elif management_mode == "📋 管理密鑰":
        all_keys = key_manager.get_api_keys()
        if not all_keys:
            st.info("📭 尚未保存任何密鑰")
            return
        
        for key_info in all_keys:
            with st.container():
                st.markdown(f"### {API_PROVIDERS.get(key_info['provider'], {}).get('icon', '🔧')} {key_info['key_name']}")
                
                col_info, col_actions = st.columns([2, 1])
                
                with col_info:
                    st.markdown(f"**提供商**: {key_info['provider']}")
                    st.markdown(f"**狀態**: {'🟢 已驗證' if key_info['validated'] else '🟡 未驗證'}")
                    if key_info['notes']:
                        st.markdown(f"**備註**: {key_info['notes']}")
                
                with col_actions:
                    if st.button("✅ 使用", key=f"use_{key_info['id']}"):
                        st.session_state.api_config = {
                            'provider': key_info['provider'],
                            'api_key': key_info['api_key'],
                            'base_url': key_info['base_url'],
                            'validated': key_info['validated'],
                            'key_name': key_info['key_name']
                        }
                        st.success(f"已載入: {key_info['key_name']}")
                        rerun_app()
                
                st.markdown("---")
    
    else:  # 發現模型
        st.markdown("### 🔍 自動模型發現")
        
        if st.session_state.api_config.get('api_key'):
            config = st.session_state.api_config
            
            if st.button("🔍 開始發現模型", type="primary"):
                with st.spinner("正在發現新模型..."):
                    if config['provider'] == "Hugging Face":
                        client = None
                    else:
                        client = OpenAI(
                            api_key=config['api_key'],
                            base_url=config['base_url']
                        )
                    
                    discovered = auto_discover_models_from_api(
                        client, config['provider'], 
                        config['api_key'], config['base_url']
                    )
                    
                    if discovered:
                        st.success(f"✅ 發現 {len(discovered)} 個新模型！")
                        for model in discovered[:5]:  # 顯示前5個
                            st.write(f"• {model}")
                    else:
                        st.info("ℹ️ 未發現新模型")
        else:
            st.warning("請先配置 API 密鑰")
        
        # 顯示已發現的模型
        st.markdown("### 📋 已發現的模型")
        discovered_models = key_manager.get_discovered_models()
        
        if discovered_models:
            # 按類別分組
            categories = {}
            for model in discovered_models:
                cat = model['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(model)
            
            for category, models in categories.items():
                st.markdown(f"#### {category.title()} ({len(models)} 個模型)")
                for model in models[:3]:  # 每類顯示前3個
                    st.write(f"{model['icon']} {model['name']} - {model['description']}")
        else:
            st.info("尚未發現任何模型")

def show_api_settings():
    """API 設置界面"""
    st.subheader("🔑 API 設置")
    
    # 密鑰管理器開關
    show_manager = st.checkbox("🔐 顯示密鑰管理", value=False)
    if show_manager:
        show_key_manager()
        st.markdown("---")
    
    # 快速設置
    st.markdown("### ⚡ 快速設置")
    
    col_quick1, col_quick2 = st.columns(2)
    
    with col_quick1:
        st.markdown("#### 🚀 快速載入")
        all_keys = key_manager.get_api_keys()
        
        if all_keys:
            grouped_keys = {}
            for key in all_keys:
                provider = key['provider']
                if provider not in grouped_keys:
                    grouped_keys[provider] = []
                grouped_keys[provider].append(key)
            
            if grouped_keys:
                selected_provider = st.selectbox(
                    "提供商:",
                    list(grouped_keys.keys()),
                    format_func=lambda x: f"{API_PROVIDERS.get(x, {}).get('icon', '🔧')} {x}"
                )
                
                provider_keys = grouped_keys[selected_provider]
                key_options = {
                    key['id']: f"{'⭐' if key['is_default'] else ''} {key['key_name']}"
                    for key in provider_keys
                }
                
                selected_key_id = st.selectbox("密鑰:", list(key_options.keys()), format_func=lambda x: key_options[x])
                
                if st.button("⚡ 載入", type="primary"):
                    selected_key = next(k for k in all_keys if k['id'] == selected_key_id)
                    st.session_state.api_config = {
                        'provider': selected_key['provider'],
                        'api_key': selected_key['api_key'],
                        'base_url': selected_key['base_url'],
                        'validated': selected_key['validated'],
                        'key_name': selected_key['key_name']
                    }
                    st.success(f"✅ 已載入: {selected_key['key_name']}")
                    rerun_app()
        else:
            st.info("尚未保存密鑰")
    
    with col_quick2:
        st.markdown("#### 🎯 當前狀態")
        if st.session_state.api_config.get('api_key'):
            config = st.session_state.api_config
            st.success("🟢 API 已配置")
            st.info(f"**提供商**: {config['provider']}")
            if config.get('key_name'):
                st.info(f"**密鑰**: {config['key_name']}")
        else:
            st.error("🔴 API 未配置")

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

def add_to_history(prompt: str, model: str, images: List[str], metadata: Dict):
    """添加到歷史記錄"""
    history_item = {
        "timestamp": datetime.datetime.now(),
        "prompt": prompt,
        "model": model,
        "images": images,
        "metadata": metadata,
        "id": str(uuid.uuid4())
    }
    st.session_state.generation_history.insert(0, history_item)
    
    # 限制記錄數量
    if len(st.session_state.generation_history) > 50:
        st.session_state.generation_history = st.session_state.generation_history[:50]

# 初始化
init_session_state()

# 檢查 API 配置
api_configured = st.session_state.api_config.get('api_key') is not None and st.session_state.api_config.get('api_key') != ''

# 側邊欄
with st.sidebar:
    show_api_settings()
    st.markdown("---")
    
    # 統計信息
    st.markdown("### 📊 統計")
    all_keys = key_manager.get_api_keys()
    discovered_models = key_manager.get_discovered_models()
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("密鑰數", len(all_keys))
        st.metric("歷史", len(st.session_state.generation_history))
    with col_stat2:
        st.metric("模型數", len(discovered_models) + len(BASE_MODELS))
        st.metric("收藏", len(st.session_state.favorite_images))

# 主標題
st.title("🎨 Flux AI & SD Generator Pro - 完整版")

# 主要內容
tab1, tab2, tab3 = st.tabs(["🚀 圖像生成", "📚 歷史記錄", "⭐ 收藏夾"])

with tab1:
    if not api_configured:
        st.warning("⚠️ 請先配置 API 密鑰")
        st.info("💡 在側邊欄中配置您的 API 密鑰以開始生成圖像")
    else:
        st.success("✅ API 已配置，開始生成圖像吧！")
        
        col_gen1, col_gen2 = st.columns([2, 1])
        
        with col_gen1:
            # 模型選擇
            all_models = get_all_available_models()
            
            # 按類別分組顯示
            flux_models = {k: v for k, v in all_models.items() if v.get('category') == 'flux'}
            sd_models = {k: v for k, v in all_models.items() if v.get('category') == 'stable-diffusion'}
            
            model_category = st.selectbox(
                "選擇模型類別:",
                ["⚡ Flux 模型", "🎨 Stable Diffusion"],
                index=0
            )
            
            if model_category == "⚡ Flux 模型":
                available_models = flux_models
            else:
                available_models = sd_models
            
            if not available_models:
                st.warning("此類別暫無可用模型，請嘗試模型發現功能")
                selected_model = None
            else:
                model_options = list(available_models.keys())
                selected_model = st.selectbox(
                    "選擇具體模型:",
                    model_options,
                    format_func=lambda x: f"{available_models[x]['icon']} {available_models[x]['name']}"
                )
            
            # 提示詞輸入
            st.markdown("### ✏️ 提示詞")
            
            # 檢查是否有重新生成請求
            default_prompt = ""
            if hasattr(st.session_state, 'regenerate_prompt'):
                default_prompt = st.session_state.regenerate_prompt
                delattr(st.session_state, 'regenerate_prompt')
            
            prompt = st.text_area(
                "描述您想要生成的圖像:",
                value=default_prompt,
                height=100,
                placeholder="例如：A majestic dragon flying over ancient mountains during sunset, highly detailed, fantasy art style"
            )
            
            # 快速提示詞模板
            st.markdown("#### 💡 快速模板")
            template_categories = {
                "人物": ["Professional portrait in natural lighting", "Young artist in creative studio", "Elderly person with wise expression"],
                "風景": ["Sunset over snow-capped mountains", "Tropical beach with crystal clear water", "Autumn forest with golden leaves"],
                "藝術": ["Abstract geometric composition", "Watercolor painting style", "Digital art with vibrant colors"],
                "科幻": ["Futuristic cityscape with flying vehicles", "Space station orbiting planet", "Cyberpunk street scene with neon"]
            }
            
            template_cols = st.columns(len(template_categories))
            for i, (category, templates) in enumerate(template_categories.items()):
                with template_cols[i]:
                    st.markdown(f"**{category}**")
                    for j, template in enumerate(templates):
                        if st.button(template[:20] + "...", key=f"template_{i}_{j}", help=template):
                            st.session_state.quick_prompt = template
                            rerun_app()
            
            # 應用快速提示詞
            if hasattr(st.session_state, 'quick_prompt'):
                prompt = st.session_state.quick_prompt
                delattr(st.session_state, 'quick_prompt')
                rerun_app()
            
            # 生成設置
            st.markdown("### ⚙️ 生成設置")
            col_size, col_num = st.columns(2)
            
            with col_size:
                size_options = ["512x512", "768x768", "1024x1024", "1152x896", "896x1152"]
                selected_size = st.selectbox("圖像尺寸:", size_options, index=2)
            
            with col_num:
                num_images = st.slider("生成數量:", 1, 4, 1)
            
            # 生成按鈕
            can_generate = selected_model and prompt.strip() and api_configured
            
            if st.button("🚀 生成圖像", type="primary", disabled=not can_generate, use_container_width=True):
                if can_generate:
                    config = st.session_state.api_config
                    
                    # 初始化客戶端
                    if config['provider'] == "Hugging Face":
                        client = None
                    else:
                        try:
                            client = OpenAI(
                                api_key=config['api_key'],
                                base_url=config['base_url']
                            )
                        except Exception as e:
                            st.error(f"API 客戶端初始化失敗: {str(e)}")
                            client = None
                    
                    if config['provider'] == "Hugging Face" or client:
                        with st.spinner(f"🎨 正在使用 {available_models[selected_model]['name']} 生成圖像..."):
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
                            
                            if success:
                                response = result
                                image_urls = [img.url for img in response.data]
                                
                                metadata = {
                                    "model_name": available_models[selected_model]['name'],
                                    "size": selected_size,
                                    "num_images": num_images,
                                    "category": available_models[selected_model]['category'],
                                    "provider": config['provider']
                                }
                                
                                add_to_history(prompt, selected_model, image_urls, metadata)
                                st.success(f"✨ 成功生成 {len(response.data)} 張圖像！")
                                
                                # 顯示生成的圖像
                                if len(response.data) == 1:
                                    st.markdown("#### 🎨 生成結果")
                                    image_id = f"{st.session_state.generation_history[0]['id']}_0"
                                    display_image_with_actions(
                                        response.data[0].url, image_id, 
                                        st.session_state.generation_history[0]
                                    )
                                else:
                                    st.markdown("#### 🎨 生成結果")
                                    img_cols = st.columns(min(len(response.data), 2))
                                    for i, image_data in enumerate(response.data):
                                        with img_cols[i % len(img_cols)]:
                                            st.markdown(f"**圖像 {i+1}**")
                                            image_id = f"{st.session_state.generation_history[0]['id']}_{i}"
                                            display_image_with_actions(
                                                image_data.url, image_id,
                                                st.session_state.generation_history[0]
                                            )
                            else:
                                st.error(f"❌ 生成失敗: {result}")
                    else:
                        st.error("❌ API 客戶端初始化失敗")
                else:
                    if not selected_model:
                        st.warning("⚠️ 請選擇模型")
                    elif not prompt.strip():
                        st.warning("⚠️ 請輸入提示詞")
        
        with col_gen2:
            st.markdown("### ℹ️ 信息面板")
            
            if selected_model and available_models:
                model_info = available_models[selected_model]
                st.info(f"**當前模型**: {model_info['name']}")
                st.info(f"**類別**: {model_info['category']}")
                st.info(f"**描述**: {model_info['description']}")
            
            st.markdown("### 📊 使用統計")
            st.metric("已生成", len(st.session_state.generation_history))
            st.metric("已收藏", len(st.session_state.favorite_images))
            
            st.markdown("### 🎯 模型發現")
            discovered_count = len(key_manager.get_discovered_models())
            st.metric("發現模型", discovered_count)
            
            if st.button("🔍 發現新模型", use_container_width=True):
                if api_configured:
                    # 觸發模型發現
                    st.info("請在側邊欄的密鑰管理中使用模型發現功能")
                else:
                    st.warning("請先配置 API")

with tab2:
    st.subheader("📚 生成歷史")
    
    if st.session_state.generation_history:
        search_term = st.text_input("🔍 搜索歷史:", placeholder="輸入關鍵詞搜索...")
        
        filtered_history = st.session_state.generation_history
        if search_term:
            filtered_history = [
                item for item in st.session_state.generation_history
                if search_term.lower() in item['prompt'].lower()
            ]
        
        st.info(f"顯示 {len(filtered_history)} / {len(st.session_state.generation_history)} 條記錄")
        
        for item in filtered_history:
            with st.container():
                st.markdown(f"### 🎨 {item['prompt'][:50]}...")
                st.caption(f"時間: {item['timestamp'].strftime('%Y-%m-%d %H:%M')} | 模型: {item.get('metadata', {}).get('model_name', item['model'])}")
                
                if item['images']:
                    img_cols = st.columns(min(len(item['images']), 3))
                    for i, img_url in enumerate(item['images']):
                        with img_cols[i % len(img_cols)]:
                            display_image_with_actions(img_url, f"history_{item['id']}_{i}", item)
                
                st.markdown("---")
    else:
        st.info("📭 尚無生成歷史")

with tab3:
    st.subheader("⭐ 我的收藏")
    
    if st.session_state.favorite_images:
        st.info(f"共收藏 {len(st.session_state.favorite_images)} 張圖像")
        
        fav_cols = st.columns(3)
        for i, fav in enumerate(st.session_state.favorite_images):
            with fav_cols[i % 3]:
                display_image_with_actions(fav['image_url'], fav['id'], fav.get('history_item'))
                st.caption(f"收藏於: {fav['timestamp'].strftime('%m-%d %H:%M')}")
    else:
        st.info("⭐ 尚無收藏圖像")

# 頁腳
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🚀 <strong>Koyeb 部署</strong> | 
    🔍 <strong>自動模型發現</strong> | 
    💾 <strong>密鑰管理</strong> | 
    ⚡ <strong>圖像生成</strong>
    <br><br>
    <small>支援 Flux AI & Stable Diffusion，自動發現模型，完整歷史管理</small>
</div>
""", unsafe_allow_html=True)
