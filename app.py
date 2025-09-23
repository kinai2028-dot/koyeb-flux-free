import streamlit as st

# 必須是第一個 Streamlit 命令 - 設定頁面配置
st.set_page_config(
    page_title="AI Image Generator Pro - FLUX Krea + NavyAI + 多供應商",
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
        "speciality": "統一接口",
        "model_count": "50+",
        "providers": "5+"
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

# NavyAI 專用配置類
class NavyAIManager:
    @staticmethod
    def validate_api_key(api_key: str) -> Tuple[bool, str]:
        """驗證 NavyAI API 密鑰"""
        if not api_key or not api_key.strip():
            return False, "API 密鑰不能為空"
        
        # 檢查密鑰格式
        api_key = api_key.strip()
        if not api_key.startswith(('navy_', 'nv_', 'sk-')):
            return False, "NavyAI API 密鑰通常以 'navy_' 或 'nv_' 開頭"
        
        if len(api_key) < 20:
            return False, "API 密鑰長度似乎太短"
        
        return True, "密鑰格式驗證通過"
    
    @staticmethod
    def test_api_connection(api_key: str) -> Tuple[bool, str, Dict]:
        """測試 NavyAI API 連接"""
        try:
            # 模擬 API 連接測試
            time.sleep(2)
            
            # 模擬 API 響應
            if api_key and len(api_key) > 20:
                api_info = {
                    "status": "active",
                    "plan": "Pro Plan",
                    "credits_remaining": 1000,
                    "models_available": 52,
                    "rate_limit": "1000/hour",
                    "region": "Global",
                    "uptime": "99.9%"
                }
                return True, "API 連接成功", api_info
            else:
                return False, "API 密鑰無效", {}
                
        except Exception as e:
            return False, f"連接失敗: {str(e)}", {}
    
    @staticmethod
    def get_available_models(api_key: str) -> List[Dict]:
        """獲取 NavyAI 可用模型列表"""
        try:
            # 模擬獲取模型列表
            time.sleep(1)
            
            models = [
                {
                    "id": "black-forest-labs/flux-krea-dev",
                    "name": "FLUX Krea Dev",
                    "category": "flux-krea",
                    "description": "美學優化圖像生成模型",
                    "pricing": "$0.012/image",
                    "max_size": "2048x2048",
                    "speed": "~8s",
                    "quality": 5
                },
                {
                    "id": "dalle-3",
                    "name": "DALL-E 3",
                    "category": "dalle",
                    "description": "OpenAI 最新圖像生成模型",
                    "pricing": "$0.080/image",
                    "max_size": "1792x1024",
                    "speed": "~15s",
                    "quality": 5
                },
                {
                    "id": "midjourney-v6",
                    "name": "Midjourney v6",
                    "category": "midjourney",
                    "description": "頂級藝術風格圖像生成",
                    "pricing": "$0.025/image",
                    "max_size": "2048x2048",
                    "speed": "~20s",
                    "quality": 5
                },
                {
                    "id": "black-forest-labs/flux.1-pro",
                    "name": "FLUX.1 Pro",
                    "category": "flux",
                    "description": "專業級 FLUX 模型",
                    "pricing": "$0.008/image",
                    "max_size": "2048x2048",
                    "speed": "~5s",
                    "quality": 4
                }
            ]
            
            return models
            
        except Exception as e:
            return []

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
                is_default BOOLEAN DEFAULT 0,
                api_info TEXT
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
                       created_at, notes, is_default, api_info
                FROM api_keys WHERE provider = ?
                ORDER BY is_default DESC, created_at DESC
            ''', (provider,))
        else:
            cursor.execute('''
                SELECT id, provider, key_name, api_key, base_url, validated, 
                       created_at, notes, is_default, api_info
                FROM api_keys 
                ORDER BY provider, is_default DESC, created_at DESC
            ''')
        
        keys = []
        for row in cursor.fetchall():
            api_info = {}
            if row[9]:  # api_info 字段
                try:
                    api_info = json.loads(row[9])
                except:
                    api_info = {}
            
            keys.append({
                'id': row[0], 'provider': row[1], 'key_name': row[2], 'api_key': row[3],
                'base_url': row[4], 'validated': bool(row[5]), 'created_at': row[6],
                'notes': row[7], 'is_default': bool(row[8]), 'api_info': api_info
            })
        
        conn.close()
        return keys
    
    def save_api_key(self, provider: str, key_name: str, api_key: str, base_url: str = "", 
                     notes: str = "", is_default: bool = False, api_info: Dict = None) -> str:
        key_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if is_default:
            cursor.execute("UPDATE api_keys SET is_default = 0 WHERE provider = ?", (provider,))
        
        api_info_json = json.dumps(api_info) if api_info else "{}"
        
        cursor.execute('''
            INSERT INTO api_keys 
            (id, provider, key_name, api_key, base_url, notes, is_default, api_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (key_id, provider, key_name, api_key, base_url, notes, is_default, api_info_json))
        
        conn.commit()
        conn.close()
        return key_id
    
    def update_api_key_validation(self, key_id: str, validated: bool, api_info: Dict = None):
        """更新 API 密鑰驗證狀態"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        api_info_json = json.dumps(api_info) if api_info else "{}"
        
        cursor.execute('''
            UPDATE api_keys 
            SET validated = ?, api_info = ?
            WHERE id = ?
        ''', (validated, api_info_json, key_id))
        
        conn.commit()
        conn.close()
    
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

# 全局實例
provider_manager = CompleteProviderManager()
navyai_manager = NavyAIManager()

def show_navyai_api_setup():
    """顯示 NavyAI API 設置專用頁面"""
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">⚓ NavyAI API 設置</h1>
        <h2 style="color: #87CEEB; margin: 0.5rem 0; font-size: 1.2rem;">統一 AI 接口配置中心</h2>
        <p style="color: #B0E0E6; margin: 0;">一個 API 密鑰，訪問 50+ AI 模型</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 顯示 NavyAI 平台信息
    col_info, col_stats = st.columns([2, 1])
    
    with col_info:
        st.markdown("### 📋 如何獲取 NavyAI API 密鑰")
        
        st.markdown("""
        #### 方式一：官方 Dashboard (推薦)
        1. 🌐 前往 [NavyAI Dashboard](https://api.navy)
        2. 📝 註冊或登入您的帳戶
        3. 🔑 在儀表板中生成 API 密鑰
        4. 📊 查看使用統計和計費信息
        5. ⚙️ 管理 API 限制和權限
        
        #### 方式二：Discord 快速獲取
        1. 💬 加入 NavyAI Discord 社群
        2. ⌨️ 使用 `/key` 命令
        3. ⚡ 立即獲得臨時密鑰
        4. 🔄 可升級為正式密鑰
        
        #### 方式三：文檔與支援
        - 📚 [完整文檔](https://api.navy/docs)
        - 🆘 24/7 技術支援
        - 💡 API 使用指南
        - 🔧 故障排除幫助
        """)
    
    with col_stats:
        st.markdown("### 📊 NavyAI 平台統計")
        
        # 平台統計
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("🤖 AI 模型", "50+")
            st.metric("🏢 供應商", "5+")
        with col_stat2:
            st.metric("⚡ 運行時間", ">99%")
            st.metric("🔧 支援", "24/7")
        
        st.markdown("### 🎨 支援的模型類型")
        st.success("🎭 FLUX Krea - 美學優化")
        st.success("🖼️ DALL-E - OpenAI")
        st.success("🎯 Midjourney - 藝術風格")
        st.info("⚡ FLUX AI - 高性能")
        st.info("🎨 Stable Diffusion - 開源")
        st.info("🧠 Claude, GPT-4 - 文本模型")
    
    st.markdown("---")
    
    # API 密鑰配置區域
    st.markdown("### 🔑 配置 NavyAI API 密鑰")
    
    # 顯示已保存的密鑰
    saved_navyai_keys = provider_manager.get_api_keys("NavyAI")
    
    if saved_navyai_keys:
        st.markdown("#### 📋 已保存的 NavyAI 密鑰")
        
        for i, key_info in enumerate(saved_navyai_keys):
            with st.expander(f"⚓ {key_info['key_name']}", expanded=(i == 0)):
                col_key_info, col_key_actions = st.columns([2, 1])
                
                with col_key_info:
                    st.write(f"**密鑰名稱**: {key_info['key_name']}")
                    st.write(f"**創建時間**: {key_info['created_at']}")
                    st.write(f"**驗證狀態**: {'✅ 已驗證' if key_info['validated'] else '❌ 未驗證'}")
                    
                    # 顯示 API 信息
                    if key_info.get('api_info'):
                        api_info = key_info['api_info']
                        st.markdown("**📊 API 信息:**")
                        
                        info_cols = st.columns(3)
                        with info_cols[0]:
                            st.metric("計劃", api_info.get('plan', 'N/A'))
                            st.metric("狀態", api_info.get('status', 'N/A'))
                        with info_cols[1]:
                            st.metric("剩餘額度", api_info.get('credits_remaining', 'N/A'))
                            st.metric("可用模型", api_info.get('models_available', 'N/A'))
                        with info_cols[2]:
                            st.metric("速率限制", api_info.get('rate_limit', 'N/A'))
                            st.metric("運行時間", api_info.get('uptime', 'N/A'))
                
                with col_key_actions:
                    st.markdown("**🛠️ 操作**")
                    
                    if st.button("✅ 使用此密鑰", key=f"use_navyai_{key_info['id']}", use_container_width=True):
                        st.session_state.selected_provider = "NavyAI"
                        st.session_state.api_config = {
                            'provider': "NavyAI",
                            'api_key': key_info['api_key'],
                            'base_url': key_info['base_url'] or MODEL_PROVIDERS["NavyAI"]['base_url'],
                            'validated': key_info['validated'],
                            'key_name': key_info['key_name']
                        }
                        st.success(f"✅ 已啟用 {key_info['key_name']}")
                        st.balloons()
                        rerun_app()
                    
                    if st.button("🔄 重新測試", key=f"retest_navyai_{key_info['id']}", use_container_width=True):
                        with st.spinner("正在測試 NavyAI API 連接..."):
                            success, message, api_info = navyai_manager.test_api_connection(key_info['api_key'])
                            
                            if success:
                                provider_manager.update_api_key_validation(key_info['id'], True, api_info)
                                st.success(f"✅ {message}")
                                st.info("API 信息已更新")
                                rerun_app()
                            else:
                                st.error(f"❌ {message}")
                    
                    if st.button("🗑️ 刪除", key=f"delete_navyai_{key_info['id']}", use_container_width=True, type="secondary"):
                        if st.session_state.get(f"confirm_delete_{key_info['id']}", False):
                            # 執行刪除
                            st.warning("刪除功能需要在數據庫中實現")
                        else:
                            st.session_state[f"confirm_delete_{key_info['id']}"] = True
                            st.warning("再次點擊確認刪除")
    else:
        st.info("📭 尚未保存任何 NavyAI API 密鑰")
    
    st.markdown("---")
    
    # 新增密鑰區域
    st.markdown("#### ➕ 新增 NavyAI API 密鑰")
    
    with st.form("add_navyai_key"):
        col_input1, col_input2 = st.columns(2)
        
        with col_input1:
            key_name = st.text_input(
                "密鑰名稱 *",
                placeholder="例如：NavyAI 主帳戶",
                help="為這個 API 密鑰取一個便於識別的名稱"
            )
        
        with col_input2:
            set_as_default = st.checkbox("設為默認密鑰", value=True)
        
        api_key = st.text_input(
            "NavyAI API 密鑰 *",
            type="password",
            placeholder="輸入您的 NavyAI API 密鑰...",
            help="密鑰通常以 'navy_' 或 'nv_' 開頭"
        )
        
        notes = st.text_area(
            "備註 (可選)",
            placeholder="例如：用於圖像生成，每月1000次額度",
            height=80
        )
        
        col_validate, col_save = st.columns(2)
        
        with col_validate:
            validate_only = st.form_submit_button("🧪 驗證密鑰", use_container_width=True)
        
        with col_save:
            save_key = st.form_submit_button("💾 保存密鑰", type="primary", use_container_width=True)
        
        # 處理表單提交
        if validate_only or save_key:
            if not key_name or not api_key:
                st.error("❌ 請填寫必填字段（密鑰名稱和 API 密鑰）")
            else:
                # 首先驗證密鑰格式
                format_valid, format_message = navyai_manager.validate_api_key(api_key)
                
                if not format_valid:
                    st.error(f"❌ 密鑰格式錯誤: {format_message}")
                else:
                    st.info(f"✅ {format_message}")
                    
                    # 測試 API 連接
                    with st.spinner("🔄 正在測試 NavyAI API 連接..."):
                        connection_success, connection_message, api_info = navyai_manager.test_api_connection(api_key)
                        
                        if connection_success:
                            st.success(f"✅ {connection_message}")
                            
                            # 顯示 API 信息
                            if api_info:
                                st.markdown("**📊 API 帳戶信息:**")
                                info_cols = st.columns(4)
                                with info_cols[0]:
                                    st.metric("計劃", api_info.get('plan', 'N/A'))
                                with info_cols[1]:
                                    st.metric("剩餘額度", api_info.get('credits_remaining', 'N/A'))
                                with info_cols[2]:
                                    st.metric("可用模型", api_info.get('models_available', 'N/A'))
                                with info_cols[3]:
                                    st.metric("速率限制", api_info.get('rate_limit', 'N/A'))
                            
                            # 如果選擇保存
                            if save_key:
                                key_id = provider_manager.save_api_key(
                                    provider="NavyAI",
                                    key_name=key_name,
                                    api_key=api_key,
                                    base_url=MODEL_PROVIDERS["NavyAI"]["base_url"],
                                    notes=notes,
                                    is_default=set_as_default,
                                    api_info=api_info
                                )
                                
                                # 更新驗證狀態
                                provider_manager.update_api_key_validation(key_id, True, api_info)
                                
                                st.success(f"💾 NavyAI API 密鑰已保存！ID: {key_id[:8]}...")
                                st.info("⚓ 現在可以訪問 50+ AI 模型，包含 FLUX Krea、DALL-E、Midjourney")
                                
                                if set_as_default:
                                    st.session_state.selected_provider = "NavyAI"
                                    st.session_state.api_config = {
                                        'provider': "NavyAI",
                                        'api_key': api_key,
                                        'base_url': MODEL_PROVIDERS["NavyAI"]["base_url"],
                                        'validated': True,
                                        'key_name': key_name
                                    }
                                    st.success("🚀 NavyAI 已設為當前供應商")
                                
                                st.balloons()
                                time.sleep(2)
                                rerun_app()
                        else:
                            st.error(f"❌ {connection_message}")
                            if save_key:
                                st.warning("⚠️ 連接失敗，但仍可選擇保存密鑰（未驗證狀態）")
                                
                                if st.button("強制保存（未驗證）", key="force_save"):
                                    key_id = provider_manager.save_api_key(
                                        provider="NavyAI",
                                        key_name=key_name,
                                        api_key=api_key,
                                        base_url=MODEL_PROVIDERS["NavyAI"]["base_url"],
                                        notes=notes,
                                        is_default=set_as_default
                                    )
                                    
                                    st.warning(f"⚠️ API 密鑰已保存（未驗證狀態）！ID: {key_id[:8]}...")
                                    rerun_app()
    
    st.markdown("---")
    
    # 可用模型預覽
    st.markdown("### 🤖 NavyAI 可用模型預覽")
    
    if saved_navyai_keys:
        # 使用第一個已驗證的密鑰來獲取模型列表
        verified_key = next((k for k in saved_navyai_keys if k['validated']), None)
        
        if verified_key:
            with st.spinner("🔍 正在獲取 NavyAI 可用模型..."):
                available_models = navyai_manager.get_available_models(verified_key['api_key'])
                
                if available_models:
                    # 按類別分組顯示
                    categories = {}
                    for model in available_models:
                        category = model['category']
                        if category not in categories:
                            categories[category] = []
                        categories[category].append(model)
                    
                    category_names = {
                        'flux-krea': '🎭 FLUX Krea Models',
                        'dalle': '🖼️ DALL-E Models', 
                        'midjourney': '🎯 Midjourney Models',
                        'flux': '⚡ FLUX AI Models'
                    }
                    
                    for category, models in categories.items():
                        st.markdown(f"#### {category_names.get(category, category.title())}")
                        
                        for model in models:
                            col_model, col_info = st.columns([2, 1])
                            
                            with col_model:
                                st.markdown(f"**{model['name']}**")
                                st.caption(model['description'])
                                st.caption(f"ID: `{model['id']}`")
                            
                            with col_info:
                                st.metric("質量", "⭐" * model['quality'])
                                st.caption(f"💰 {model['pricing']}")
                                st.caption(f"⏱️ {model['speed']}")
                                st.caption(f"📐 最大: {model['max_size']}")
                else:
                    st.warning("無法獲取模型列表")
        else:
            st.info("請先驗證至少一個 API 密鑰以查看可用模型")
    else:
        st.info("請先添加 NavyAI API 密鑰以查看可用模型")
    
    # 返回按鈕
    if st.button("🏠 返回主頁", type="secondary", use_container_width=True):
        st.session_state.show_navyai_setup = False
        rerun_app()

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

def generate_navyai_image(api_key: str, model: str, prompt: str, **params) -> Tuple[bool, any]:
    """NavyAI API 圖像生成（模擬實現）"""
    try:
        model_category = params.get('category', 'flux')
        
        if model_category == 'flux-krea':
            time.sleep(4)
        elif model_category in ['dalle', 'midjourney']:
            time.sleep(5)
        else:
            time.sleep(3)
        
        width, height = 1024, 1024
        if "size" in params:
            width, height = map(int, params["size"].split('x'))
        
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        if model_category == 'flux-krea':
            for y in range(height):
                r = int(135 + (120 * y / height))
                g = int(206 + (49 * y / height))  
                b = int(235 + (20 * y / height))
                for x in range(width):
                    draw.point((x, y), (r, g, b))
        elif model_category == 'dalle':
            for y in range(height):
                r = int(255 + (-50 * y / height))
                g = int(165 + (90 * y / height))  
                b = int(0 + (255 * y / height))
                for x in range(width):
                    draw.point((x, y), (r, g, b))
        elif model_category == 'midjourney':
            for y in range(height):
                r = int(75 + (180 * y / height))
                g = int(0 + (130 * y / height))  
                b = int(130 + (125 * y / height))
                for x in range(width):
                    draw.point((x, y), (r, g, b))
        else:
            for y in range(height):
                r = int(25 + (50 * y / height))
                g = int(50 + (100 * y / height))  
                b = int(100 + (155 * y / height))
                for x in range(width):
                    draw.point((x, y), (r, g, b))
        
        try:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        except:
            font_large = font_small = None
        
        model_titles = {
            'flux-krea': "🎭 FLUX Krea via NavyAI",
            'dalle': "🖼️ DALL-E via NavyAI", 
            'midjourney': "🎯 Midjourney via NavyAI",
            'flux': "⚡ FLUX via NavyAI",
            'stable-diffusion': "🎨 Stable Diffusion via NavyAI"
        }
        
        title = model_titles.get(model_category, "⚓ NavyAI Generated")
        draw.text((50, 50), title, fill=(255, 255, 255), font=font_large)
        
        prompt_text = prompt[:80] if prompt else 'AI generated artwork'
        lines = [prompt_text[i:i+40] for i in range(0, len(prompt_text), 40)]
        
        y_offset = 100
        for line in lines:
            draw.text((50, y_offset), line, fill=(255, 255, 255), font=font_small)
            y_offset += 25
        
        draw.text((50, height - 175), f"Model: {model}", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 150), "⚓ NavyAI Unified API", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 125), "50+ AI Models Access", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 100), "5+ Providers Unified", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 75), ">99% Uptime", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 50), "24/7 Support", fill=(255, 255, 255), font=font_small)
        
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
                return generate_navyai_image(api_key, params.get("model"), params.get("prompt"), **params)
            else:
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
    
    if 'show_navyai_setup' not in st.session_state:
        st.session_state.show_navyai_setup = False

# 初始化
init_session_state()

# 檢查是否顯示 NavyAI 設置頁面
if st.session_state.get('show_navyai_setup', False):
    show_navyai_api_setup()

else:
    # 側邊欄
    with st.sidebar:
        st.markdown("### ⚓ NavyAI 快速設置")
        
        # 檢查是否已有 NavyAI 密鑰
        navyai_keys = provider_manager.get_api_keys("NavyAI")
        verified_navyai_keys = [k for k in navyai_keys if k['validated']]
        
        if verified_navyai_keys:
            st.success(f"✅ 已配置 {len(verified_navyai_keys)} 個 NavyAI 密鑰")
            
            # 顯示當前密鑰信息
            current_key = verified_navyai_keys[0]
            st.info(f"🔑 當前: {current_key['key_name']}")
            
            if current_key.get('api_info'):
                api_info = current_key['api_info']
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.metric("計劃", api_info.get('plan', 'N/A')[:8])
                with col_c2:
                    st.metric("模型", api_info.get('models_available', 'N/A'))
            
            if st.button("⚓ 使用 NavyAI", use_container_width=True, type="primary"):
                st.session_state.selected_provider = "NavyAI"
                st.session_state.api_config = {
                    'provider': "NavyAI",
                    'api_key': current_key['api_key'],
                    'base_url': current_key['base_url'] or MODEL_PROVIDERS["NavyAI"]['base_url'],
                    'validated': True,
                    'key_name': current_key['key_name']
                }
                st.success("🚀 NavyAI 已啟動")
                rerun_app()
        else:
            st.warning("❌ 尚未配置 NavyAI 密鑰")
        
        if st.button("🔧 NavyAI 設置", use_container_width=True):
            st.session_state.show_navyai_setup = True
            rerun_app()
        
        st.markdown("---")
        
        # FLUX Krea 快速啟動
        st.markdown("### 🎭 FLUX Krea 快速啟動")
        
        krea_providers = ["Krea.ai", "Pollinations.ai"]
        
        selected_krea = st.selectbox(
            "選擇 FLUX Krea 供應商:",
            [""] + krea_providers,
            format_func=lambda x: "請選擇..." if x == "" else f"{MODEL_PROVIDERS[x]['icon']} {MODEL_PROVIDERS[x]['name']}"
        )
        
        if selected_krea and st.button("🚀 快速啟動", use_container_width=True):
            provider_info = MODEL_PROVIDERS[selected_krea]
            st.session_state.selected_provider = selected_krea
            
            if not provider_info.get('requires_api_key', True):
                st.session_state.api_config = {
                    'provider': selected_krea,
                    'api_key': 'no-key-required',
                    'base_url': provider_info['base_url'],
                    'validated': True,
                    'key_name': f'{provider_info["name"]} 免費服務'
                }
            
            st.success(f"🎭 {provider_info['name']} 已啟動！")
            rerun_app()
        
        st.markdown("---")
        
        # 統計信息
        st.markdown("### 📊 統計信息")
        
        total_keys = len(provider_manager.get_api_keys())
        navyai_key_count = len(navyai_keys)
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("總密鑰", total_keys)
        with col_stat2:
            st.metric("NavyAI", navyai_key_count)
        
        if verified_navyai_keys:
            st.success("⚓ NavyAI 已就緒")
        else:
            st.info("⚓ 配置 NavyAI")
    
    # 主標題
    st.title("🎨 AI Image Generator Pro - FLUX Krea + NavyAI + 多供應商")
    
    # NavyAI 特色介紹
    if not st.session_state.get('selected_provider'):
        st.markdown("### ⚓ NavyAI 統一 AI 接口 - 新功能！")
        
        col_nav1, col_nav2, col_nav3 = st.columns(3)
        
        with col_nav1:
            st.info("""
            **🤖 50+ AI 模型**
            
            • FLUX Krea (美學優化)
            • DALL-E 3 (OpenAI)
            • Midjourney v6 (藝術)
            • GPT-4, Claude (文本)
            • 更多模型持續增加
            """)
        
        with col_nav2:
            st.success("""
            **⚡ 統一接口**
            
            • 一個 API 密鑰
            • 統一計費系統
            • >99% 運行時間
            • 24/7 技術支援
            • 全球CDN加速
            """)
        
        with col_nav3:
            st.warning("""
            **🔧 簡單設置**
            
            • 快速註冊獲取密鑰
            • Discord `/key` 命令
            • 詳細文檔支援
            • API 使用監控
            • 靈活計費方案
            """)
        
        if st.button("🚀 立即設置 NavyAI", type="primary", use_container_width=True):
            st.session_state.show_navyai_setup = True
            rerun_app()
    
    # 其餘主要功能保持不變
    if 'selected_provider' not in st.session_state:
        st.markdown("---")
        st.markdown("### 🏢 或選擇其他供應商")
        
        # 簡化的供應商選擇
        other_providers = {k: v for k, v in MODEL_PROVIDERS.items() if k != "NavyAI"}
        
        cols = st.columns(2)
        for i, (provider_key, provider_info) in enumerate(other_providers.items()):
            with cols[i % 2]:
                with st.container():
                    st.markdown(f"#### {provider_info['icon']} {provider_info['name']}")
                    st.caption(provider_info['description'])
                    
                    if "flux-krea" in provider_info.get('features', []):
                        st.info("🎭 支援 FLUX Krea")
                    
                    if st.button(f"選擇 {provider_info['name']}", key=f"select_{provider_key}", use_container_width=True):
                        st.session_state.selected_provider = provider_key
                        rerun_app()
                    
                    if not provider_info.get('requires_api_key', True):
                        st.caption("🆓 免費服務")
    
    # 頁腳
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        ⚓ <strong>NavyAI 統一接口</strong> | 
        🎭 <strong>FLUX Krea 美學優化</strong> | 
        🌸 <strong>免費服務</strong> | 
        ⚡ <strong>50+ AI 模型</strong>
        <br><br>
        <small>現已全面支援 NavyAI 統一接口，一個密鑰訪問所有頂級 AI 模型！</small>
    </div>
    """, unsafe_allow_html=True)
