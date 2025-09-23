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
    page_title="Flux & SD Generator Pro - 自設供應商版",
    page_icon="🎨",
    layout="wide"
)

# 預設模型供應商配置
DEFAULT_MODEL_PROVIDERS = {
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
    }
}

# 自定義供應商和模型管理系統
class CustomProviderManager:
    def __init__(self):
        self.db_path = "custom_providers.db"
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
        all_providers = DEFAULT_MODEL_PROVIDERS.copy()
        
        custom_providers = self.get_custom_providers()
        for provider in custom_providers:
            all_providers[provider['provider_name']] = provider
        
        return all_providers
    
    def delete_custom_provider(self, provider_id: str):
        """刪除自定義供應商"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 軟刪除
        cursor.execute("UPDATE custom_providers SET is_active = 0 WHERE id = ?", (provider_id,))
        
        conn.commit()
        conn.close()
    
    def update_custom_provider(self, provider_id: str, **kwargs):
        """更新自定義供應商"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        update_fields = []
        values = []
        
        for field, value in kwargs.items():
            if field in ['display_name', 'icon', 'description', 'api_type', 'base_url', 
                        'key_prefix', 'pricing', 'speed', 'quality', 'auth_type']:
                update_fields.append(f"{field} = ?")
                values.append(value)
            elif field in ['features', 'headers']:
                update_fields.append(f"{field} = ?")
                values.append(json.dumps(value))
            elif field in ['timeout', 'max_retries', 'rate_limit']:
                update_fields.append(f"{field} = ?")
                values.append(int(value))
        
        if update_fields:
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            values.append(provider_id)
            
            query = f"UPDATE custom_providers SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(query, values)
            conn.commit()
        
        conn.close()
    
    # 其他方法保持不變...
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

# 全局實例
custom_provider_manager = CustomProviderManager()

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
        
        st.markdown("### 🔧 高級設置")
        
        with st.expander("高級配置（可選）"):
            col_timeout, col_retries, col_rate = st.columns(3)
            
            with col_timeout:
                timeout = st.number_input("超時時間 (秒)", min_value=5, max_value=300, value=30)
            
            with col_retries:
                max_retries = st.number_input("最大重試次數", min_value=0, max_value=10, value=3)
            
            with col_rate:
                rate_limit = st.number_input("速率限制 (請求/分鐘)", min_value=1, max_value=1000, value=60)
            
            # 自定義請求標頭
            st.markdown("#### 自定義 HTTP 標頭")
            custom_headers = {}
            
            header_count = st.number_input("標頭數量", min_value=0, max_value=10, value=0)
            
            for i in range(int(header_count)):
                col_header_key, col_header_value = st.columns(2)
                
                with col_header_key:
                    header_key = st.text_input(f"標頭名稱 {i+1}", key=f"header_key_{i}")
                
                with col_header_value:
                    header_value = st.text_input(f"標頭值 {i+1}", key=f"header_value_{i}")
                
                if header_key and header_value:
                    custom_headers[header_key] = header_value
        
        # 提交按鈕
        col_submit, col_test = st.columns(2)
        
        with col_submit:
            submit_button = st.form_submit_button("💾 創建供應商", type="primary", use_container_width=True)
        
        with col_test:
            test_button = st.form_submit_button("🧪 測試配置", use_container_width=True)
        
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
                    'headers': custom_headers,
                    'auth_type': auth_type,
                    'timeout': timeout,
                    'max_retries': max_retries,
                    'rate_limit': rate_limit
                }
                
                provider_id = custom_provider_manager.save_custom_provider(**provider_data)
                
                if provider_id:
                    st.success(f"✅ 自定義供應商 '{display_name}' 創建成功！")
                    st.info(f"🆔 供應商 ID: {provider_id[:8]}...")
                    time.sleep(1)
                    rerun_app()
                else:
                    st.error(f"❌ 創建失敗：供應商 ID '{provider_name}' 已存在")
        
        elif test_button:
            if not base_url:
                st.error("❌ 請填寫 API 端點 URL")
            else:
                # 測試配置
                with st.spinner("🧪 測試 API 配置..."):
                    test_result = test_custom_api_config(base_url, api_type, custom_headers, auth_type, timeout)
                    
                    if test_result['success']:
                        st.success(f"✅ {test_result['message']}")
                        if test_result.get('additional_info'):
                            st.info(f"ℹ️ {test_result['additional_info']}")
                    else:
                        st.error(f"❌ {test_result['message']}")

def test_custom_api_config(base_url: str, api_type: str, headers: Dict, auth_type: str, timeout: int) -> Dict:
    """測試自定義 API 配置"""
    try:
        test_headers = headers.copy()
        
        # 根據認證方式添加測試標頭
        if auth_type == "bearer":
            test_headers["Authorization"] = "Bearer test_token"
        elif auth_type == "api_key":
            test_headers["X-API-Key"] = "test_api_key"
        
        # 嘗試連接 API
        if api_type == "openai_compatible":
            # 測試 OpenAI 兼容端點
            test_url = f"{base_url.rstrip('/')}/models"
        elif api_type == "huggingface":
            # 測試 HuggingFace 端點
            test_url = f"{base_url.rstrip('/')}/models"
        else:
            # 通用測試
            test_url = base_url.rstrip('/')
        
        response = requests.get(test_url, headers=test_headers, timeout=timeout)
        
        if response.status_code == 200:
            return {
                'success': True,
                'message': f"API 端點連接成功 (HTTP {response.status_code})",
                'additional_info': f"響應時間: {response.elapsed.total_seconds():.2f}s"
            }
        elif response.status_code == 401:
            return {
                'success': True,
                'message': "API 端點可訪問，但需要有效認證",
                'additional_info': "這是正常的，請確保您有有效的 API 密鑰"
            }
        elif response.status_code == 403:
            return {
                'success': True,
                'message': "API 端點可訪問，但權限受限",
                'additional_info': "請檢查 API 密鑰權限"
            }
        else:
            return {
                'success': False,
                'message': f"API 返回異常狀態碼: {response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'message': f"連接超時（{timeout}秒）"
        }
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'message': "無法連接到 API 端點，請檢查 URL 是否正確"
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"測試失敗: {str(e)[:100]}"
        }

def show_custom_provider_manager():
    """顯示自定義供應商管理器"""
    st.subheader("🔧 自定義供應商管理")
    
    custom_providers = custom_provider_manager.get_custom_providers()
    
    if not custom_providers:
        st.info("📭 尚未創建任何自定義供應商")
        st.markdown("點擊下方按鈕創建您的第一個自定義供應商。")
        return
    
    st.info(f"📊 已創建 {len(custom_providers)} 個自定義供應商")
    
    for provider in custom_providers:
        with st.container():
            # 供應商信息展示
            col_info, col_actions = st.columns([3, 1])
            
            with col_info:
                st.markdown(f"### {provider['icon']} {provider['display_name']}")
                st.caption(f"**ID**: `{provider['provider_name']}` | **類型**: {provider['api_type']}")
                
                if provider['description']:
                    st.markdown(f"**描述**: {provider['description']}")
                
                st.markdown(f"**端點**: `{provider['base_url']}`")
                
                # 功能標籤
                if provider['features']:
                    features_text = " ".join([f"`{feature}`" for feature in provider['features']])
                    st.markdown(f"**功能**: {features_text}")
                
                # 性能指標
                st.markdown(f"**定價**: {provider['pricing']} | **速度**: {provider['speed']} | **品質**: {provider['quality']}")
            
            with col_actions:
                # 編輯按鈕
                if st.button("✏️ 編輯", key=f"edit_{provider['id']}", use_container_width=True):
                    st.session_state.editing_provider = provider
                    rerun_app()
                
                # 測試按鈕
                if st.button("🧪 測試", key=f"test_{provider['id']}", use_container_width=True):
                    with st.spinner("測試中..."):
                        test_result = test_custom_api_config(
                            provider['base_url'],
                            provider['api_type'],
                            provider['headers'],
                            provider['auth_type'],
                            provider['timeout']
                        )
                        
                        if test_result['success']:
                            st.success(f"✅ {test_result['message']}")
                        else:
                            st.error(f"❌ {test_result['message']}")
                
                # 刪除按鈕
                if st.button("🗑️ 刪除", key=f"delete_{provider['id']}", use_container_width=True):
                    if st.session_state.get(f"confirm_delete_{provider['id']}", False):
                        custom_provider_manager.delete_custom_provider(provider['id'])
                        st.success(f"已刪除供應商: {provider['display_name']}")
                        rerun_app()
                    else:
                        st.session_state[f"confirm_delete_{provider['id']}"] = True
                        st.warning("再次點擊確認刪除")
            
            st.markdown("---")

def show_provider_editor():
    """顯示供應商編輯器"""
    if 'editing_provider' not in st.session_state:
        return
    
    provider = st.session_state.editing_provider
    
    st.subheader(f"✏️ 編輯供應商: {provider['display_name']}")
    
    with st.form("edit_provider_form"):
        # 基本信息（供應商 ID 不可編輯）
        st.text_input("供應商 ID", value=provider['provider_name'], disabled=True)
        
        display_name = st.text_input("顯示名稱", value=provider['display_name'])
        
        col_icon, col_desc = st.columns([1, 3])
        
        with col_icon:
            icon = st.text_input("圖標", value=provider['icon'])
        
        with col_desc:
            description = st.text_area("描述", value=provider['description'], height=100)
        
        # API 配置
        col_type, col_url = st.columns(2)
        
        with col_type:
            api_type = st.selectbox(
                "API 類型",
                ["openai_compatible", "huggingface", "replicate", "custom"],
                index=["openai_compatible", "huggingface", "replicate", "custom"].index(provider['api_type'])
            )
        
        with col_url:
            base_url = st.text_input("API 端點 URL", value=provider['base_url'])
        
        # 功能支持
        features = st.multiselect(
            "支持的功能",
            ["flux", "stable-diffusion", "dall-e", "midjourney", "video-generation", "audio-generation", "custom-models"],
            default=provider['features']
        )
        
        # 性能指標
        col_pricing, col_speed, col_quality = st.columns(3)
        
        with col_pricing:
            pricing = st.text_input("定價模式", value=provider['pricing'])
        
        with col_speed:
            speed_options = ["極慢", "慢", "中等", "快速", "極快", "未知"]
            speed_index = speed_options.index(provider['speed']) if provider['speed'] in speed_options else 5
            speed = st.selectbox("速度等級", speed_options, index=speed_index)
        
        with col_quality:
            quality_options = ["低", "中", "高", "優秀", "頂級", "未知"]
            quality_index = quality_options.index(provider['quality']) if provider['quality'] in quality_options else 5
            quality = st.selectbox("品質等級", quality_options, index=quality_index)
        
        # 提交按鈕
        col_save, col_cancel = st.columns(2)
        
        with col_save:
            save_button = st.form_submit_button("💾 保存更改", type="primary", use_container_width=True)
        
        with col_cancel:
            cancel_button = st.form_submit_button("❌ 取消", use_container_width=True)
        
        if save_button:
            # 更新供應商信息
            update_data = {
                'display_name': display_name,
                'icon': icon,
                'description': description,
                'api_type': api_type,
                'base_url': base_url,
                'features': features,
                'pricing': pricing,
                'speed': speed,
                'quality': quality
            }
            
            custom_provider_manager.update_custom_provider(provider['id'], **update_data)
            st.success(f"✅ 供應商 '{display_name}' 已更新")
            
            del st.session_state.editing_provider
            rerun_app()
        
        elif cancel_button:
            del st.session_state.editing_provider
            rerun_app()

def validate_api_key(api_key: str, base_url: str, provider: str) -> Tuple[bool, str]:
    """驗證 API 密鑰是否有效"""
    try:
        all_providers = custom_provider_manager.get_all_providers()
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

def show_provider_selector():
    """顯示供應商選擇器（包含自定義供應商）"""
    st.subheader("🏢 選擇 API 供應商")
    
    # 獲取所有供應商
    all_providers = custom_provider_manager.get_all_providers()
    
    # 分類顯示
    default_providers = {k: v for k, v in all_providers.items() if not v.get('is_custom', False)}
    custom_providers = {k: v for k, v in all_providers.items() if v.get('is_custom', False)}
    
    # 預設供應商
    if default_providers:
        st.markdown("### 🏭 預設供應商")
        
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
    st.markdown("### 🏢 供應商狀態")
    
    if 'selected_provider' in st.session_state:
        provider = st.session_state.selected_provider
        all_providers = custom_provider_manager.get_all_providers()
        provider_info = all_providers.get(provider, {})
        
        if provider_info.get('is_custom'):
            st.success(f"{provider_info['icon']} {provider_info['display_name']} (自定義)")
        else:
            st.success(f"{provider_info['icon']} {provider_info['name']}")
        
        if api_configured:
            st.success("🟢 API 已配置")
        else:
            st.error("🔴 API 未配置")
    else:
        st.info("未選擇供應商")
    
    st.markdown("---")
    
    # 統計信息
    st.markdown("### 📊 統計")
    total_keys = len(custom_provider_manager.get_api_keys())
    custom_providers_count = len(custom_provider_manager.get_custom_providers())
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("API 密鑰", total_keys)
    with col_stat2:
        st.metric("自定義供應商", custom_providers_count)

# 主標題
st.title("🎨 Flux & SD Generator Pro - 自設供應商版")

# 主要內容
if 'show_custom_creator' in st.session_state and st.session_state.show_custom_creator:
    show_custom_provider_creator()
    if st.button("⬅️ 返回", key="back_from_creator"):
        del st.session_state.show_custom_creator
        rerun_app()

elif 'show_custom_manager' in st.session_state and st.session_state.show_custom_manager:
    if 'editing_provider' in st.session_state:
        show_provider_editor()
    else:
        show_custom_provider_manager()
    
    if st.button("⬅️ 返回", key="back_from_manager"):
        del st.session_state.show_custom_manager
        if 'editing_provider' in st.session_state:
            del st.session_state.editing_provider
        rerun_app()

else:
    show_provider_selector()

# 頁腳
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🛠️ <strong>自定義供應商支持</strong> | 
    🔧 <strong>靈活配置</strong> | 
    📊 <strong>統一管理</strong> | 
    🧪 <strong>配置測試</strong>
    <br><br>
    <small>支援創建和管理自定義 API 供應商，適配任何 AI 圖像生成服務</small>
</div>
""", unsafe_allow_html=True)
