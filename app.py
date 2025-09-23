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
    page_title="Flux AI & SD Generator Pro",
    page_icon="🎨",
    layout="wide"
)

# 密鑰管理系統 - 簡化版（避免復雜依賴）
class SimpleKeyManager:
    def __init__(self):
        self.db_path = "simple_keys.db"
        self.init_database()
    
    def init_database(self):
        """初始化簡單的密鑰存檔數據庫"""
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
        
        conn.commit()
        conn.close()
    
    def save_api_key(self, provider: str, key_name: str, api_key: str, base_url: str = "", 
                     notes: str = "", is_default: bool = False) -> str:
        """保存 API 密鑰"""
        key_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if is_default:
            cursor.execute(
                "UPDATE api_keys SET is_default = 0 WHERE provider = ?",
                (provider,)
            )
        
        cursor.execute('''
            INSERT INTO api_keys 
            (id, provider, key_name, api_key, base_url, notes, is_default)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (key_id, provider, key_name, api_key, base_url, notes, is_default))
        
        conn.commit()
        conn.close()
        
        return key_id
    
    def get_api_keys(self, provider: str = None) -> List[Dict]:
        """獲取 API 密鑰列表"""
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
                'id': row[0],
                'provider': row[1],
                'key_name': row[2],
                'api_key': row[3],
                'base_url': row[4],
                'validated': bool(row[5]),
                'created_at': row[6],
                'notes': row[7],
                'is_default': bool(row[8])
            })
        
        conn.close()
        return keys
    
    def delete_api_key(self, key_id: str):
        """刪除 API 密鑰"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
        conn.commit()
        conn.close()
    
    def update_key_validation(self, key_id: str, validated: bool):
        """更新密鑰驗證狀態"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE api_keys SET validated = ? WHERE id = ?",
            (validated, key_id)
        )
        conn.commit()
        conn.close()

# 全局密鑰管理器實例
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

def show_key_manager():
    """顯示密鑰管理界面"""
    st.subheader("🔐 API 密鑰管理中心")
    
    # 使用簡單的標題而非 tabs 來避免復雜性
    management_mode = st.radio(
        "選擇操作模式:",
        ["💾 保存密鑰", "📋 管理密鑰", "📊 統計信息"],
        horizontal=True
    )
    
    if management_mode == "💾 保存密鑰":
        st.markdown("### 💾 保存新的 API 密鑰")
        
        col_provider, col_name = st.columns(2)
        
        with col_provider:
            save_provider = st.selectbox(
                "選擇提供商:",
                list(API_PROVIDERS.keys()),
                format_func=lambda x: f"{API_PROVIDERS[x]['icon']} {API_PROVIDERS[x]['name']}",
                key="save_provider"
            )
        
        with col_name:
            key_name = st.text_input(
                "密鑰名稱:",
                placeholder="例如：主要密鑰、測試密鑰、備用密鑰",
                help="為此密鑰取一個便於識別的名稱"
            )
        
        provider_info = API_PROVIDERS[save_provider]
        
        # API 密鑰輸入
        new_api_key = st.text_input(
            "API 密鑰:",
            type="password",
            placeholder=f"輸入 {provider_info['name']} 的 API 密鑰...",
            help=f"密鑰通常以 '{provider_info['key_prefix']}' 開頭"
        )
        
        # 詳細配置區域 - 使用普通的 markdown 標題
        st.markdown("#### 📋 詳細配置（可選）")
        
        col_url, col_notes = st.columns(2)
        
        with col_url:
            save_base_url = st.text_input(
                "API 端點 URL:",
                value=provider_info['base_url_default'],
                help="API 服務的基礎 URL"
            )
        
        with col_notes:
            notes = st.text_area(
                "備註:",
                placeholder="記錄此密鑰的用途、限制或其他重要信息...",
                height=80
            )
        
        is_default = st.checkbox(
            "設為默認密鑰",
            help="將此密鑰設為該提供商的默認選擇"
        )
        
        # 保存按鈕
        col_save, col_test = st.columns(2)
        
        with col_save:
            if st.button("💾 保存密鑰", type="primary", use_container_width=True):
                if not key_name.strip():
                    st.error("❌ 請輸入密鑰名稱")
                elif not new_api_key.strip():
                    st.error("❌ 請輸入 API 密鑰")
                else:
                    try:
                        key_id = key_manager.save_api_key(
                            provider=save_provider,
                            key_name=key_name.strip(),
                            api_key=new_api_key.strip(),
                            base_url=save_base_url,
                            notes=notes,
                            is_default=is_default
                        )
                        
                        st.success(f"✅ 密鑰已安全保存！ID: {key_id[:8]}...")
                        time.sleep(1)
                        rerun_app()
                        
                    except Exception as e:
                        st.error(f"❌ 保存失敗: {str(e)}")
        
        with col_test:
            if st.button("🧪 測試並保存", use_container_width=True):
                if not key_name.strip() or not new_api_key.strip():
                    st.error("❌ 請填寫完整信息")
                else:
                    with st.spinner("正在測試 API 密鑰..."):
                        is_valid, message = validate_api_key(
                            new_api_key.strip(), save_base_url, save_provider
                        )
                        
                        if is_valid:
                            key_id = key_manager.save_api_key(
                                provider=save_provider,
                                key_name=key_name.strip(),
                                api_key=new_api_key.strip(),
                                base_url=save_base_url,
                                notes=notes,
                                is_default=is_default
                            )
                            
                            key_manager.update_key_validation(key_id, True)
                            st.success(f"✅ 測試成功並已保存！{message}")
                            time.sleep(1)
                            rerun_app()
                        else:
                            st.error(f"❌ 測試失敗: {message}")
    
    elif management_mode == "📋 管理密鑰":
        st.markdown("### 📋 已保存的 API 密鑰")
        
        all_keys = key_manager.get_api_keys()
        if not all_keys:
            st.info("📭 尚未保存任何 API 密鑰")
            return
        
        providers_with_keys = list(set(key['provider'] for key in all_keys))
        selected_provider_filter = st.selectbox(
            "篩選提供商:",
            ["全部"] + providers_with_keys,
            format_func=lambda x: x if x == "全部" else f"{API_PROVIDERS.get(x, {}).get('icon', '🔧')} {x}"
        )
        
        filtered_keys = all_keys if selected_provider_filter == "全部" else [
            key for key in all_keys if key['provider'] == selected_provider_filter
        ]
        
        st.info(f"顯示 {len(filtered_keys)} / {len(all_keys)} 個密鑰")
        
        # 顯示密鑰列表 - 使用簡單的容器而非 expander
        for i, key_info in enumerate(filtered_keys):
            provider_info = API_PROVIDERS.get(key_info['provider'], {})
            
            st.markdown("---")
            st.markdown(f"### {provider_info.get('icon', '🔧')} {key_info['key_name']}")
            
            col_info, col_actions = st.columns([2, 1])
            
            with col_info:
                st.markdown(f"**提供商**: {key_info['provider']}")
                st.markdown(f"**狀態**: {'🟢 已驗證' if key_info['validated'] else '🟡 未驗證'}")
                st.markdown(f"**默認**: {'✅ 是' if key_info['is_default'] else '❌ 否'}")
                st.markdown(f"**創建時間**: {key_info['created_at']}")
                
                if key_info['notes']:
                    st.markdown(f"**備註**: {key_info['notes']}")
                
                # 顯示密鑰（遮罩）
                masked_key = '*' * 20 + key_info['api_key'][-8:] if len(key_info['api_key']) > 8 else '*' * len(key_info['api_key'])
                st.markdown(f"**密鑰**: `{masked_key}`")
            
            with col_actions:
                # 使用此密鑰
                if st.button("✅ 使用", key=f"use_{key_info['id']}", use_container_width=True):
                    st.session_state.api_config = {
                        'provider': key_info['provider'],
                        'api_key': key_info['api_key'],
                        'base_url': key_info['base_url'],
                        'validated': key_info['validated'],
                        'key_id': key_info['id'],
                        'key_name': key_info['key_name']
                    }
                    
                    st.success(f"已載入: {key_info['key_name']}")
                    rerun_app()
                
                # 測試密鑰
                if st.button("🧪 測試", key=f"test_{key_info['id']}", use_container_width=True):
                    with st.spinner("測試中..."):
                        is_valid, message = validate_api_key(
                            key_info['api_key'], key_info['base_url'], key_info['provider']
                        )
                        
                        key_manager.update_key_validation(key_info['id'], is_valid)
                        
                        if is_valid:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                        
                        time.sleep(1)
                        rerun_app()
                
                # 刪除密鑰
                if st.button("🗑️ 刪除", key=f"delete_{key_info['id']}", use_container_width=True):
                    if st.session_state.get(f"confirm_delete_{key_info['id']}", False):
                        key_manager.delete_api_key(key_info['id'])
                        st.success("密鑰已刪除")
                        rerun_app()
                    else:
                        st.session_state[f"confirm_delete_{key_info['id']}"] = True
                        st.warning("再次點擊確認刪除")
    
    else:  # 統計信息
        st.markdown("### 📊 使用統計")
        
        all_keys = key_manager.get_api_keys()
        
        if not all_keys:
            st.info("📭 尚無統計數據")
            return
        
        # 概覽統計
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric("總密鑰數", len(all_keys))
        
        with col_stat2:
            validated_count = len([k for k in all_keys if k['validated']])
            st.metric("已驗證", validated_count)
        
        with col_stat3:
            providers_count = len(set(k['provider'] for k in all_keys))
            st.metric("提供商數", providers_count)
        
        # 按提供商分組統計
        st.markdown("#### 📈 按提供商統計")
        
        provider_stats = {}
        for key in all_keys:
            provider = key['provider']
            if provider not in provider_stats:
                provider_stats[provider] = {'total': 0, 'validated': 0}
            provider_stats[provider]['total'] += 1
            if key['validated']:
                provider_stats[provider]['validated'] += 1
        
        for provider, stats in provider_stats.items():
            provider_info = API_PROVIDERS.get(provider, {})
            icon = provider_info.get('icon', '🔧')
            
            col_provider, col_total, col_validated, col_rate = st.columns([2, 1, 1, 1])
            
            with col_provider:
                st.write(f"{icon} {provider}")
            with col_total:
                st.write(f"總數: {stats['total']}")
            with col_validated:
                st.write(f"已驗證: {stats['validated']}")
            with col_rate:
                rate = (stats['validated'] / stats['total'] * 100) if stats['total'] > 0 else 0
                st.write(f"驗證率: {rate:.1f}%")

def show_api_settings_with_keymanager():
    """顯示帶密鑰管理器的 API 設置界面"""
    st.subheader("🔑 API 設置與密鑰管理")
    
    # 顯示密鑰管理器 - 使用可摺疊區域
    show_manager = st.checkbox("🔐 顯示密鑰管理中心", value=False)
    
    if show_manager:
        show_key_manager()
    
    st.markdown("---")
    st.markdown("### ⚡ 快速設置")
    
    # 快速載入已保存的密鑰
    col_quick1, col_quick2 = st.columns(2)
    
    with col_quick1:
        st.markdown("#### 🚀 快速載入")
        
        all_keys = key_manager.get_api_keys()
        if all_keys:
            # 按提供商分組
            grouped_keys = {}
            for key in all_keys:
                provider = key['provider']
                if provider not in grouped_keys:
                    grouped_keys[provider] = []
                grouped_keys[provider].append(key)
            
            selected_provider = st.selectbox(
                "選擇提供商:",
                list(grouped_keys.keys()),
                format_func=lambda x: f"{API_PROVIDERS.get(x, {}).get('icon', '🔧')} {x}"
            )
            
            if selected_provider:
                provider_keys = grouped_keys[selected_provider]
                
                # 優先顯示默認密鑰
                default_keys = [k for k in provider_keys if k['is_default']]
                other_keys = [k for k in provider_keys if not k['is_default']]
                sorted_keys = default_keys + other_keys
                
                key_options = {
                    key['id']: f"{'⭐ ' if key['is_default'] else ''}{key['key_name']} "
                            f"({'🟢' if key['validated'] else '🟡'})"
                    for key in sorted_keys
                }
                
                selected_key_id = st.selectbox(
                    "選擇密鑰:",
                    list(key_options.keys()),
                    format_func=lambda x: key_options[x]
                )
                
                if st.button("⚡ 快速載入", type="primary", use_container_width=True):
                    selected_key = next(k for k in all_keys if k['id'] == selected_key_id)
                    
                    st.session_state.api_config = {
                        'provider': selected_key['provider'],
                        'api_key': selected_key['api_key'],
                        'base_url': selected_key['base_url'],
                        'validated': selected_key['validated'],
                        'key_id': selected_key_id,
                        'key_name': selected_key['key_name']
                    }
                    
                    st.success(f"✅ 已載入: {selected_key['key_name']}")
                    rerun_app()
        else:
            st.info("📭 尚未保存任何密鑰")
    
    with col_quick2:
        st.markdown("#### 🎯 當前配置")
        
        if st.session_state.api_config.get('api_key'):
            config = st.session_state.api_config
            provider_info = API_PROVIDERS.get(config['provider'], {})
            
            st.success("🟢 API 已配置")
            st.info(f"**提供商**: {provider_info.get('icon', '🔧')} {config['provider']}")
            
            if config.get('key_name'):
                st.info(f"**密鑰名稱**: {config['key_name']}")
            
            if config.get('validated'):
                st.success("✅ 已驗證")
            else:
                st.warning("⚠️ 未驗證")
            
            # 測試當前配置
            if st.button("🧪 測試當前配置", use_container_width=True):
                with st.spinner("測試中..."):
                    is_valid, message = validate_api_key(
                        config['api_key'], config['base_url'], config['provider']
                    )
                    
                    if is_valid:
                        st.success(f"✅ {message}")
                        config['validated'] = True
                        
                        # 更新數據庫中的驗證狀態
                        if config.get('key_id'):
                            key_manager.update_key_validation(config['key_id'], True)
                    else:
                        st.error(f"❌ {message}")
                        config['validated'] = False
                        
                        if config.get('key_id'):
                            key_manager.update_key_validation(config['key_id'], False)
        else:
            st.error("🔴 API 未配置")
            st.info("請使用上方的快速載入或新增密鑰")

def validate_api_key(api_key: str, base_url: str, provider: str) -> Tuple[bool, str]:
    """驗證 API 密鑰是否有效"""
    try:
        if provider == "Hugging Face":
            headers = {"Authorization": f"Bearer {api_key}"}
            test_url = f"{base_url}/models/stabilityai/stable-diffusion-xl-base-1.0"
            response = requests.get(test_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return True, "Hugging Face API 密鑰驗證成功"
            else:
                return False, f"HTTP {response.status_code}: 驗證失敗"
        else:
            test_client = OpenAI(api_key=api_key, base_url=base_url)
            response = test_client.models.list()
            return True, "API 密鑰驗證成功"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return False, "API 密鑰無效或已過期"
        elif "403" in error_msg or "Forbidden" in error_msg:
            return False, "API 密鑰沒有足夠權限"
        elif "404" in error_msg:
            return False, "API 端點不存在或不正確"
        elif "timeout" in error_msg.lower():
            return False, "API 連接超時"
        else:
            return False, f"API 驗證失敗: {error_msg[:100]}"

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

def init_api_client():
    """初始化 API 客戶端"""
    if 'api_config' in st.session_state and st.session_state.api_config.get('api_key'):
        config = st.session_state.api_config
        if config['provider'] == "Hugging Face":
            return None
        try:
            return OpenAI(
                api_key=config['api_key'],
                base_url=config['base_url']
            )
        except Exception:
            return None
    return None

# 初始化
init_session_state()
client = init_api_client()
api_configured = client is not None or (st.session_state.api_config.get('provider') == "Hugging Face" and st.session_state.api_config.get('api_key'))

# 側邊欄
with st.sidebar:
    show_api_settings_with_keymanager()
    st.markdown("---")
    
    # 顯示統計信息
    st.markdown("### 📊 密鑰統計")
    all_keys = key_manager.get_api_keys()
    validated_keys = [k for k in all_keys if k['validated']]
    
    col_total, col_valid = st.columns(2)
    with col_total:
        st.metric("總密鑰", len(all_keys))
    with col_valid:
        st.metric("已驗證", len(validated_keys))

# 主標題
st.title("🎨 Flux AI & SD Generator Pro - 密鑰存檔版")

# 主要內容
if api_configured:
    st.success("✅ API 配置完成，可以開始生成圖像")
    
    # 顯示當前使用的密鑰信息
    config = st.session_state.api_config
    if config.get('key_name'):
        st.info(f"🔑 當前使用: {config['key_name']} ({config['provider']})")
    
    # 簡單的圖像生成界面
    st.markdown("### 🎨 圖像生成界面")
    
    col_gen1, col_gen2 = st.columns([2, 1])
    
    with col_gen1:
        prompt = st.text_area(
            "輸入提示詞:",
            height=100,
            placeholder="描述您想要生成的圖像..."
        )
        
        if st.button("🚀 生成圖像", type="primary", disabled=not prompt.strip()):
            if prompt.strip():
                st.info("🚧 圖像生成功能開發中...")
                st.success("✅ API 配置正常，可以進行實際的圖像生成")
            else:
                st.warning("⚠️ 請輸入提示詞")
    
    with col_gen2:
        st.markdown("#### ℹ️ 生成設置")
        st.selectbox("圖像尺寸", ["512x512", "1024x1024", "1152x896"])
        st.slider("生成數量", 1, 4, 1)
        st.selectbox("模型選擇", ["flux.1-schnell", "stable-diffusion-xl"])
    
else:
    st.warning("⚠️ 請在側邊欄配置 API 密鑰")
    
    # 顯示幫助信息
    st.markdown("### 🔐 密鑰管理功能")
    st.markdown("""
    **功能亮點:**
    - 💾 **安全存檔**: 將多個 API 密鑰安全存儲在本地數據庫
    - ⚡ **快速切換**: 一鍵在不同密鑰和提供商間切換
    - 🧪 **自動驗證**: 保存前自動測試密鑰有效性
    - 📊 **使用統計**: 追蹤密鑰狀態和使用情況
    - 🏷️ **智能管理**: 為密鑰添加名稱和備註便於管理
    - ⭐ **默認設置**: 為每個提供商設置默認密鑰
    
    **支持的 API 提供商:**
    - ⚓ Navy API
    - 🤖 OpenAI Compatible
    - 🤗 Hugging Face
    - 🤝 Together AI
    
    **使用步驟:**
    1. 勾選「顯示密鑰管理中心」
    2. 選擇「保存密鑰」模式
    3. 輸入密鑰信息並保存
    4. 使用「快速載入」選擇密鑰
    5. 開始生成圖像
    """)

# 頁腳
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    🚀 <strong>Koyeb 部署</strong> | 
    🔐 <strong>安全密鑰存檔</strong> | 
    💾 <strong>多密鑰管理</strong> | 
    ⚡ <strong>快速切換</strong>
    <br><br>
    <small>支援本地 SQLite 存儲、多提供商管理和使用統計</small>
</div>
""", unsafe_allow_html=True)
