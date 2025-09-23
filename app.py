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
from cryptography.fernet import Fernet
import hashlib

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
    page_title="Flux AI & SD Generator Pro - 密鑰存檔版",
    page_icon="🎨",
    layout="wide"
)

# 密鑰加密和存檔系統
class APIKeyManager:
    def __init__(self, db_path="api_keys.db"):
        self.db_path = db_path
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        self.init_database()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """獲取或創建加密密鑰"""
        key_file = "encryption.key"
        
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            return key
    
    def init_database(self):
        """初始化密鑰存檔數據庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                key_name TEXT NOT NULL,
                encrypted_key TEXT NOT NULL,
                base_url TEXT,
                key_prefix TEXT,
                validated BOOLEAN DEFAULT 0,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                is_default BOOLEAN DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS key_usage_logs (
                id TEXT PRIMARY KEY,
                key_id TEXT,
                action TEXT,
                success BOOLEAN,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(key_id) REFERENCES api_keys(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def encrypt_key(self, api_key: str) -> str:
        """加密 API 密鑰"""
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_key(self, encrypted_key: str) -> str:
        """解密 API 密鑰"""
        return self.cipher.decrypt(encrypted_key.encode()).decode()
    
    def save_api_key(self, provider: str, key_name: str, api_key: str, base_url: str = "", 
                     key_prefix: str = "", notes: str = "", is_default: bool = False) -> str:
        """保存 API 密鑰"""
        key_id = str(uuid.uuid4())
        encrypted_key = self.encrypt_key(api_key)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 如果設為默認，先清除其他默認設置
        if is_default:
            cursor.execute(
                "UPDATE api_keys SET is_default = 0 WHERE provider = ?",
                (provider,)
            )
        
        cursor.execute('''
            INSERT INTO api_keys 
            (id, provider, key_name, encrypted_key, base_url, key_prefix, notes, is_default)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (key_id, provider, key_name, encrypted_key, base_url, key_prefix, notes, is_default))
        
        conn.commit()
        conn.close()
        
        return key_id
    
    def get_api_keys(self, provider: str = None) -> List[Dict]:
        """獲取 API 密鑰列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if provider:
            cursor.execute('''
                SELECT id, provider, key_name, base_url, key_prefix, validated, 
                       last_used, created_at, notes, is_default
                FROM api_keys WHERE provider = ?
                ORDER BY is_default DESC, created_at DESC
            ''', (provider,))
        else:
            cursor.execute('''
                SELECT id, provider, key_name, base_url, key_prefix, validated, 
                       last_used, created_at, notes, is_default
                FROM api_keys 
                ORDER BY provider, is_default DESC, created_at DESC
            ''')
        
        keys = []
        for row in cursor.fetchall():
            keys.append({
                'id': row[0],
                'provider': row[1],
                'key_name': row[2],
                'base_url': row[3],
                'key_prefix': row[4],
                'validated': bool(row[5]),
                'last_used': row[6],
                'created_at': row[7],
                'notes': row[8],
                'is_default': bool(row[9])
            })
        
        conn.close()
        return keys
    
    def get_decrypted_key(self, key_id: str) -> Optional[str]:
        """獲取解密的 API 密鑰"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT encrypted_key FROM api_keys WHERE id = ?",
            (key_id,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return self.decrypt_key(result[0])
        return None
    
    def update_key_validation(self, key_id: str, validated: bool):
        """更新密鑰驗證狀態"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE api_keys 
            SET validated = ?, last_used = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (validated, key_id))
        
        conn.commit()
        conn.close()
    
    def delete_api_key(self, key_id: str):
        """刪除 API 密鑰"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
        cursor.execute("DELETE FROM key_usage_logs WHERE key_id = ?", (key_id,))
        
        conn.commit()
        conn.close()
    
    def log_key_usage(self, key_id: str, action: str, success: bool, error_message: str = ""):
        """記錄密鑰使用日誌"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO key_usage_logs (id, key_id, action, success, error_message)
            VALUES (?, ?, ?, ?, ?)
        ''', (str(uuid.uuid4()), key_id, action, success, error_message))
        
        conn.commit()
        conn.close()
    
    def get_default_key(self, provider: str) -> Optional[Dict]:
        """獲取默認密鑰"""
        keys = self.get_api_keys(provider)
        default_keys = [k for k in keys if k['is_default']]
        return default_keys[0] if default_keys else None
    
    def export_keys(self, include_keys: bool = False) -> str:
        """導出密鑰配置（可選擇是否包含密鑰本身）"""
        keys = self.get_api_keys()
        export_data = []
        
        for key_info in keys:
            export_item = {
                'provider': key_info['provider'],
                'key_name': key_info['key_name'],
                'base_url': key_info['base_url'],
                'key_prefix': key_info['key_prefix'],
                'notes': key_info['notes'],
                'is_default': key_info['is_default']
            }
            
            if include_keys:
                decrypted_key = self.get_decrypted_key(key_info['id'])
                export_item['api_key'] = decrypted_key
            
            export_data.append(export_item)
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)

# 全局密鑰管理器實例
key_manager = APIKeyManager()

# API 提供商配置（增強版）
API_PROVIDERS = {
    "OpenAI Compatible": {
        "name": "OpenAI Compatible API",
        "base_url_default": "https://api.openai.com/v1",
        "key_prefix": "sk-",
        "description": "OpenAI 官方或兼容的 API 服務",
        "icon": "🤖",
        "supports": ["flux", "stable-diffusion"]
    },
    "Navy": {
        "name": "Navy API",
        "base_url_default": "https://api.navy/v1", 
        "key_prefix": "sk-",
        "description": "Navy 提供的 AI 圖像生成服務",
        "icon": "⚓",
        "supports": ["flux", "stable-diffusion"]
    },
    "Hugging Face": {
        "name": "Hugging Face API",
        "base_url_default": "https://api-inference.huggingface.co",
        "key_prefix": "hf_",
        "description": "Hugging Face 推理 API",
        "icon": "🤗",
        "supports": ["flux", "stable-diffusion"]
    },
    "Together AI": {
        "name": "Together AI",
        "base_url_default": "https://api.together.xyz/v1",
        "key_prefix": "",
        "description": "Together AI 平台",
        "icon": "🤝",
        "supports": ["flux", "stable-diffusion"]
    },
    "Fireworks AI": {
        "name": "Fireworks AI",
        "base_url_default": "https://api.fireworks.ai/inference/v1",
        "key_prefix": "",
        "description": "Fireworks AI 快速推理",
        "icon": "🎆",
        "supports": ["flux", "stable-diffusion"]
    },
    "Replicate": {
        "name": "Replicate AI",
        "base_url_default": "https://api.replicate.com/v1",
        "key_prefix": "r8_",
        "description": "Replicate 雲端 AI 模型平台",
        "icon": "🔄",
        "supports": ["flux", "stable-diffusion"]
    }
}

def show_key_manager():
    """顯示密鑰管理界面"""
    st.subheader("🔐 API 密鑰管理中心")
    
    # 標籤頁
    key_tabs = st.tabs(["💾 存檔密鑰", "📋 管理密鑰", "⚙️ 密鑰設置", "📊 使用統計"])
    
    # 存檔密鑰標籤
    with key_tabs[0]:
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
        
        # 可選配置
        with st.expander("📋 詳細配置（可選）"):
            col_url, col_prefix = st.columns(2)
            
            with col_url:
                save_base_url = st.text_input(
                    "API 端點 URL:",
                    value=provider_info['base_url_default'],
                    help="API 服務的基礎 URL"
                )
            
            with col_prefix:
                save_key_prefix = st.text_input(
                    "密鑰前綴:",
                    value=provider_info['key_prefix'],
                    help="API 密鑰的前綴格式"
                )
            
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
                            key_prefix=save_key_prefix,
                            notes=notes,
                            is_default=is_default
                        )
                        
                        st.success(f"✅ 密鑰已安全保存！ID: {key_id[:8]}...")
                        key_manager.log_key_usage(key_id, "save", True)
                        
                        # 清空表單
                        st.rerun()
                        
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
                                key_prefix=save_key_prefix,
                                notes=notes,
                                is_default=is_default
                            )
                            
                            key_manager.update_key_validation(key_id, True)
                            key_manager.log_key_usage(key_id, "test_and_save", True)
                            
                            st.success(f"✅ 測試成功並已保存！{message}")
                            st.rerun()
                        else:
                            st.error(f"❌ 測試失敗: {message}")
    
    # 管理密鑰標籤
    with key_tabs[1]:
        st.markdown("### 📋 已保存的 API 密鑰")
        
        # 按提供商篩選
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
        
        # 篩選密鑰
        filtered_keys = all_keys if selected_provider_filter == "全部" else [
            key for key in all_keys if key['provider'] == selected_provider_filter
        ]
        
        st.info(f"顯示 {len(filtered_keys)} / {len(all_keys)} 個密鑰")
        
        # 顯示密鑰列表
        for key_info in filtered_keys:
            provider_info = API_PROVIDERS.get(key_info['provider'], {})
            
            with st.expander(
                f"{provider_info.get('icon', '🔧')} {key_info['key_name']} "
                f"({'✅ 默認' if key_info['is_default'] else ''}) "
                f"({'🟢 已驗證' if key_info['validated'] else '🟡 未驗證'})"
            ):
                col_info, col_actions = st.columns([2, 1])
                
                with col_info:
                    st.markdown(f"**提供商**: {key_info['provider']}")
                    st.markdown(f"**名稱**: {key_info['key_name']}")
                    st.markdown(f"**狀態**: {'🟢 已驗證' if key_info['validated'] else '🟡 未驗證'}")
                    st.markdown(f"**創建時間**: {key_info['created_at']}")
                    
                    if key_info['last_used']:
                        st.markdown(f"**最後使用**: {key_info['last_used']}")
                    
                    if key_info['notes']:
                        st.markdown(f"**備註**: {key_info['notes']}")
                    
                    st.markdown(f"**端點**: {key_info['base_url']}")
                
                with col_actions:
                    # 使用此密鑰
                    if st.button("✅ 使用", key=f"use_{key_info['id']}", use_container_width=True):
                        decrypted_key = key_manager.get_decrypted_key(key_info['id'])
                        if decrypted_key:
                            st.session_state.api_config = {
                                'provider': key_info['provider'],
                                'api_key': decrypted_key,
                                'base_url': key_info['base_url'],
                                'validated': key_info['validated'],
                                'key_id': key_info['id'],
                                'key_name': key_info['key_name']
                            }
                            
                            key_manager.log_key_usage(key_info['id'], "use", True)
                            st.success(f"已載入: {key_info['key_name']}")
                            rerun_app()
                    
                    # 測試密鑰
                    if st.button("🧪 測試", key=f"test_{key_info['id']}", use_container_width=True):
                        decrypted_key = key_manager.get_decrypted_key(key_info['id'])
                        if decrypted_key:
                            with st.spinner("測試中..."):
                                is_valid, message = validate_api_key(
                                    decrypted_key, key_info['base_url'], key_info['provider']
                                )
                                
                                key_manager.update_key_validation(key_info['id'], is_valid)
                                key_manager.log_key_usage(
                                    key_info['id'], "test", is_valid, message if not is_valid else ""
                                )
                                
                                if is_valid:
                                    st.success(f"✅ {message}")
                                else:
                                    st.error(f"❌ {message}")
                                
                                rerun_app()
                    
                    # 設為默認
                    if not key_info['is_default']:
                        if st.button("⭐ 設為默認", key=f"default_{key_info['id']}", use_container_width=True):
                            # 清除同提供商的其他默認設置
                            conn = sqlite3.connect(key_manager.db_path)
                            cursor = conn.cursor()
                            cursor.execute(
                                "UPDATE api_keys SET is_default = 0 WHERE provider = ?",
                                (key_info['provider'],)
                            )
                            cursor.execute(
                                "UPDATE api_keys SET is_default = 1 WHERE id = ?",
                                (key_info['id'],)
                            )
                            conn.commit()
                            conn.close()
                            
                            st.success("已設為默認密鑰")
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
                    
                    # 顯示密鑰（危險操作）
                    if st.button("👁️ 顯示密鑰", key=f"show_{key_info['id']}", use_container_width=True):
                        if st.session_state.get(f"confirm_show_{key_info['id']}", False):
                            decrypted_key = key_manager.get_decrypted_key(key_info['id'])
                            st.code(decrypted_key, language="text")
                            key_manager.log_key_usage(key_info['id'], "view", True)
                        else:
                            st.session_state[f"confirm_show_{key_info['id']}"] = True
                            st.warning("⚠️ 再次點擊確認顯示（注意安全）")
    
    # 密鑰設置標籤
    with key_tabs[2]:
        st.markdown("### ⚙️ 密鑰管理設置")
        
        col_export, col_import = st.columns(2)
        
        with col_export:
            st.markdown("#### 📤 導出設置")
            
            include_keys_in_export = st.checkbox(
                "包含密鑰內容",
                help="⚠️ 勾選此項將會在導出文件中包含實際的 API 密鑰，請謹慎處理"
            )
            
            if st.button("📤 導出配置", use_container_width=True):
                export_data = key_manager.export_keys(include_keys_in_export)
                
                # 生成下載文件
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"api_keys_export_{timestamp}.json"
                
                st.download_button(
                    label="⬇️ 下載導出文件",
                    data=export_data,
                    file_name=filename,
                    mime="application/json",
                    use_container_width=True
                )
                
                if include_keys_in_export:
                    st.warning("⚠️ 導出文件包含實際密鑰，請妥善保管！")
        
        with col_import:
            st.markdown("#### 📥 導入設置")
            
            uploaded_file = st.file_uploader(
                "選擇配置文件",
                type=['json'],
                help="上傳之前導出的 API 密鑰配置文件"
            )
            
            if uploaded_file is not None:
                try:
                    import_data = json.load(uploaded_file)
                    
                    st.info(f"發現 {len(import_data)} 個密鑰配置")
                    
                    if st.button("📥 導入配置", type="primary", use_container_width=True):
                        import_count = 0
                        
                        for key_config in import_data:
                            if 'api_key' in key_config and key_config['api_key']:
                                key_manager.save_api_key(
                                    provider=key_config['provider'],
                                    key_name=key_config['key_name'],
                                    api_key=key_config['api_key'],
                                    base_url=key_config.get('base_url', ''),
                                    key_prefix=key_config.get('key_prefix', ''),
                                    notes=key_config.get('notes', ''),
                                    is_default=key_config.get('is_default', False)
                                )
                                import_count += 1
                        
                        st.success(f"✅ 成功導入 {import_count} 個密鑰配置")
                        rerun_app()
                        
                except Exception as e:
                    st.error(f"❌ 導入失敗: {str(e)}")
        
        # 安全設置
        st.markdown("#### 🔒 安全設置")
        
        col_security1, col_security2 = st.columns(2)
        
        with col_security1:
            if st.button("🔄 重新生成加密密鑰", use_container_width=True):
                st.warning("⚠️ 此操作將使所有已保存的密鑰無法解密！")
                if st.checkbox("我了解風險，確認操作"):
                    # 這裡可以實現重新加密功能
                    st.info("🚧 重新加密功能開發中")
        
        with col_security2:
            if st.button("🗑️ 清空所有密鑰", use_container_width=True):
                if st.checkbox("確認刪除所有密鑰"):
                    conn = sqlite3.connect(key_manager.db_path)
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM api_keys")
                    cursor.execute("DELETE FROM key_usage_logs")
                    conn.commit()
                    conn.close()
                    
                    st.success("所有密鑰已清除")
                    rerun_app()
    
    # 使用統計標籤
    with key_tabs[3]:
        st.markdown("### 📊 使用統計")
        
        all_keys = key_manager.get_api_keys()
        
        if not all_keys:
            st.info("📭 尚無統計數據")
            return
        
        # 概覽統計
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("總密鑰數", len(all_keys))
        
        with col_stat2:
            validated_count = len([k for k in all_keys if k['validated']])
            st.metric("已驗證", validated_count)
        
        with col_stat3:
            providers_count = len(set(k['provider'] for k in all_keys))
            st.metric("提供商數", providers_count)
        
        with col_stat4:
            default_count = len([k for k in all_keys if k['is_default']])
            st.metric("默認密鑰", default_count)
        
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
    
    # 顯示密鑰管理器
    with st.expander("🔐 密鑰管理中心", expanded=False):
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
                    decrypted_key = key_manager.get_decrypted_key(selected_key_id)
                    
                    if decrypted_key:
                        st.session_state.api_config = {
                            'provider': selected_key['provider'],
                            'api_key': decrypted_key,
                            'base_url': selected_key['base_url'],
                            'validated': selected_key['validated'],
                            'key_id': selected_key_id,
                            'key_name': selected_key['key_name']
                        }
                        
                        key_manager.log_key_usage(selected_key_id, "quick_load", True)
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
                            key_manager.log_key_usage(config['key_id'], "test_current", True)
                    else:
                        st.error(f"❌ {message}")
                        config['validated'] = False
                        
                        if config.get('key_id'):
                            key_manager.update_key_validation(config['key_id'], False)
                            key_manager.log_key_usage(config['key_id'], "test_current", False, message)
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
    
    # 這裡可以添加圖像生成的主要界面
    st.markdown("### 🎨 圖像生成界面")
    st.info("🚧 圖像生成界面開發中...")
    
else:
    st.warning("⚠️ 請在側邊欄配置 API 密鑰")
    
    # 顯示幫助信息
    st.markdown("### 🔐 密鑰管理功能")
    st.markdown("""
    **新功能亮點:**
    - 🔒 **安全加密**: 所有 API 密鑰使用 AES 加密存儲
    - 💾 **多密鑰管理**: 支持保存多個提供商的多個密鑰
    - ⚡ **快速切換**: 一鍵在不同密鑰間切換
    - 📊 **使用統計**: 跟蹤密鑰使用情況和驗證狀態
    - 📤 **導出導入**: 安全地備份和恢復密鑰配置
    - 🎯 **默認設置**: 為每個提供商設置默認密鑰
    
    **使用步驟:**
    1. 點擊側邊欄的「密鑰管理中心」
    2. 在「存檔密鑰」標籤中添加您的 API 密鑰
    3. 使用「快速載入」選擇要使用的密鑰
    4. 開始生成圖像
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
    <small>支援 AES 加密存儲、多提供商管理、使用統計和安全備份</small>
</div>
""", unsafe_allow_html=True)
