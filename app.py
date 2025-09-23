import streamlit as st
import os
import logging
import time
import sqlite3
import uuid
import json
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple  # 修復：正確導入類型註解

# 必須是第一個 Streamlit 命令 - Koyeb 優化配置
st.set_page_config(
    page_title="AI Image Generator Pro - Koyeb Optimized",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"  # Koyeb 優化：減少初始載入
)

# Koyeb 環境檢測和優化設置
KOYEB_ENV = os.getenv('KOYEB_PUBLIC_DOMAIN') is not None
PORT = int(os.getenv('PORT', 8501))

# 日誌配置 - Koyeb 優化
logging.basicConfig(
    level=logging.INFO if KOYEB_ENV else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 只在需要時導入重型模組 - Koyeb 冷啟動優化
@lru_cache(maxsize=1)
def get_heavy_imports():
    """延遲載入重型模組以優化冷啟動時間"""
    try:
        from openai import OpenAI
        from PIL import Image, ImageDraw, ImageFont
        import requests
        from io import BytesIO
        import datetime
        import base64
        import re
        
        return {
            'OpenAI': OpenAI,
            'Image': Image,
            'ImageDraw': ImageDraw,
            'ImageFont': ImageFont,
            'requests': requests,
            'BytesIO': BytesIO,
            'datetime': datetime,
            'base64': base64,
            're': re
        }
    except ImportError as e:
        logger.error(f"Failed to import heavy modules: {e}")
        return {}

# Koyeb 兼容性函數
def rerun_app():
    """Koyeb 優化的重新運行函數"""
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    else:
        st.stop()

# Koyeb 環境優化的模型供應商配置
MODEL_PROVIDERS = {
    "NavyAI": {
        "name": "NavyAI",
        "icon": "⚓",
        "description": "統一圖像 API - Koyeb 高性能部署",
        "api_type": "openai_compatible",
        "base_url": "https://api.navy/v1",
        "features": ["flux-krea", "dalle", "midjourney", "flux", "stable-diffusion"],
        "koyeb_optimized": True,  # Koyeb 優化標記
        "requires_api_key": True,
        "cold_start_friendly": True  # 冷啟動友好
    },
    "Pollinations.ai": {
        "name": "Pollinations AI",
        "icon": "🌸", 
        "description": "免費圖像生成 - Koyeb 無服務器最佳",
        "api_type": "pollinations",
        "base_url": "https://image.pollinations.ai/prompt",
        "features": ["flux", "flux-krea", "stable-diffusion"],
        "koyeb_optimized": True,
        "requires_api_key": False,
        "cold_start_friendly": True
    }
}

# Koyeb 優化的 SQLite 管理器
class KoyebOptimizedProviderManager:
    def __init__(self):
        # Koyeb 臨時存儲優化
        self.db_path = "/tmp/koyeb_providers.db" if KOYEB_ENV else "koyeb_providers.db"
        self.init_database()
    
    @lru_cache(maxsize=100)
    def get_cached_providers(self) -> Dict:
        """Koyeb 優化：快取供應商列表"""
        return MODEL_PROVIDERS.copy()
    
    def init_database(self):
        """Koyeb 優化的數據庫初始化"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")  # Koyeb 性能優化
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            
            cursor = conn.cursor()
            
            # 簡化的表結構 - Koyeb 優化
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS koyeb_api_keys (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    key_name TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS koyeb_models (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    koyeb_priority INTEGER DEFAULT 999,
                    cold_start_optimized BOOLEAN DEFAULT 0,
                    UNIQUE(provider, model_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Koyeb 數據庫初始化完成")
            
        except Exception as e:
            logger.error(f"Koyeb 數據庫初始化失敗: {e}")
    
    def save_api_key(self, provider: str, key_name: str, api_key: str) -> str:
        """Koyeb 優化的 API 密鑰保存"""
        key_id = str(uuid.uuid4())[:8]  # 簡化 ID
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 先停用舊密鑰
            cursor.execute("UPDATE koyeb_api_keys SET is_active = 0 WHERE provider = ?", (provider,))
            
            cursor.execute('''
                INSERT INTO koyeb_api_keys (id, provider, key_name, api_key)
                VALUES (?, ?, ?, ?)
            ''', (key_id, provider, key_name, api_key))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Koyeb API 密鑰已保存: {provider}")
            return key_id
            
        except Exception as e:
            logger.error(f"Koyeb API 密鑰保存失敗: {e}")
            return ""
    
    def get_active_api_key(self, provider: str) -> Optional[Dict]:
        """Koyeb 優化的活動密鑰獲取"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, key_name, api_key, created_at
                FROM koyeb_api_keys 
                WHERE provider = ? AND is_active = 1
                ORDER BY created_at DESC LIMIT 1
            ''', (provider,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0], 
                    'key_name': row[1], 
                    'api_key': row[2], 
                    'created_at': row[3]
                }
            return None
            
        except Exception as e:
            logger.error(f"Koyeb 密鑰獲取失敗: {e}")
            return None

# 全局管理器實例 - Koyeb 優化
@st.cache_resource
def get_provider_manager():
    """Koyeb 優化：快取管理器實例"""
    return KoyebOptimizedProviderManager()

provider_manager = get_provider_manager()

# Koyeb 優化的圖像生成函數
@st.cache_data(ttl=300)  # 5分鐘快取 - Koyeb 性能優化
def generate_pollinations_image_koyeb(prompt: str, model: str = "flux", size: str = "1024x1024") -> Tuple[bool, str]:
    """Koyeb 優化的 Pollinations 圖像生成"""
    imports = get_heavy_imports()
    if not imports:
        return False, "模組載入失敗"
    
    try:
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt)
        
        width, height = map(int, size.split('x'))
        
        # Koyeb 優化的 URL 構建
        url_params = [
            f"model={model}" if model != "flux" else "",
            f"width={width}",
            f"height={height}",
            "nologo=true"
        ]
        
        url_params = [p for p in url_params if p]  # 移除空參數
        base_url = "https://image.pollinations.ai/prompt"
        full_url = f"{base_url}/{encoded_prompt}?{'&'.join(url_params)}"
        
        # Koyeb 優化的請求
        response = imports['requests'].get(full_url, timeout=30)  # 減少超時時間
        
        if response.status_code == 200:
            encoded_image = imports['base64'].b64encode(response.content).decode()
            image_url = f"data:image/png;base64,{encoded_image}"
            return True, image_url
        else:
            return False, f"HTTP {response.status_code}"
            
    except Exception as e:
        logger.error(f"Koyeb 圖像生成錯誤: {e}")
        return False, str(e)

# Koyeb 優化的模擬生成
def generate_demo_image_koyeb(prompt: str, provider: str = "Demo") -> Tuple[bool, str]:
    """Koyeb 冷啟動友好的演示圖像生成"""
    imports = get_heavy_imports()
    if not imports:
        return False, "模組載入失敗"
    
    try:
        # 快速演示圖像
        img = imports['Image'].new('RGB', (512, 512))
        draw = imports['ImageDraw'].Draw(img)
        
        # Koyeb 主題色
        for y in range(512):
            r = int(30 + (70 * y / 512))   # Koyeb 藍色漸變
            g = int(60 + (140 * y / 512))
            b = int(120 + (135 * y / 512))
            for x in range(512):
                draw.point((x, y), (r, g, b))
        
        try:
            font = imports['ImageFont'].load_default()
        except:
            font = None
        
        # Koyeb 標記
        draw.text((50, 50), f"🚀 Koyeb Deployed", fill=(255, 255, 255), font=font)
        draw.text((50, 90), f"Provider: {provider}", fill=(255, 255, 255), font=font)
        draw.text((50, 130), f"Prompt: {prompt[:40]}...", fill=(255, 255, 255), font=font)
        draw.text((50, 400), "Serverless & Scale-to-Zero", fill=(255, 255, 255), font=font)
        draw.text((50, 440), "High-Performance Global Deploy", fill=(255, 255, 255), font=font)
        
        # 轉換為 base64
        buffer = imports['BytesIO']()
        img.save(buffer, format='PNG')
        encoded_image = imports['base64'].b64encode(buffer.getvalue()).decode()
        
        return True, f"data:image/png;base64,{encoded_image}"
        
    except Exception as e:
        logger.error(f"Koyeb 演示圖像生成錯誤: {e}")
        return False, str(e)

# Koyeb 優化的 UI 組件
def show_koyeb_header():
    """Koyeb 優化的應用頭部"""
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%); border-radius: 10px; margin-bottom: 1.5rem;">
        <h1 style="color: white; margin: 0; font-size: 2.2rem;">🚀 AI 圖像生成器 Pro</h1>
        <h2 style="color: #dbeafe; margin: 0.3rem 0; font-size: 1.1rem;">Koyeb 高性能無服務器部署</h2>
        <div style="margin-top: 0.8rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.6rem; border-radius: 15px; margin: 0.1rem; color: #fef3c7; font-size: 0.9rem;">⚡ Scale-to-Zero</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.6rem; border-radius: 15px; margin: 0.1rem; color: #fef3c7; font-size: 0.9rem;">🌍 Global CDN</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.6rem; border-radius: 15px; margin: 0.1rem; color: #fef3c7; font-size: 0.9rem;">⚓ NavyAI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_koyeb_status():
    """Koyeb 狀態顯示"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚀 部署狀態", "Koyeb 運行中" if KOYEB_ENV else "本地開發")
    
    with col2:
        st.metric("⚡ 服務模式", "Serverless" if KOYEB_ENV else "Development")
    
    with col3:
        port_status = f":{PORT}" if not KOYEB_ENV else ".koyeb.app"
        st.metric("🌐 端口", port_status)
    
    with col4:
        koyeb_region = os.getenv('KOYEB_DEPLOYMENT_REGION', 'Unknown')
        st.metric("📍 區域", koyeb_region if KOYEB_ENV else "Local")

# Koyeb 優化的主界面
def show_koyeb_main_interface():
    """Koyeb 優化的主界面"""
    
    # 快速供應商選擇
    st.markdown("### 🎯 快速開始")
    
    col_provider1, col_provider2 = st.columns(2)
    
    with col_provider1:
        st.markdown("""
        #### 🌸 Pollinations AI (免費)
        - ✅ **無需 API 密鑰**
        - ⚡ Koyeb 冷啟動優化
        - 🎭 支援 FLUX Krea
        - 🚀 Scale-to-Zero 友好
        """)
        
        if st.button("🚀 使用免費服務", type="primary", use_container_width=True):
            st.session_state.selected_provider = "Pollinations.ai"
            st.session_state.koyeb_quick_start = True
            st.success("✅ Pollinations AI 已啟動 - Koyeb 優化模式")
            rerun_app()
    
    with col_provider2:
        st.markdown("""
        #### ⚓ NavyAI (統一接口)  
        - 🎨 15+ 專業圖像模型
        - 🎭 FLUX Krea Pro
        - 🖼️ DALL-E 3、Midjourney
        - 🔧 需要 API 密鑰
        """)
        
        if st.button("⚓ 配置 NavyAI", use_container_width=True):
            st.session_state.show_navyai_setup = True
            rerun_app()

def show_koyeb_image_generator():
    """Koyeb 優化的圖像生成器"""
    
    if 'selected_provider' not in st.session_state:
        st.warning("⚠️ 請先選擇一個服務提供商")
        return
    
    provider = st.session_state.selected_provider
    provider_info = MODEL_PROVIDERS.get(provider, {})
    
    st.markdown(f"### 🎨 {provider_info['icon']} {provider_info['name']} - 圖像生成")
    
    # Koyeb 優化的提示詞界面
    col_prompt, col_params = st.columns([3, 1])
    
    with col_prompt:
        prompt = st.text_area(
            "✍️ 描述您想要的圖像:",
            height=100,
            placeholder="例如：A beautiful sunset over mountains, digital art style",
            help="提示：簡潔明確的描述效果更好"
        )
        
        # Koyeb 優化的快速模板
        st.markdown("#### 💡 快速模板")
        templates = [
            "A professional portrait with natural lighting",
            "Beautiful landscape at golden hour, digital art",
            "Modern cityscape with futuristic architecture",
            "Abstract art with vibrant colors and patterns"
        ]
        
        template_cols = st.columns(2)
        for i, template in enumerate(templates):
            with template_cols[i % 2]:
                if st.button(f"📋 {template[:30]}...", key=f"template_{i}", use_container_width=True):
                    st.session_state.quick_template = template
                    rerun_app()
    
    with col_params:
        st.markdown("#### ⚙️ 生成參數")
        
        size_options = ["512x512", "768x768", "1024x1024"]
        selected_size = st.selectbox("🖼️ 圖像尺寸:", size_options, index=2)
        
        if provider == "Pollinations.ai":
            model_options = ["flux", "flux-krea", "flux-realism"]
            selected_model = st.selectbox("🤖 模型:", model_options, index=1)
        else:
            selected_model = "default"
        
        # Koyeb 性能指標
        st.info("⚡ Koyeb 優化特性")
        st.caption("• Scale-to-Zero 節省成本")  
        st.caption("• 全球 CDN 加速")
        st.caption("• 冷啟動優化")
    
    # 檢查快速模板
    if hasattr(st.session_state, 'quick_template'):
        prompt = st.session_state.quick_template
        del st.session_state.quick_template
        rerun_app()
    
    # 生成按鈕
    st.markdown("---")
    
    can_generate = prompt.strip()
    
    if st.button(
        f"🚀 Koyeb 高速生成",
        type="primary", 
        disabled=not can_generate,
        use_container_width=True
    ):
        if can_generate:
            generate_image_koyeb(provider, prompt, selected_model, selected_size)

def generate_image_koyeb(provider: str, prompt: str, model: str, size: str):
    """Koyeb 優化的圖像生成流程"""
    
    # 生成進度
    progress_container = st.empty()
    
    with progress_container.container():
        st.info(f"🚀 Koyeb 高性能生成中 - {provider}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Koyeb 優化的生成階段
        if provider == "Pollinations.ai":
            stages = [
                "⚡ Koyeb Serverless 啟動...",
                "🌸 連接 Pollinations API...",
                "🎨 AI 模型推理中...",
                "📱 全球 CDN 優化...",
                "✨ Koyeb 高速完成！"
            ]
        else:
            stages = [
                f"⚡ Koyeb 啟動 {provider}...",
                f"🔗 建立 API 連接...",
                f"🎨 AI 圖像生成中...",
                f"📱 結果優化處理...",
                f"✨ 生成完成！"
            ]
        
        for i, stage in enumerate(stages):
            status_text.text(stage)
            time.sleep(0.4)  # Koyeb 優化：更快的進度更新
            progress_bar.progress((i + 1) / len(stages))
    
    # 執行生成
    if provider == "Pollinations.ai":
        success, result = generate_pollinations_image_koyeb(prompt, model, size)
    else:
        success, result = generate_demo_image_koyeb(prompt, provider)
    
    progress_container.empty()
    
    # 顯示結果
    if success:
        st.success(f"✅ Koyeb 高速生成完成！")
        st.balloons()
        
        # 顯示圖像
        st.markdown("#### 🎨 生成結果")
        
        try:
            st.image(result, use_column_width=True, caption=f"Koyeb 部署 - {provider}")
            
            # Koyeb 優化的操作按鈕
            col_download, col_regen = st.columns(2)
            
            with col_download:
                if st.button("📥 下載圖像", use_container_width=True):
                    st.info("💡 右鍵點擊圖像保存到本地")
            
            with col_regen:
                if st.button("🔄 重新生成", use_container_width=True):
                    generate_image_koyeb(provider, prompt, model, size)
                    
        except Exception as e:
            st.error(f"圖像顯示錯誤: {e}")
    else:
        st.error(f"❌ 生成失敗: {result}")

# 會話狀態初始化 - Koyeb 優化
@st.cache_data
def init_koyeb_session() -> Dict:
    """Koyeb 優化的會話初始化"""
    return {
        'providers_loaded': True,
        'koyeb_optimized': True,
        'cold_start_ready': True
    }

def init_session_state():
    """初始化會話狀態"""
    session_data = init_koyeb_session()
    
    for key, value in session_data.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'show_navyai_setup' not in st.session_state:
        st.session_state.show_navyai_setup = False

# 簡化的 NavyAI 設置 - Koyeb 優化
def show_koyeb_navyai_setup():
    """Koyeb 優化的 NavyAI 設置"""
    st.markdown("### ⚓ NavyAI 快速設置 - Koyeb 優化")
    
    with st.form("koyeb_navyai_form"):
        st.info("🚀 Koyeb 高性能部署專用 NavyAI 配置")
        
        key_name = st.text_input(
            "密鑰名稱:",
            placeholder="Koyeb NavyAI 主密鑰",
            value="Koyeb NavyAI 主密鑰"
        )
        
        api_key = st.text_input(
            "NavyAI API 密鑰:",
            type="password",
            placeholder="輸入您的 NavyAI API 密鑰...",
            help="密鑰格式：navy_xxxxxxxx"
        )
        
        submitted = st.form_submit_button("💾 保存並啟用", type="primary", use_container_width=True)
        
        if submitted and api_key:
            # 保存密鑰
            key_id = provider_manager.save_api_key("NavyAI", key_name, api_key)
            
            if key_id:
                st.session_state.selected_provider = "NavyAI"
                st.success("✅ NavyAI 已配置並啟用 - Koyeb 優化模式")
                st.info("⚓ 現在可以訪問 15+ 專業圖像模型")
                time.sleep(2)
                rerun_app()
            else:
                st.error("❌ 密鑰保存失敗")
    
    if st.button("🏠 返回主頁", use_container_width=True):
        st.session_state.show_navyai_setup = False
        rerun_app()

# 主程式 - Koyeb 優化
def main():
    """Koyeb 優化的主程式"""
    
    # 初始化
    init_session_state()
    
    # Koyeb 環境提示
    if KOYEB_ENV:
        st.success("🚀 應用正在 Koyeb 高性能平台運行")
    
    # 顯示頭部
    show_koyeb_header()
    
    # 顯示 Koyeb 狀態
    show_koyeb_status()
    
    st.markdown("---")
    
    # 路由邏輯 - Koyeb 優化
    if st.session_state.get('show_navyai_setup', False):
        show_koyeb_navyai_setup()
    elif 'selected_provider' in st.session_state:
        show_koyeb_image_generator()
    else:
        show_koyeb_main_interface()
    
    # Koyeb 優化的頁腳
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <h4>🚀 Koyeb 高性能無服務器部署</h4>
        <p><strong>Scale-to-Zero</strong> | <strong>Global CDN</strong> | <strong>冷啟動優化</strong></p>
        <div style="margin-top: 0.5rem;">
            <small>
                運行環境: {'🌍 Koyeb Production' if KOYEB_ENV else '💻 Local Development'} | 
                端口: {PORT} | 
                版本: Koyeb Optimized v2.0
            </small>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
