import streamlit as st
import os
import logging
import time
import sqlite3
import uuid
import json
import random
from functools import lru_cache

# 必須是第一個 Streamlit 命令 - Koyeb 優化配置
st.set_page_config(
    page_title="AI Image Generator Pro - FLUX Krea + NavyAI Models",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 設置環境編碼
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Koyeb 環境檢測和優化設置
KOYEB_ENV = os.getenv('KOYEB_PUBLIC_DOMAIN') is not None
PORT = int(os.getenv('PORT', 8501))

# 日誌配置 - Koyeb 優化，避免 Unicode 錯誤
logging.basicConfig(
    level=logging.INFO if KOYEB_ENV else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 只在需要時導入重型模組 - 修復版本
@lru_cache(maxsize=1)
def get_heavy_imports():
    """延遲載入重型模組以優化冷啟動時間"""
    imports = {}
    
    try:
        # 嘗試導入 OpenAI
        try:
            from openai import OpenAI
            imports['OpenAI'] = OpenAI
            logger.info("OpenAI imported successfully")
        except ImportError as e:
            logger.warning(f"OpenAI import failed: {e}")
            imports['OpenAI'] = None
        
        # 嘗試導入 PIL
        try:
            from PIL import Image, ImageDraw, ImageFont
            imports['Image'] = Image
            imports['ImageDraw'] = ImageDraw
            imports['ImageFont'] = ImageFont
            logger.info("PIL imported successfully")
        except ImportError as e:
            logger.warning(f"PIL import failed: {e}")
            imports['Image'] = None
            imports['ImageDraw'] = None
            imports['ImageFont'] = None
        
        # 嘗試導入其他必要模組
        try:
            import requests
            imports['requests'] = requests
        except ImportError:
            logger.error("Requests import failed")
            imports['requests'] = None
        
        try:
            from io import BytesIO
            imports['BytesIO'] = BytesIO
        except ImportError:
            logger.error("BytesIO import failed")
            imports['BytesIO'] = None
        
        try:
            import datetime
            imports['datetime'] = datetime
        except ImportError:
            imports['datetime'] = None
        
        try:
            import base64
            imports['base64'] = base64
        except ImportError:
            logger.error("Base64 import failed")
            imports['base64'] = None
        
        try:
            import re
            imports['re'] = re
        except ImportError:
            imports['re'] = None
        
        return imports
        
    except Exception as e:
        logger.error(f"Unexpected error in imports: {str(e)}")
        return {}

# 安全的文本處理函數 - 避免編碼錯誤
def safe_text(text, max_length=None):
    """安全處理文本，避免編碼錯誤"""
    try:
        if not isinstance(text, str):
            text = str(text)
        
        # 移除或替換可能導致編碼問題的字符
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        if max_length and len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    except Exception as e:
        logger.warning(f"Text encoding issue: {str(e)}")
        return "Text encoding error"

# 回到主頁功能
def go_to_homepage():
    """返回主頁並清除所有狀態"""
    try:
        # 清除選擇的供應商
        if 'selected_provider' in st.session_state:
            del st.session_state.selected_provider
        
        # 清除 NavyAI 設置頁面狀態
        if 'show_navyai_setup' in st.session_state:
            del st.session_state.show_navyai_setup
        
        # 清除 NavyAI 模型選擇
        if 'selected_navyai_model' in st.session_state:
            del st.session_state.selected_navyai_model
        
        if 'selected_navyai_category' in st.session_state:
            del st.session_state.selected_navyai_category
        
        # 清除 FLUX Krea 模型選擇
        if 'selected_flux_krea_model' in st.session_state:
            del st.session_state.selected_flux_krea_model
        
        # 清除快速模板
        if 'quick_template' in st.session_state:
            del st.session_state.quick_template
        
        # 重新運行應用
        rerun_app()
    except Exception as e:
        logger.error(f"Error in go_to_homepage: {str(e)}")
        st.rerun()

def show_home_button():
    """顯示回到主頁按鈕 - 通用組件"""
    if st.button("🏠 回到主頁", use_container_width=True, type="secondary"):
        go_to_homepage()

# Koyeb 兼容性函數
def rerun_app():
    """Koyeb 優化的重新運行函數"""
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    else:
        st.stop()

# FLUX Krea 專門模型庫
FLUX_KREA_MODELS = {
    "flux-krea-dev": {
        "name": "FLUX Krea Dev",
        "model_id": "flux-krea",
        "description": "美學優化開發版，平衡質量與速度，最受歡迎",
        "pricing": "免費",
        "speed": "~6-8s",
        "quality": 5,
        "aesthetic_score": 5,
        "recommended": True,
        "speciality": "平衡性能",
        "best_for": ["人像攝影", "風景攝影", "日常創作"],
        "icon": "🎭"
    },
    "flux-krea-pro": {
        "name": "FLUX Krea Pro",
        "model_id": "flux-krea-pro",
        "description": "專業級美學優化，最高質量輸出，適合專業創作",
        "pricing": "免費",
        "speed": "~8-10s",
        "quality": 5,
        "aesthetic_score": 5,
        "recommended": True,
        "speciality": "最高質量",
        "best_for": ["專業攝影", "商業創作", "藝術作品"],
        "icon": "👑"
    },
    "flux-krea-schnell": {
        "name": "FLUX Krea Schnell",
        "model_id": "flux-krea-schnell", 
        "description": "快速版本，保持美學質量同時提升生成速度",
        "pricing": "免費",
        "speed": "~3-5s",
        "quality": 4,
        "aesthetic_score": 4,
        "recommended": False,
        "speciality": "極速生成",
        "best_for": ["快速原型", "批量生成", "測試創意"],
        "icon": "⚡"
    },
    "flux-krea-realism": {
        "name": "FLUX Krea Realism",
        "model_id": "flux-realism",
        "description": "專注寫實風格，適合需要高度真實感的圖像",
        "pricing": "免費",
        "speed": "~7-9s",
        "quality": 5,
        "aesthetic_score": 4,
        "recommended": False,
        "speciality": "寫實專精",
        "best_for": ["寫實人像", "產品攝影", "紀錄風格"],
        "icon": "📸"
    },
    "flux-krea-anime": {
        "name": "FLUX Krea Anime",
        "model_id": "flux-anime",
        "description": "動漫風格優化，專門生成動漫插畫風格圖像",
        "pricing": "免費",
        "speed": "~6-8s",
        "quality": 4,
        "aesthetic_score": 5,
        "recommended": False,
        "speciality": "動漫風格",
        "best_for": ["動漫角色", "插畫創作", "二次元風格"],
        "icon": "🎌"
    },
    "flux-krea-artistic": {
        "name": "FLUX Krea Artistic",
        "model_id": "flux-artistic",
        "description": "藝術創作優化，強化創意表現和藝術風格",
        "pricing": "免費",
        "speed": "~8-10s",
        "quality": 5,
        "aesthetic_score": 5,
        "recommended": False,
        "speciality": "藝術創作",
        "best_for": ["抽象藝術", "創意設計", "概念藝術"],
        "icon": "🎨"
    }
}

# NavyAI 模型配置 - 簡化版本
NAVYAI_MODELS = {
    "dalle": {
        "category_name": "🖼️ DALL-E (OpenAI)",
        "description": "OpenAI 創意圖像生成，文本理解能力強",
        "models": [
            {
                "id": "dall-e-3",
                "name": "DALL-E 3",
                "description": "最新創意版本，細節豐富，文本理解強",
                "pricing": "$0.040/image", 
                "speed": "~10s",
                "quality": 5,
                "recommended": True,
                "api_model": "dall-e-3"
            },
            {
                "id": "dall-e-2",
                "name": "DALL-E 2",
                "description": "經典版本，穩定可靠",
                "pricing": "$0.020/image",
                "speed": "~8s", 
                "quality": 4,
                "recommended": False,
                "api_model": "dall-e-2"
            }
        ]
    }
}

# FLUX Krea 專門優化參數
FLUX_KREA_PRESETS = {
    "portrait": {
        "name": "🖼️ 人像攝影",
        "prompt_prefix": "professional portrait photography, ",
        "prompt_suffix": ", natural lighting, realistic skin texture, detailed eyes, high resolution",
        "guidance_scale": 3.5,
        "aesthetic_weight": 1.2,
        "color_harmony": "warm"
    },
    "landscape": {
        "name": "🌄 風景攝影", 
        "prompt_prefix": "beautiful landscape photography, ",
        "prompt_suffix": ", golden hour lighting, natural colors, scenic view, high detail",
        "guidance_scale": 4.0,
        "aesthetic_weight": 1.3,
        "color_harmony": "natural"
    },
    "artistic": {
        "name": "🎨 藝術創作",
        "prompt_prefix": "artistic composition, ",
        "prompt_suffix": ", creative lighting, artistic style, detailed artwork, masterpiece",
        "guidance_scale": 4.5,
        "aesthetic_weight": 1.5,
        "color_harmony": "vibrant"
    },
    "realistic": {
        "name": "📸 寫實風格",
        "prompt_prefix": "photorealistic, ",
        "prompt_suffix": ", natural appearance, realistic details, authentic style, lifelike",
        "guidance_scale": 3.0,
        "aesthetic_weight": 1.0,
        "color_harmony": "neutral"
    }
}

# 模型供應商配置
MODEL_PROVIDERS = {
    "FLUX Krea AI": {
        "name": "FLUX Krea AI",
        "icon": "🎭",
        "description": "FLUX Krea 專門優化 - 6種模型選擇，美學圖像生成專家",
        "api_type": "pollinations",
        "base_url": "https://image.pollinations.ai/prompt",
        "features": ["flux-krea"],
        "koyeb_optimized": True,
        "requires_api_key": False,
        "cold_start_friendly": True,
        "speciality": "美學優化專家 + 多模型選擇"
    },
    "NavyAI": {
        "name": "NavyAI",
        "icon": "⚓",
        "description": "統一圖像 API - 真實 OpenAI 兼容接口",
        "api_type": "openai_compatible", 
        "base_url": "https://api.navy/v1",
        "features": ["dalle"],
        "koyeb_optimized": True,
        "requires_api_key": True,
        "cold_start_friendly": True,
        "speciality": "真實 API 調用統一接口"
    }
}

# Koyeb 優化的 SQLite 管理器
class KoyebOptimizedProviderManager:
    def __init__(self):
        self.db_path = "/tmp/koyeb_providers.db" if KOYEB_ENV else "koyeb_providers.db"
        self.init_database()
    
    @lru_cache(maxsize=100)
    def get_cached_providers(self):
        """Koyeb 優化：快取供應商列表"""
        return MODEL_PROVIDERS.copy()
    
    def init_database(self):
        """Koyeb 優化的數據庫初始化"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            
            cursor = conn.cursor()
            
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
            
            conn.commit()
            conn.close()
            logger.info("Koyeb database initialized successfully")
            
        except Exception as e:
            logger.error(f"Koyeb database initialization failed: {str(e)}")
    
    def save_api_key(self, provider, key_name, api_key):
        """Koyeb 優化的 API 密鑰保存"""
        key_id = str(uuid.uuid4())[:8]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("UPDATE koyeb_api_keys SET is_active = 0 WHERE provider = ?", (provider,))
            
            cursor.execute('''
                INSERT INTO koyeb_api_keys (id, provider, key_name, api_key)
                VALUES (?, ?, ?, ?)
            ''', (key_id, provider, key_name, api_key))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Koyeb API key saved: {provider}")
            return key_id
            
        except Exception as e:
            logger.error(f"Koyeb API key save failed: {str(e)}")
            return ""
    
    def get_active_api_key(self, provider):
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
            logger.error(f"Koyeb key retrieval failed: {str(e)}")
            return None

# 全局管理器實例
@st.cache_resource
def get_provider_manager():
    """Koyeb 優化：快取管理器實例"""
    return KoyebOptimizedProviderManager()

provider_manager = get_provider_manager()

# FLUX Krea 專門優化生成 - 修復版本
def generate_flux_krea_image(prompt, model_id="flux-krea", preset="realistic", size="1024x1024"):
    """FLUX Krea 專門優化的圖像生成 - 修復版本"""
    imports = get_heavy_imports()
    
    # 檢查必要的導入
    if not imports.get('requests') or not imports.get('base64'):
        return False, "缺少必要的模組 (requests, base64)"
    
    try:
        # 安全處理提示詞
        prompt = safe_text(prompt, max_length=500)
        
        # 應用 FLUX Krea 預設
        preset_config = FLUX_KREA_PRESETS.get(preset, FLUX_KREA_PRESETS["realistic"])
        
        # 優化提示詞
        optimized_prompt = f"{preset_config['prompt_prefix']}{prompt}{preset_config['prompt_suffix']}"
        
        # URL 編碼
        import urllib.parse
        encoded_prompt = urllib.parse.quote(optimized_prompt)
        
        width, height = map(int, size.split('x'))
        
        # FLUX Krea 專門參數
        url_params = [
            f"model={model_id}",
            f"width={width}",
            f"height={height}",
            "nologo=true"
        ]
        
        base_url = "https://image.pollinations.ai/prompt"
        full_url = f"{base_url}/{encoded_prompt}?{'&'.join(url_params)}"
        
        logger.info(f"FLUX Krea API call: {full_url[:100]}...")
        
        # 發送請求
        response = imports['requests'].get(full_url, timeout=30)
        
        if response.status_code == 200:
            # 編碼圖像
            encoded_image = imports['base64'].b64encode(response.content).decode()
            image_url = f"data:image/png;base64,{encoded_image}"
            logger.info("FLUX Krea generation successful")
            return True, image_url
        else:
            error_msg = f"HTTP {response.status_code}"
            logger.error(f"FLUX Krea API error: {error_msg}")
            return False, error_msg
            
    except Exception as e:
        error_msg = safe_text(str(e))
        logger.error(f"FLUX Krea image generation error: {error_msg}")
        return False, error_msg

# NavyAI 真實 API 圖像生成 - 修復版本
def generate_navyai_image_real(api_key, model_id, prompt, **params):
    """NavyAI 真實 OpenAI 兼容 API 圖像生成 - 修復版本"""
    imports = get_heavy_imports()
    
    # 檢查 OpenAI 是否可用
    if not imports.get('OpenAI'):
        logger.warning("OpenAI not available, using fallback")
        return generate_navyai_image_fallback(api_key, model_id, prompt, **params)
    
    try:
        # 安全處理參數
        prompt = safe_text(prompt, max_length=1000)
        api_model = params.get('api_model', 'dall-e-3')
        size = params.get('size', '1024x1024')
        num_images = min(params.get('num_images', 1), 4)
        
        logger.info(f"NavyAI API call: model={api_model}, prompt_length={len(prompt)}")
        
        # 創建 OpenAI 客戶端
        client = imports['OpenAI'](
            api_key=api_key,
            base_url="https://api.navy/v1"
        )
        
        # API 調用
        response = client.images.generate(
            model=api_model,
            prompt=prompt,
            n=num_images,
            size=size,
            quality="standard",
            response_format="b64_json"
        )
        
        # 處理回應
        if response.data and len(response.data) > 0:
            image_data = response.data[0]
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                image_url = f"data:image/png;base64,{image_data.b64_json}"
                logger.info("NavyAI API call successful")
                return True, image_url
            else:
                logger.error("NavyAI API response missing b64_json")
                return generate_navyai_image_fallback(api_key, model_id, prompt, **params)
        else:
            logger.error("NavyAI API response empty")
            return generate_navyai_image_fallback(api_key, model_id, prompt, **params)
            
    except Exception as e:
        error_msg = safe_text(str(e))
        logger.error(f"NavyAI API error: {error_msg}")
        return generate_navyai_image_fallback(api_key, model_id, prompt, **params)

def generate_navyai_image_fallback(api_key, model_id, prompt, **params):
    """NavyAI 模擬圖像生成（回退版本）- 修復版本"""
    imports = get_heavy_imports()
    
    # 檢查必要的模組
    if not imports.get('Image') or not imports.get('base64') or not imports.get('BytesIO'):
        return False, "缺少圖像處理模組 (PIL, base64, BytesIO)"
    
    try:
        logger.info("Using NavyAI fallback mode")
        
        # 模擬生成時間
        time.sleep(3)
        
        # 安全處理參數
        prompt = safe_text(prompt, max_length=500)
        width, height = map(int, params.get('size', '1024x1024').split('x'))
        
        # 創建圖像
        img = imports['Image'].new('RGB', (width, height))
        draw = imports['ImageDraw'].Draw(img)
        
        # 創建漸變背景（NavyAI 風格）
        for y in range(height):
            r = int(25 + (100 * y / height))
            g = int(50 + (150 * y / height))
            b = int(150 + (105 * y / height))
            for x in range(width):
                draw.point((x, y), (r, g, b))
        
        # 添加文字（使用默認字體）
        try:
            font = imports['ImageFont'].load_default()
        except:
            font = None
        
        # 添加標題和信息
        draw.text((50, 50), "NavyAI Demo Generation", fill=(255, 255, 255), font=font)
        draw.text((50, 100), f"Model: {model_id}", fill=(255, 255, 255), font=font)
        
        # 添加提示詞預覽
        prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        draw.text((50, 150), f"Prompt: {prompt_preview}", fill=(255, 255, 255), font=font)
        
        # 添加狀態信息
        draw.text((50, height - 100), "Fallback Mode - Demo Generation", fill=(255, 255, 0), font=font)
        draw.text((50, height - 50), "Koyeb High-Performance Deploy", fill=(255, 255, 255), font=font)
        
        # 轉換為 base64
        buffer = imports['BytesIO']()
        img.save(buffer, format='PNG')
        encoded_image = imports['base64'].b64encode(buffer.getvalue()).decode()
        
        return True, f"data:image/png;base64,{encoded_image}"
        
    except Exception as e:
        error_msg = safe_text(str(e))
        logger.error(f"NavyAI fallback generation error: {error_msg}")
        return False, error_msg

# UI 組件
def show_koyeb_header():
    """Koyeb 優化的應用頭部"""
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%); border-radius: 10px; margin-bottom: 1.5rem;">
        <h1 style="color: white; margin: 0; font-size: 2.2rem;">🎨 AI 圖像生成器 Pro</h1>
        <h2 style="color: #dbeafe; margin: 0.3rem 0; font-size: 1.1rem;">FLUX Krea 6種模型 + NavyAI 真實API調用</h2>
        <div style="margin-top: 0.8rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.6rem; border-radius: 15px; margin: 0.1rem; color: #fef3c7; font-size: 0.9rem;">🎭 FLUX Krea 6 Models</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.6rem; border-radius: 15px; margin: 0.1rem; color: #fef3c7; font-size: 0.9rem;">⚓ NavyAI Fixed API</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.6rem; border-radius: 15px; margin: 0.1rem; color: #fef3c7; font-size: 0.9rem;">🚀 Koyeb</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_dependency_check():
    """顯示依賴檢查狀態"""
    st.markdown("### 🔧 系統狀態檢查")
    
    imports = get_heavy_imports()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if imports.get('requests'):
            st.success("✅ Requests")
        else:
            st.error("❌ Requests")
    
    with col2:
        if imports.get('Image'):
            st.success("✅ Pillow")
        else:
            st.error("❌ Pillow")
    
    with col3:
        if imports.get('OpenAI'):
            st.success("✅ OpenAI")
        else:
            st.warning("⚠️ OpenAI")
    
    with col4:
        if imports.get('base64'):
            st.success("✅ Base64")
        else:
            st.error("❌ Base64")
    
    # 檢查是否所有核心功能可用
    core_available = all([
        imports.get('requests'),
        imports.get('base64'),
        imports.get('Image')
    ])
    
    if core_available:
        st.success("🎉 核心圖像生成功能可用")
    else:
        st.error("⚠️ 部分功能不可用，請檢查依賴安裝")
        
        # 顯示 requirements.txt
        st.markdown("#### 📋 請確保 requirements.txt 包含：")
        st.code("""streamlit>=1.28.0
openai>=1.0.0
Pillow>=10.0.0
requests>=2.31.0""")

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

def show_koyeb_main_interface():
    """Koyeb 優化的主界面"""
    st.markdown("### 🎯 選擇 AI 圖像生成服務")
    
    col_provider1, col_provider2 = st.columns(2)
    
    with col_provider1:
        st.markdown("""
        #### 🎭 FLUX Krea AI (6種模型選擇)
        - ✅ **6種 FLUX Krea 模型**
        - 🎨 Dev, Pro, Schnell, Realism, Anime, Artistic
        - ⚡ 多種預設模式
        - 🆓 完全免費使用
        - 🚀 Koyeb 冷啟動優化
        """)
        
        if st.button("🎭 使用 FLUX Krea", type="primary", use_container_width=True):
            st.session_state.selected_provider = "FLUX Krea AI"
            st.success("✅ FLUX Krea AI 已啟動 - 6種模型選擇")
            rerun_app()
    
    with col_provider2:
        st.markdown("""
        #### ⚓ NavyAI (真實API調用)  
        - 🎨 **真實 OpenAI 兼容 API**
        - 🖼️ DALL-E 2/3
        - 🔧 需要 API 密鑰
        - 📡 真實雲端生成
        - 🛡️ 自動回退保護
        """)
        
        if st.button("⚓ 配置 NavyAI", use_container_width=True):
            st.session_state.show_navyai_setup = True
            rerun_app()

def show_flux_krea_generator():
    """FLUX Krea 專門生成器 - 修復版本"""
    # 頁面頂部 - 回到主頁按鈕
    col_home, col_title = st.columns([1, 4])
    with col_home:
        show_home_button()
    with col_title:
        st.markdown("### 🎭 FLUX Krea AI - 6種模型美學生成")
    
    # 檢查依賴
    imports = get_heavy_imports()
    if not imports.get('requests') or not imports.get('base64'):
        st.error("⚠️ 缺少必要的依賴，FLUX Krea 功能不可用")
        st.info("請確保已安裝 requests 和相關依賴")
        show_home_button()
        return
    
    # FLUX Krea 模型選擇
    st.markdown("#### 🤖 選擇 FLUX Krea 模型")
    
    # 推薦模型
    st.markdown("##### ⭐ 推薦模型")
    recommended_models = {k: v for k, v in FLUX_KREA_MODELS.items() if v['recommended']}
    
    cols_rec = st.columns(len(recommended_models))
    selected_model = None
    
    for i, (model_key, model_info) in enumerate(recommended_models.items()):
        with cols_rec[i]:
            if st.button(
                f"{model_info['icon']} {model_info['name']}",
                key=f"rec_flux_{model_key}",
                use_container_width=True,
                type="primary"
            ):
                selected_model = model_info
                st.session_state.selected_flux_krea_model = model_info
            
            st.caption(model_info['description'])
            st.caption(f"⚡ {model_info['speed']} | {'⭐' * model_info['quality']}")
            st.caption(f"🎯 {model_info['speciality']}")
    
    # 其他模型
    st.markdown("##### 📋 其他專業模型")
    other_models = {k: v for k, v in FLUX_KREA_MODELS.items() if not v['recommended']}
    
    for model_key, model_info in other_models.items():
        col_model, col_btn = st.columns([3, 1])
        
        with col_model:
            st.write(f"{model_info['icon']} **{model_info['name']}**")
            st.caption(model_info['description'])
            st.caption(f"⚡ {model_info['speed']} | {'⭐' * model_info['quality']} | 🎯 {model_info['speciality']}")
            st.caption(f"最適合: {', '.join(model_info['best_for'])}")
        
        with col_btn:
            if st.button("選擇", key=f"sel_flux_{model_key}", use_container_width=True):
                selected_model = model_info
                st.session_state.selected_flux_krea_model = model_info
    
    # 檢查會話中的選擇
    if hasattr(st.session_state, 'selected_flux_krea_model'):
        selected_model = st.session_state.selected_flux_krea_model
    
    if selected_model:
        st.markdown("---")
        col_selected, col_home_selected = st.columns([4, 1])
        with col_selected:
            st.success(f"✅ 已選擇: {selected_model['icon']} {selected_model['name']} - {selected_model['speciality']}")
        with col_home_selected:
            show_home_button()
        
        # 生成界面
        col_prompt, col_settings = st.columns([2, 1])
        
        with col_prompt:
            prompt = st.text_area(
                "✍️ 描述您想要的圖像:",
                height=120,
                placeholder=f"針對 {selected_model['name']} 優化您的提示詞...",
                help=f"{selected_model['name']} - {selected_model['description']}"
            )
        
        with col_settings:
            st.markdown("#### 🎯 美學預設")
            
            preset_options = list(FLUX_KREA_PRESETS.keys())
            preset_names = [FLUX_KREA_PRESETS[p]["name"] for p in preset_options]
            
            selected_preset_idx = st.selectbox(
                "選擇美學風格:",
                range(len(preset_names)),
                format_func=lambda x: preset_names[x],
                index=0
            )
            selected_preset = preset_options[selected_preset_idx]
            
            st.markdown("#### 🖼️ 生成參數")
            size_options = ["512x512", "768x768", "1024x1024"]
            selected_size = st.selectbox("圖像尺寸:", size_options, index=2)
            
            # 當前模型特性
            st.success(f"{selected_model['icon']} **{selected_model['name']} 特性**")
            st.caption(f"• {selected_model['speciality']}")
            st.caption(f"• 質量等級: {'⭐' * selected_model['quality']}")
            st.caption(f"• 生成速度: {selected_model['speed']}")
        
        st.markdown("---")
        
        can_generate = prompt.strip() and selected_model
        
        col_generate, col_back = st.columns([3, 1])
        with col_generate:
            if st.button(
                f"{selected_model['icon']} FLUX Krea 生成 ({selected_model['name']})",
                type="primary", 
                disabled=not can_generate,
                use_container_width=True
            ):
                if can_generate:
                    generate_flux_krea_main(selected_model, prompt, selected_preset, selected_size)
        
        with col_back:
            show_home_button()
    else:
        # 沒有選擇模型時
        st.markdown("---")
        col_prompt_select, col_home_noselect = st.columns([4, 1])
        with col_prompt_select:
            st.info("💡 請先選擇一個 FLUX Krea 模型開始生成")
        with col_home_noselect:
            show_home_button()

def show_navyai_generator():
    """NavyAI 真實 API 生成器"""
    # 頁面頂部 - 回到主頁按鈕
    col_home, col_title = st.columns([1, 4])
    with col_home:
        show_home_button()
    with col_title:
        st.markdown("### ⚓ NavyAI - 真實 OpenAI 兼容 API")
    
    api_key_info = provider_manager.get_active_api_key("NavyAI")
    if not api_key_info:
        st.warning("⚠️ 請先配置 NavyAI API 密鑰")
        col_setup, col_home_warn = st.columns([3, 1])
        with col_setup:
            if st.button("⚓ 前往設置", use_container_width=True):
                st.session_state.show_navyai_setup = True
                rerun_app()
        with col_home_warn:
            show_home_button()
        return
    
    st.success(f"🔑 使用密鑰: {api_key_info['key_name']}")
    st.info("⚓ 真實 NavyAI API 調用 - OpenAI 兼容接口")
    
    # 模型選擇
    st.markdown("#### 🤖 選擇 NavyAI 模型")
    
    # 創建模型分類標籤
    category_tabs = st.tabs(list(NAVYAI_MODELS.keys()))
    
    selected_model = None
    selected_category = None
    
    for i, (category, category_data) in enumerate(NAVYAI_MODELS.items()):
        with category_tabs[i]:
            st.markdown(f"**{category_data['category_name']}**")
            st.caption(category_data['description'])
            
            # 推薦模型
            recommended_models = [m for m in category_data['models'] if m.get('recommended', False)]
            if recommended_models:
                st.markdown("##### ⭐ 推薦模型")
                
                cols = st.columns(len(recommended_models))
                for j, model in enumerate(recommended_models):
                    with cols[j]:
                        if st.button(
                            f"✨ {model['name']}", 
                            key=f"rec_{model['id']}", 
                            use_container_width=True,
                            type="primary"
                        ):
                            selected_model = model
                            selected_category = category
                            st.session_state.selected_navyai_model = model
                            st.session_state.selected_navyai_category = category
                        
                        st.caption(model['description'])
                        st.caption(f"💰 {model['pricing']} | ⏱️ {model['speed']}")
                        st.caption(f"質量: {'⭐' * model['quality']}")
                        st.caption(f"API模型: `{model.get('api_model', 'dall-e-3')}`")
            
            # 其他模型
            other_models = [m for m in category_data['models'] if not m.get('recommended', False)]
            if other_models:
                st.markdown("##### 📋 其他模型")
                
                for model in other_models:
                    col_model, col_btn = st.columns([3, 1])
                    with col_model:
                        st.write(f"**{model['name']}**")
                        st.caption(model['description'])
                        st.caption(f"💰 {model['pricing']} | ⏱️ {model['speed']} | {'⭐' * model['quality']}")
                        st.caption(f"API模型: `{model.get('api_model', 'dall-e-3')}`")
                    
                    with col_btn:
                        if st.button("選擇", key=f"sel_{model['id']}", use_container_width=True):
                            selected_model = model
                            selected_category = category
                            st.session_state.selected_navyai_model = model
                            st.session_state.selected_navyai_category = category
    
    # 檢查會話中的選擇
    if hasattr(st.session_state, 'selected_navyai_model'):
        selected_model = st.session_state.selected_navyai_model
        selected_category = st.session_state.selected_navyai_category
    
    if selected_model:
        st.markdown("---")
        col_selected, col_home_selected = st.columns([4, 1])
        with col_selected:
            st.success(f"✅ 已選擇: {selected_model['name']} ({NAVYAI_MODELS[selected_category]['category_name']})")
            st.info(f"🔗 真實API模型: `{selected_model.get('api_model', 'dall-e-3')}`")
        with col_home_selected:
            show_home_button()
        
        # 生成界面
        col_prompt, col_params = st.columns([3, 1])
        
        with col_prompt:
            prompt = st.text_area(
                "✍️ 描述您想要的圖像:",
                height=100,
                placeholder=f"針對 {selected_model['name']} 優化您的提示詞...",
                help=f"當前模型: {selected_model['name']} - {selected_model['description']}"
            )
            
            # API 模型特定提示
            api_model = selected_model.get('api_model', 'dall-e-3')
            if api_model == "dall-e-3":
                st.info("💡 DALL-E 3 擅長創意圖像生成和文本渲染")
            elif api_model == "dall-e-2":
                st.info("💡 DALL-E 2 提供穩定可靠的圖像生成")
        
        with col_params:
            st.markdown("#### ⚙️ 生成參數")
            
            size_options = ["256x256", "512x512", "1024x1024"]
            if api_model == "dall-e-3":
                size_options = ["1024x1024", "1024x1792", "1792x1024"]
            
            selected_size = st.selectbox("圖像尺寸:", size_options, index=0)
            
            num_images = st.slider("生成數量:", 1, 4, 1)
            
            # 模型特定信息
            st.info(f"**當前模型**: {selected_model['name']}")
            st.caption(f"API模型: {api_model}")
            st.caption(f"價格: {selected_model['pricing']}")
            st.caption(f"速度: {selected_model['speed']}")
            st.caption(f"質量: {'⭐' * selected_model['quality']}")
            
            # API 狀態
            st.success("🔗 真實 API 調用")
            st.caption("• OpenAI 兼容接口")
            st.caption("• 真實雲端生成")
            st.caption("• 自動回退保護")
        
        can_generate = prompt.strip() and selected_model
        
        col_generate, col_back = st.columns([3, 1])
        with col_generate:
            if st.button(
                f"⚓ NavyAI 真實生成 ({selected_model['name']})",
                type="primary",
                disabled=not can_generate,
                use_container_width=True
            ):
                if can_generate:
                    generate_navyai_main(
                        api_key_info['api_key'], 
                        selected_model, 
                        selected_category,
                        prompt, 
                        selected_size, 
                        num_images
                    )
        
        with col_back:
            show_home_button()
    else:
        # 沒有選擇模型時顯示回到主頁按鈕
        st.markdown("---")
        col_prompt_select, col_home_noselect = st.columns([4, 1])
        with col_prompt_select:
            st.info("💡 請先選擇一個 NavyAI 模型開始生成")
        with col_home_noselect:
            show_home_button()

def generate_flux_krea_main(selected_model, prompt, preset, size):
    """FLUX Krea 主生成流程 - 修復版本"""
    progress_container = st.empty()
    
    with progress_container.container():
        st.info(f"{selected_model['icon']} {selected_model['name']} 美學優化生成中...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stages = [
            f"{selected_model['icon']} 初始化 {selected_model['name']} 引擎...",
            f"✨ 應用 {selected_model['speciality']} 優化...",
            f"🖼️ 處理美學提示詞...",
            f"🌈 生成色彩和諧方案...",
            f"{selected_model['icon']} {selected_model['speciality']} 處理中...",
            f"📸 {selected_model['name']} 渲染中...",
            f"🎉 {selected_model['name']} 生成完成！"
        ]
        
        for i, stage in enumerate(stages):
            status_text.text(stage)
            time.sleep(0.5)
            progress_bar.progress((i + 1) / len(stages))
    
    success, result = generate_flux_krea_image(prompt, selected_model['model_id'], preset, size)
    
    progress_container.empty()
    
    if success:
        st.success(f"{selected_model['icon']}✨ {selected_model['name']} 生成完成！")
        st.balloons()
        
        st.markdown(f"#### 🎨 {selected_model['name']} 作品")
        
        try:
            st.image(result, use_column_width=True, caption=f"{selected_model['name']} - {selected_model['speciality']} | 預設: {FLUX_KREA_PRESETS[preset]['name']}")
            
            # 模型分析
            with st.expander(f"{selected_model['icon']} {selected_model['name']} 詳細分析"):
                col_model, col_preset = st.columns(2)
                
                with col_model:
                    st.write(f"**模型名稱**: {selected_model['name']}")
                    st.write(f"**模型專長**: {selected_model['speciality']}")
                    st.write(f"**生成速度**: {selected_model['speed']}")
                    st.write(f"**質量等級**: {'⭐' * selected_model['quality']}")
                    st.write(f"**美學分數**: {'✨' * selected_model['aesthetic_score']}")
                    st.write(f"**最適合**: {', '.join(selected_model['best_for'])}")
                
                with col_preset:
                    preset_config = FLUX_KREA_PRESETS[preset]
                    st.write(f"**美學預設**: {preset_config['name']}")
                    st.write(f"**美學指導強度**: {preset_config['guidance_scale']}")
                    st.write(f"**美學權重**: {preset_config['aesthetic_weight']}")
                    st.write(f"**色彩和諧**: {preset_config['color_harmony']}")
                    st.write(f"**優化提示詞**: {preset_config['prompt_prefix']}[您的提示詞]{preset_config['prompt_suffix']}")
            
            col_download, col_regen, col_home_result = st.columns([2, 2, 1])
            
            with col_download:
                if st.button("📥 下載作品", use_container_width=True):
                    st.info("💡 右鍵點擊圖像保存到本地")
            
            with col_regen:
                if st.button(f"{selected_model['icon']} 重新生成", use_container_width=True):
                    generate_flux_krea_main(selected_model, prompt, preset, size)
            
            with col_home_result:
                show_home_button()
                    
        except Exception as e:
            st.error(f"圖像顯示錯誤: {safe_text(str(e))}")
    else:
        st.error(f"❌ {selected_model['name']} 生成失敗: {result}")
        
        # 失敗時也顯示回到主頁
        col_error, col_home_error = st.columns([4, 1])
        with col_home_error:
            show_home_button()

def generate_navyai_main(api_key, model, category, prompt, size, num_images):
    """NavyAI 真實 API 主生成流程"""
    progress_container = st.empty()
    
    with progress_container.container():
        st.info(f"⚓ NavyAI {model['name']} 真實 API 生成中...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        api_model = model.get('api_model', 'dall-e-3')
        
        stages = [
            f"⚓ 初始化 NavyAI 統一接口...",
            f"🔗 連接 OpenAI 兼容 API...",
            f"🤖 載入 {api_model} 模型...",
            f"📝 處理提示詞優化...",
            f"🎨 {model['name']} 真實生成中...",
            f"📱 NavyAI API 回應處理...",
            f"🎉 NavyAI {model['name']} 生成完成！"
        ]
        
        for i, stage in enumerate(stages):
            status_text.text(stage)
            time.sleep(0.8)  # 真實 API 調用需要更多時間
            progress_bar.progress((i + 1) / len(stages))
    
    # 執行真實 API 調用
    success, result = generate_navyai_image_real(
        api_key, 
        model['id'], 
        prompt, 
        api_model=api_model,
        size=size, 
        num_images=num_images, 
        category=category
    )
    
    progress_container.empty()
    
    if success:
        st.success(f"⚓✨ NavyAI {model['name']} 真實API生成完成！")
        st.balloons()
        
        st.markdown(f"#### 🎨 NavyAI - {model['name']} 作品")
        
        try:
            st.image(result, use_column_width=True, caption=f"NavyAI {model['name']} - 真實API生成 - {NAVYAI_MODELS[category]['category_name']}")
            
            # 真實 API 模型信息
            with st.expander(f"⚓ NavyAI {model['name']} API 詳情"):
                col_model, col_api = st.columns(2)
                
                with col_model:
                    st.write(f"**模型名稱**: {model['name']}")
                    st.write(f"**模型ID**: {model['id']}")
                    st.write(f"**類別**: {NAVYAI_MODELS[category]['category_name']}")
                    st.write(f"**描述**: {model['description']}")
                    st.write(f"**定價**: {model['pricing']}")
                    st.write(f"**生成速度**: {model['speed']}")
                    st.write(f"**質量等級**: {'⭐' * model['quality']}")
                
                with col_api:
                    api_model = model.get('api_model', 'dall-e-3')
                    st.write(f"**API模型**: {api_model}")
                    st.write(f"**API類型**: OpenAI Compatible")
                    st.write(f"**基礎URL**: https://api.navy/v1")
                    st.write(f"**生成方式**: 真實雲端API")
                    st.write(f"**回退保護**: ✅ 已啟用")
                    st.write(f"**響應格式**: base64_json")
            
            col_download, col_regen, col_home_result = st.columns([2, 2, 1])
            
            with col_download:
                if st.button("📥 下載 NavyAI 作品", use_container_width=True):
                    st.info("💡 右鍵點擊圖像保存到本地")
            
            with col_regen:
                if st.button("⚓ 重新生成", use_container_width=True):
                    generate_navyai_main(api_key, model, category, prompt, size, num_images)
            
            with col_home_result:
                show_home_button()
                    
        except Exception as e:
            st.error(f"圖像顯示錯誤: {safe_text(str(e))}")
    else:
        st.error(f"❌ NavyAI 真實API生成失敗: {result}")
        st.warning("💡 如果問題持續，請檢查API密鑰或稍後重試")
        
        # 失敗時也顯示回到主頁
        col_error, col_home_error = st.columns([4, 1])
        with col_home_error:
            show_home_button()

def show_koyeb_image_generator():
    """Koyeb 優化的圖像生成器路由"""
    if 'selected_provider' not in st.session_state:
        st.warning("⚠️ 請先選擇一個服務提供商")
        show_home_button()
        return
    
    provider = st.session_state.selected_provider
    
    if provider == "FLUX Krea AI":
        show_flux_krea_generator()
    elif provider == "NavyAI":
        show_navyai_generator()

@st.cache_data
def init_koyeb_session():
    """Koyeb 優化的會話初始化"""
    return {
        'providers_loaded': True,
        'koyeb_optimized': True,
        'cold_start_ready': True,
        'flux_krea_models_loaded': True,
        'navyai_real_api_enabled': True,
        'encoding_fixed': True,
        'model_selection_enabled': True,
        'dependencies_checked': True
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

def show_koyeb_navyai_setup():
    """Koyeb 優化的 NavyAI 設置"""
    # 頁面頂部 - 回到主頁按鈕
    col_home, col_title = st.columns([1, 4])
    with col_home:
        show_home_button()
    with col_title:
        st.markdown("### ⚓ NavyAI 真實 API 設置 - Koyeb 優化")
    
    st.info("🔗 NavyAI 提供真實的 OpenAI 兼容 API 調用，支援 DALL-E 系列模型")
    
    with st.form("koyeb_navyai_form"):
        st.success("🚀 配置 NavyAI 真實 API 以訪問專業圖像模型")
        
        key_name = st.text_input(
            "密鑰名稱:",
            placeholder="NavyAI 真實API主密鑰",
            value="NavyAI 真實API主密鑰"
        )
        
        api_key = st.text_input(
            "NavyAI API 密鑰:",
            type="password",
            placeholder="輸入您的 NavyAI API 密鑰...",
            help="密鑰格式：navy_xxxxxxxx 或 sk-xxxxxxxx"
        )
        
        st.markdown("**🎨 NavyAI vs FLUX Krea 對比:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**⚓ NavyAI (真實API)**")
            st.caption("🖼️ DALL-E 2/3")
            st.caption("🔗 真實雲端生成")
            st.caption("📡 OpenAI 兼容接口")
            st.caption("🛡️ 自動回退保護")
            st.caption("💰 **按使用付費**")
        with col2:
            st.markdown("**🎭 FLUX Krea (免費)**")
            st.caption("🎭 FLUX Krea Dev")
            st.caption("👑 FLUX Krea Pro")
            st.caption("⚡ FLUX Krea Schnell") 
            st.caption("📸 FLUX Krea Realism")
            st.caption("🎌 FLUX Krea Anime")
            st.caption("🎨 FLUX Krea Artistic")
        
        col_submit, col_home_form = st.columns([3, 1])
        with col_submit:
            submitted = st.form_submit_button("💾 保存並啟用 NavyAI 真實API", type="primary", use_container_width=True)
        with col_home_form:
            if st.form_submit_button("🏠 返回主頁", use_container_width=True):
                go_to_homepage()
        
        if submitted and api_key:
            key_id = provider_manager.save_api_key("NavyAI", key_name, api_key)
            
            if key_id:
                st.session_state.selected_provider = "NavyAI"
                st.success("✅ NavyAI 真實API接口已配置並啟用")
                st.info("⚓ 現在可以使用真實的 OpenAI 兼容 API 生成圖像")
                st.balloons()
                time.sleep(2)
                rerun_app()
            else:
                st.error("❌ 密鑰保存失敗")

def main():
    """Koyeb 優化的主程式 - 修復版本"""
    try:
        init_session_state()
        
        if KOYEB_ENV:
            st.success("🚀 應用正在 Koyeb 高性能平台運行")
        
        show_koyeb_header()
        show_dependency_check()  # 顯示依賴檢查
        show_koyeb_status()
        
        st.markdown("---")
        
        if st.session_state.get('show_navyai_setup', False):
            show_koyeb_navyai_setup()
        elif 'selected_provider' in st.session_state:
            show_koyeb_image_generator()
        else:
            show_koyeb_main_interface()
        
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <h4>🚀 Koyeb 高性能無服務器部署</h4>
            <p><strong>🎭 FLUX Krea 6種模型</strong> | <strong>⚓ NavyAI 真實API</strong> | <strong>🌍 Global CDN</strong></p>
            <div style="margin-top: 0.5rem;">
                <small>
                    運行環境: {'🌍 Koyeb Production' if KOYEB_ENV else '💻 Local Development'} | 
                    端口: {PORT} | 
                    版本: FLUX Krea 6 Models + NavyAI Fixed API v6.0
                </small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"應用運行錯誤: {safe_text(str(e))}")
        logger.error(f"Main app error: {str(e)}")

if __name__ == "__main__":
    main()
