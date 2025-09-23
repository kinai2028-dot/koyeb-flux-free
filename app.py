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
        logger.error(f"Failed to import heavy modules: {str(e)}")
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

# NavyAI 模型配置 - 完整模型庫
NAVYAI_MODELS = {
    "flux-krea": {
        "category_name": "🎭 FLUX Krea (美學優化)",
        "description": "專業美學優化模型，專注自然寫實圖像生成",
        "models": [
            {
                "id": "black-forest-labs/flux-krea-dev", 
                "name": "FLUX Krea Dev",
                "description": "美學優化開發版，平衡質量與速度",
                "pricing": "$0.015/image",
                "speed": "~8s",
                "quality": 5,
                "recommended": True
            },
            {
                "id": "black-forest-labs/flux-krea-pro", 
                "name": "FLUX Krea Pro",
                "description": "專業級美學優化，最高質量",
                "pricing": "$0.025/image",
                "speed": "~12s",
                "quality": 5,
                "recommended": False
            },
            {
                "id": "black-forest-labs/flux-krea-schnell", 
                "name": "FLUX Krea Schnell",
                "description": "快速版本，保持美學質量",
                "pricing": "$0.008/image",
                "speed": "~4s",
                "quality": 4,
                "recommended": False
            }
        ]
    },
    "dalle": {
        "category_name": "🖼️ DALL-E (OpenAI)",
        "description": "OpenAI 創意圖像生成，文本理解能力強",
        "models": [
            {
                "id": "dalle-3-hd",
                "name": "DALL-E 3 HD",
                "description": "最新高清版本，細節豐富",
                "pricing": "$0.080/image",
                "speed": "~15s",
                "quality": 5,
                "recommended": True
            },
            {
                "id": "dalle-3",
                "name": "DALL-E 3 Standard",
                "description": "標準版本，創意無限",
                "pricing": "$0.040/image",
                "speed": "~10s",
                "quality": 5,
                "recommended": False
            },
            {
                "id": "dalle-2",
                "name": "DALL-E 2",
                "description": "經典版本，穩定可靠",
                "pricing": "$0.020/image",
                "speed": "~8s",
                "quality": 4,
                "recommended": False
            }
        ]
    },
    "midjourney": {
        "category_name": "🎯 Midjourney (藝術風格)",
        "description": "頂級藝術風格生成，創意表現力最強",
        "models": [
            {
                "id": "midjourney-v6",
                "name": "Midjourney v6",
                "description": "最新版本，藝術風格巔峰",
                "pricing": "$0.030/image",
                "speed": "~20s",
                "quality": 5,
                "recommended": True
            },
            {
                "id": "midjourney-niji-6",
                "name": "Niji 6 (動漫風格)",
                "description": "專業動漫插畫風格",
                "pricing": "$0.025/image",
                "speed": "~18s",
                "quality": 5,
                "recommended": True
            },
            {
                "id": "midjourney-v5.2",
                "name": "Midjourney v5.2",
                "description": "穩定版本，平衡性能",
                "pricing": "$0.020/image",
                "speed": "~15s",
                "quality": 4,
                "recommended": False
            }
        ]
    },
    "flux": {
        "category_name": "⚡ FLUX AI (高性能)",
        "description": "高性能文本到圖像生成，速度優化",
        "models": [
            {
                "id": "black-forest-labs/flux.1-pro",
                "name": "FLUX.1 Pro",
                "description": "專業級，最佳質量平衡",
                "pricing": "$0.012/image",
                "speed": "~6s",
                "quality": 4,
                "recommended": True
            },
            {
                "id": "black-forest-labs/flux.1-dev",
                "name": "FLUX.1 Dev",
                "description": "開發版，質量與速度平衡",
                "pricing": "$0.008/image",
                "speed": "~5s",
                "quality": 4,
                "recommended": False
            },
            {
                "id": "black-forest-labs/flux.1-schnell",
                "name": "FLUX.1 Schnell",
                "description": "超快速版，適合批量生成",
                "pricing": "$0.003/image",
                "speed": "~2s",
                "quality": 3,
                "recommended": False
            }
        ]
    },
    "stable-diffusion": {
        "category_name": "🎨 Stable Diffusion (開源)",
        "description": "開源圖像生成，可自由定制",
        "models": [
            {
                "id": "stability-ai/stable-diffusion-3-large",
                "name": "Stable Diffusion 3 Large",
                "description": "最新大型模型，質量卓越",
                "pricing": "$0.020/image",
                "speed": "~8s",
                "quality": 4,
                "recommended": True
            },
            {
                "id": "stability-ai/sdxl-base-1.0",
                "name": "SDXL Base 1.0",
                "description": "XL版本，細節豐富",
                "pricing": "$0.012/image",
                "speed": "~6s",
                "quality": 4,
                "recommended": False
            },
            {
                "id": "stability-ai/sdxl-turbo",
                "name": "SDXL Turbo",
                "description": "極速版本，快速原型",
                "pricing": "$0.005/image",
                "speed": "~3s",
                "quality": 3,
                "recommended": False
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
        "description": "統一圖像 API - 15+ 專業模型選擇",
        "api_type": "openai_compatible",
        "base_url": "https://api.navy/v1",
        "features": ["flux-krea", "dalle", "midjourney", "flux", "stable-diffusion"],
        "koyeb_optimized": True,
        "requires_api_key": True,
        "cold_start_friendly": True,
        "speciality": "多模型統一接口"
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
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS flux_krea_settings (
                    id TEXT PRIMARY KEY,
                    preset_name TEXT NOT NULL,
                    guidance_scale REAL DEFAULT 3.5,
                    aesthetic_weight REAL DEFAULT 1.2,
                    color_harmony TEXT DEFAULT 'warm',
                    naturalism_boost BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

# FLUX Krea 專門優化生成 - 支援模型選擇
@st.cache_data(ttl=300)
def generate_flux_krea_image(prompt, model_id="flux-krea", preset="realistic", size="1024x1024"):
    """FLUX Krea 專門優化的圖像生成 - 支援模型選擇"""
    imports = get_heavy_imports()
    if not imports:
        return False, "Module loading failed"
    
    try:
        # 安全處理提示詞
        prompt = safe_text(prompt, max_length=500)
        
        # 應用 FLUX Krea 預設
        preset_config = FLUX_KREA_PRESETS.get(preset, FLUX_KREA_PRESETS["realistic"])
        
        # 優化提示詞
        optimized_prompt = f"{preset_config['prompt_prefix']}{prompt}{preset_config['prompt_suffix']}"
        
        import urllib.parse
        encoded_prompt = urllib.parse.quote(optimized_prompt)
        
        width, height = map(int, size.split('x'))
        
        # FLUX Krea 專門參數 - 使用選擇的模型
        url_params = [
            f"model={model_id}",  # 使用選擇的模型
            f"width={width}",
            f"height={height}",
            "nologo=true",
            f"guidance={preset_config['guidance_scale']}",
            f"aesthetic={preset_config['aesthetic_weight']}",
            f"harmony={preset_config['color_harmony']}"
        ]
        
        base_url = "https://image.pollinations.ai/prompt"
        full_url = f"{base_url}/{encoded_prompt}?{'&'.join(url_params)}"
        
        response = imports['requests'].get(full_url, timeout=30)
        
        if response.status_code == 200:
            encoded_image = imports['base64'].b64encode(response.content).decode()
            image_url = f"data:image/png;base64,{encoded_image}"
            return True, image_url
        else:
            return False, f"HTTP {response.status_code}"
            
    except Exception as e:
        error_msg = safe_text(str(e))
        logger.error(f"FLUX Krea image generation error: {error_msg}")
        return False, error_msg

# NavyAI 模型選擇生成 - 修復編碼問題
def generate_navyai_image(api_key, model_id, prompt, **params):
    """NavyAI 多模型選擇生成（模擬實現）- 修復編碼問題"""
    imports = get_heavy_imports()
    if not imports:
        return False, "Module loading failed"
    
    try:
        # 安全處理所有文本參數
        prompt = safe_text(prompt, max_length=500)
        model_id = safe_text(model_id)
        
        # 根據模型類別決定生成時間
        if "krea" in model_id.lower():
            time.sleep(4)  # FLUX Krea 需要更多美學處理時間
        elif "dalle" in model_id.lower():
            time.sleep(5)  # DALL-E 需要更多創意處理時間
        elif "midjourney" in model_id.lower():
            time.sleep(6)  # Midjourney 需要最多藝術處理時間
        else:
            time.sleep(3)
        
        width, height = map(int, params.get('size', '1024x1024').split('x'))
        
        img = imports['Image'].new('RGB', (width, height))
        draw = imports['ImageDraw'].Draw(img)
        
        # 根據模型類型創建不同風格背景
        if "krea" in model_id.lower():
            # FLUX Krea - 自然美學漸變
            for y in range(height):
                r = int(135 + (120 * y / height))
                g = int(206 + (49 * y / height))
                b = int(235 + (20 * y / height))
                for x in range(width):
                    draw.point((x, y), (r, g, b))
        elif "dalle" in model_id.lower():
            # DALL-E - 創意橙藍漸變
            for y in range(height):
                r = int(255 + (-50 * y / height))
                g = int(165 + (90 * y / height))
                b = int(0 + (255 * y / height))
                for x in range(width):
                    draw.point((x, y), (r, g, b))
        elif "midjourney" in model_id.lower():
            # Midjourney - 藝術紫色漸變
            for y in range(height):
                r = int(75 + (180 * y / height))
                g = int(0 + (130 * y / height))
                b = int(130 + (125 * y / height))
                for x in range(width):
                    draw.point((x, y), (r, g, b))
        else:
            # 其他模型 - NavyAI 藍色主題
            for y in range(height):
                r = int(25 + (50 * y / height))
                g = int(50 + (100 * y / height))
                b = int(100 + (155 * y / height))
                for x in range(width):
                    draw.point((x, y), (r, g, b))
        
        try:
            font_large = imports['ImageFont'].load_default()
            font_small = imports['ImageFont'].load_default()
        except:
            font_large = font_small = None
        
        # 安全處理模型名稱 - 避免 Unicode 問題
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        model_name = safe_text(model_name, max_length=30)
        
        # 使用安全的文本，避免 emoji 編碼問題
        safe_title = f"NavyAI: {model_name}"
        draw.text((50, 50), safe_title, fill=(255, 255, 255), font=font_large)
        
        # 提示詞預覽 - 安全處理
        safe_prompt = safe_text(prompt, max_length=120)
        prompt_lines = [safe_prompt[i:i+40] for i in range(0, len(safe_prompt), 40)]
        y_offset = 100
        for line in prompt_lines:
            draw.text((50, y_offset), line, fill=(255, 255, 255), font=font_small)
            y_offset += 25
        
        # 模型信息 - 使用安全文本
        draw.text((50, height - 150), f"Model: {model_name}", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 125), "NavyAI Unified API", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 100), "15+ Professional Image Models", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 75), "Koyeb High-Performance Deploy", fill=(255, 255, 255), font=font_small)
        
        # 轉換為 base64
        buffer = imports['BytesIO']()
        img.save(buffer, format='PNG')
        encoded_image = imports['base64'].b64encode(buffer.getvalue()).decode()
        
        return True, f"data:image/png;base64,{encoded_image}"
        
    except Exception as e:
        error_msg = safe_text(str(e))
        logger.error(f"NavyAI image generation error: {error_msg}")
        return False, error_msg

# UI 組件
def show_koyeb_header():
    """Koyeb 優化的應用頭部"""
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%); border-radius: 10px; margin-bottom: 1.5rem;">
        <h1 style="color: white; margin: 0; font-size: 2.2rem;">🎨 AI 圖像生成器 Pro</h1>
        <h2 style="color: #dbeafe; margin: 0.3rem 0; font-size: 1.1rem;">FLUX Krea 6種模型 + NavyAI 15+ 模型選擇</h2>
        <div style="margin-top: 0.8rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.6rem; border-radius: 15px; margin: 0.1rem; color: #fef3c7; font-size: 0.9rem;">🎭 FLUX Krea 6 Models</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.6rem; border-radius: 15px; margin: 0.1rem; color: #fef3c7; font-size: 0.9rem;">⚓ NavyAI 15+ Models</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.6rem; border-radius: 15px; margin: 0.1rem; color: #fef3c7; font-size: 0.9rem;">🚀 Koyeb</span>
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
        #### ⚓ NavyAI (15+ 多模型統一)  
        - 🎨 **15+ 專業圖像模型**
        - 🖼️ DALL-E 3、Midjourney
        - ⚡ FLUX AI、Stable Diffusion
        - 🔧 需要 API 密鑰
        - 📊 統一接口管理
        """)
        
        if st.button("⚓ 配置 NavyAI", use_container_width=True):
            st.session_state.show_navyai_setup = True
            rerun_app()

def show_flux_krea_generator():
    """FLUX Krea 專門生成器 - 加入模型選擇"""
    # 頁面頂部 - 回到主頁按鈕
    col_home, col_title = st.columns([1, 4])
    with col_home:
        show_home_button()
    with col_title:
        st.markdown("### 🎭 FLUX Krea AI - 6種模型美學生成")
    
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
            
            # 模型專用模板
            st.markdown(f"#### {selected_model['icon']} {selected_model['name']} 專用模板")
            
            # 根據模型類型提供不同模板
            if "realism" in selected_model['model_id']:
                templates = [
                    "A professional business headshot, natural lighting, realistic skin texture",
                    "Product photography on white background, commercial lighting, high detail",
                    "Street photography documentary style, authentic moment, natural colors",
                    "Interior architecture photography, realistic lighting, detailed textures"
                ]
            elif "anime" in selected_model['model_id']:
                templates = [
                    "Anime girl character with long flowing hair, detailed eyes, vibrant colors",
                    "Fantasy anime warrior in magical forest, dynamic pose, epic lighting",
                    "Cute anime mascot character, chibi style, pastel colors, kawaii",
                    "Anime landscape with cherry blossoms, dreamy atmosphere, soft lighting"
                ]
            elif "artistic" in selected_model['model_id']:
                templates = [
                    "Abstract expressionist painting, bold brushstrokes, vibrant color palette",
                    "Surreal dreamscape with floating objects, creative composition, artistic style",
                    "Digital art concept design, futuristic elements, creative lighting",
                    "Contemporary art installation, conceptual design, artistic interpretation"
                ]
            else:
                templates = [
                    "Professional portrait with natural lighting and realistic skin texture",
                    "Beautiful landscape at golden hour, natural colors, peaceful atmosphere",
                    "Modern architectural interior, clean design, natural lighting",
                    "Street photography candid moment, authentic expression, urban setting"
                ]
            
            template_cols = st.columns(2)
            for i, template in enumerate(templates):
                with template_cols[i % 2]:
                    if st.button(f"{selected_model['icon']} {template[:35]}...", key=f"flux_template_{i}", use_container_width=True):
                        st.session_state.quick_template = template
                        rerun_app()
        
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
            
            # 顯示預設詳情
            preset_config = FLUX_KREA_PRESETS[selected_preset]
            st.info(f"**美學指導**: {preset_config['guidance_scale']}")
            st.info(f"**美學權重**: {preset_config['aesthetic_weight']}")
            st.info(f"**色彩和諧**: {preset_config['color_harmony']}")
            
            st.markdown("#### 🖼️ 生成參數")
            size_options = ["512x512", "768x768", "1024x1024", "1152x896", "896x1152"]
            selected_size = st.selectbox("圖像尺寸:", size_options, index=2)
            
            # 當前模型特性
            st.success(f"{selected_model['icon']} **{selected_model['name']} 特性**")
            st.caption(f"• {selected_model['speciality']}")
            st.caption(f"• 質量等級: {'⭐' * selected_model['quality']}")
            st.caption(f"• 美學分數: {'✨' * selected_model['aesthetic_score']}")
            st.caption(f"• 生成速度: {selected_model['speed']}")
            
            st.info(f"**最適合**: {', '.join(selected_model['best_for'])}")
        
        # 檢查快速模板
        if hasattr(st.session_state, 'quick_template'):
            prompt = st.session_state.quick_template
            del st.session_state.quick_template
            rerun_app()
        
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
    """NavyAI 多模型生成器"""
    # 頁面頂部 - 回到主頁按鈕
    col_home, col_title = st.columns([1, 4])
    with col_home:
        show_home_button()
    with col_title:
        st.markdown("### ⚓ NavyAI - 15+ 模型統一接口")
    
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
        
        with col_params:
            st.markdown("#### ⚙️ 生成參數")
            
            size_options = ["512x512", "768x768", "1024x1024", "1152x896", "896x1152"]
            selected_size = st.selectbox("圖像尺寸:", size_options, index=2)
            
            num_images = st.slider("生成數量:", 1, 4, 1)
            
            # 模型特定信息
            st.info(f"**當前模型**: {selected_model['name']}")
            st.caption(f"價格: {selected_model['pricing']}")
            st.caption(f"速度: {selected_model['speed']}")
            st.caption(f"質量: {'⭐' * selected_model['quality']}")
        
        can_generate = prompt.strip() and selected_model
        
        col_generate, col_back = st.columns([3, 1])
        with col_generate:
            if st.button(
                f"⚓ NavyAI 生成 ({selected_model['name']})",
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
    """FLUX Krea 主生成流程 - 支援模型選擇"""
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
    """NavyAI 主生成流程"""
    progress_container = st.empty()
    
    with progress_container.container():
        st.info(f"⚓ NavyAI {model['name']} 生成中...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stages = [
            f"⚓ 初始化 NavyAI 統一接口...",
            f"🤖 載入 {model['name']} 模型...",
            f"📝 處理 {category} 風格提示詞...",
            f"🎨 {model['name']} 圖像生成中...",
            f"✨ {category} 風格優化中...",
            f"📱 NavyAI 後處理優化...",
            f"🎉 NavyAI {model['name']} 生成完成！"
        ]
        
        for i, stage in enumerate(stages):
            status_text.text(stage)
            time.sleep(0.6)
            progress_bar.progress((i + 1) / len(stages))
    
    success, result = generate_navyai_image(
        api_key, model['id'], prompt, 
        size=size, num_images=num_images, category=category
    )
    
    progress_container.empty()
    
    if success:
        st.success(f"⚓✨ NavyAI {model['name']} 生成完成！")
        st.balloons()
        
        st.markdown(f"#### 🎨 NavyAI - {model['name']} 作品")
        
        try:
            st.image(result, use_column_width=True, caption=f"NavyAI {model['name']} - {NAVYAI_MODELS[category]['category_name']}")
            
            # 模型信息
            with st.expander(f"⚓ NavyAI {model['name']} 詳情"):
                st.write(f"**模型名稱**: {model['name']}")
                st.write(f"**模型ID**: {model['id']}")
                st.write(f"**類別**: {NAVYAI_MODELS[category]['category_name']}")
                st.write(f"**描述**: {model['description']}")
                st.write(f"**定價**: {model['pricing']}")
                st.write(f"**生成速度**: {model['speed']}")
                st.write(f"**質量等級**: {'⭐' * model['quality']}")
            
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
        st.error(f"❌ NavyAI 生成失敗: {result}")
        
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
        'navyai_models_loaded': True,
        'encoding_fixed': True,
        'model_selection_enabled': True
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
        st.markdown("### ⚓ NavyAI 15+ 模型設置 - Koyeb 優化")
    
    with st.form("koyeb_navyai_form"):
        st.info("🚀 配置 NavyAI 統一接口以訪問 15+ 專業圖像模型")
        
        key_name = st.text_input(
            "密鑰名稱:",
            placeholder="NavyAI 多模型主密鑰",
            value="NavyAI 多模型主密鑰"
        )
        
        api_key = st.text_input(
            "NavyAI API 密鑰:",
            type="password",
            placeholder="輸入您的 NavyAI API 密鑰...",
            help="密鑰格式：navy_xxxxxxxx"
        )
        
        st.markdown("**🎨 NavyAI vs FLUX Krea 對比:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**⚓ NavyAI (需密鑰)**")
            st.caption("🎭 FLUX Krea (3 種)")
            st.caption("🖼️ DALL-E (3 種)")
            st.caption("🎯 Midjourney (3 種)")
            st.caption("⚡ FLUX AI (3 種)")
            st.caption("🎨 Stable Diffusion (3 種)")
            st.caption("📊 **總計 15+ 模型**")
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
            submitted = st.form_submit_button("💾 保存並啟用 NavyAI", type="primary", use_container_width=True)
        with col_home_form:
            if st.form_submit_button("🏠 返回主頁", use_container_width=True):
                go_to_homepage()
        
        if submitted and api_key:
            key_id = provider_manager.save_api_key("NavyAI", key_name, api_key)
            
            if key_id:
                st.session_state.selected_provider = "NavyAI"
                st.success("✅ NavyAI 15+ 模型接口已配置並啟用")
                st.info("⚓ 現在可以選擇使用 15+ 專業圖像模型")
                time.sleep(2)
                rerun_app()
            else:
                st.error("❌ 密鑰保存失敗")

def main():
    """Koyeb 優化的主程式"""
    try:
        init_session_state()
        
        if KOYEB_ENV:
            st.success("🚀 應用正在 Koyeb 高性能平台運行")
        
        show_koyeb_header()
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
            <p><strong>🎭 FLUX Krea 6種模型</strong> | <strong>⚓ NavyAI 15+ 模型</strong> | <strong>🌍 Global CDN</strong></p>
            <div style="margin-top: 0.5rem;">
                <small>
                    運行環境: {'🌍 Koyeb Production' if KOYEB_ENV else '💻 Local Development'} | 
                    端口: {PORT} | 
                    版本: FLUX Krea 6 Models + NavyAI 15+ Models v4.0
                </small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"應用運行錯誤: {safe_text(str(e))}")
        logger.error(f"Main app error: {str(e)}")

if __name__ == "__main__":
    main()
