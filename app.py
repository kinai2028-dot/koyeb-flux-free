import streamlit as st
import os
import logging
import time
import sqlite3
import uuid
import json
import random
from functools import lru_cache

# 必須是第一個 Streamlit 命令 - 現代化配置
st.set_page_config(
    page_title="AI Image Studio Pro - FLUX Krea + NavyAI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 現代化 CSS 樣式
def load_custom_css():
    """載入自定義 CSS 樣式"""
    st.markdown("""
    <style>
    /* 隱藏 Streamlit 默認元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 全域樣式 */
    .main {
        padding-top: 1rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* 現代化卡片樣式 */
    .modern-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* 暗色主題卡片 */
    .dark-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        color: white;
    }
    
    /* 英雄區塊樣式 */
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin: 2rem 0;
        color: white;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.95;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* 功能卡片 */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        height: 100%;
        transition: all 0.3s ease;
        border: 1px solid rgba(226, 232, 240, 0.5);
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border-color: #3b82f6;
    }
    
    .feature-icon {
        font-size: 3.5rem;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .feature-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1e293b;
        background: linear-gradient(45deg, #1e293b, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .feature-desc {
        color: #64748b;
        line-height: 1.8;
        font-size: 1rem;
    }
    
    /* 模型選擇卡片 */
    .model-card {
        background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px solid transparent;
        cursor: pointer;
        transition: all 0.3s ease;
        height: 100%;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .model-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .model-card:hover {
        border-color: #3b82f6;
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(59, 130, 246, 0.1), 0 10px 10px -5px rgba(59, 130, 246, 0.04);
    }
    
    .model-card:hover::before {
        opacity: 1;
    }
    
    .model-card.selected {
        border-color: #10b981;
        background: linear-gradient(145deg, #ecfdf5 0%, #f0fdf4 100%);
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(16, 185, 129, 0.15);
    }
    
    .model-card.selected::before {
        background: linear-gradient(90deg, #10b981, #059669);
        opacity: 1;
    }
    
    .model-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .model-name {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #1e293b;
    }
    
    .model-desc {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    
    .model-specs {
        font-size: 0.85rem;
        color: #7c3aed;
        font-weight: 600;
        padding: 0.5rem;
        background: rgba(124, 58, 237, 0.1);
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    /* 狀態指示器 */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-indicator.success {
        background: rgba(34, 197, 94, 0.15);
        color: #059669;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .status-indicator.warning {
        background: rgba(245, 158, 11, 0.15);
        color: #d97706;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .status-indicator.error {
        background: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    /* 進度容器 */
    .progress-container {
        background: linear-gradient(145deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    .progress-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    
    .progress-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1e293b;
    }
    
    .progress-desc {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* 側邊欄樣式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%);
    }
    
    /* 按鈕樣式增強 */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* 輸入框樣式 */
    .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .stSelectbox > div > div > select {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
    }
    
    /* 響應式設計 */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        .hero-subtitle {
            font-size: 1rem;
        }
        .feature-card {
            margin-bottom: 1rem;
        }
        .model-card {
            margin-bottom: 1rem;
        }
    }
    
    /* 動畫效果 */
    @keyframes fadeIn {
        from { 
            opacity: 0; 
            transform: translateY(30px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    @keyframes pulse {
        0%, 100% { 
            transform: scale(1); 
        }
        50% { 
            transform: scale(1.05); 
        }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }
    
    .slide-up {
        animation: slideUp 0.6s ease-out;
    }
    
    /* 自定義滾動條 */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #cbd5e1, #94a3b8);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #94a3b8, #64748b);
    }
    </style>
    """, unsafe_allow_html=True)

# 設置環境編碼
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Koyeb 環境檢測和優化設置
KOYEB_ENV = os.getenv('KOYEB_PUBLIC_DOMAIN') is not None
PORT = int(os.getenv('PORT', 8501))

# 日誌配置
logging.basicConfig(
    level=logging.INFO if KOYEB_ENV else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 延遲載入重型模組
@lru_cache(maxsize=1)
def get_heavy_imports():
    """延遲載入重型模組以優化冷啟動時間"""
    imports = {}
    
    try:
        try:
            from openai import OpenAI
            imports['OpenAI'] = OpenAI
            logger.info("OpenAI imported successfully")
        except ImportError as e:
            logger.warning(f"OpenAI import failed: {e}")
            imports['OpenAI'] = None
        
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
            import base64
            imports['base64'] = base64
        except ImportError:
            logger.error("Base64 import failed")
            imports['base64'] = None
        
        return imports
        
    except Exception as e:
        logger.error(f"Unexpected error in imports: {str(e)}")
        return {}

# 安全文本處理
def safe_text(text, max_length=None):
    """安全處理文本，避免編碼錯誤"""
    try:
        if not isinstance(text, str):
            text = str(text)
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        if max_length and len(text) > max_length:
            text = text[:max_length] + "..."
        return text
    except Exception as e:
        logger.warning(f"Text encoding issue: {str(e)}")
        return "Text encoding error"

# 導航功能
def go_to_homepage():
    """返回主頁並清除所有狀態"""
    try:
        # 清除所有會話狀態
        keys_to_clear = [
            'selected_provider', 'show_navyai_setup', 'selected_navyai_model',
            'selected_navyai_category', 'selected_flux_krea_model', 'quick_template',
            'current_page', 'show_gallery', 'generated_images'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state.current_page = 'home'
        rerun_app()
    except Exception as e:
        logger.error(f"Error in go_to_homepage: {str(e)}")
        st.rerun()

def rerun_app():
    """重新運行應用"""
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    else:
        st.stop()

# FLUX Krea 模型庫
FLUX_KREA_MODELS = {
    "flux-krea-dev": {
        "name": "FLUX Krea Dev",
        "model_id": "flux-krea",
        "description": "平衡質量與速度的美學優化版本，適合日常創作使用",
        "pricing": "免費",
        "speed": "~6-8s",
        "quality": 5,
        "recommended": True,
        "speciality": "平衡性能",
        "best_for": ["人像攝影", "風景攝影", "日常創作"],
        "icon": "🎭",
        "color": "#3b82f6",
        "gradient": "linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)"
    },
    "flux-krea-pro": {
        "name": "FLUX Krea Pro",
        "model_id": "flux-krea-pro",
        "description": "專業級美學優化，提供最高質量輸出，適合商業創作",
        "pricing": "免費",
        "speed": "~8-10s",
        "quality": 5,
        "recommended": True,
        "speciality": "最高質量",
        "best_for": ["專業攝影", "商業創作", "藝術作品"],
        "icon": "👑",
        "color": "#7c3aed",
        "gradient": "linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%)"
    },
    "flux-krea-schnell": {
        "name": "FLUX Krea Schnell",
        "model_id": "flux-krea-schnell",
        "description": "快速版本，在保持美學質量的同時提升生成速度",
        "pricing": "免費",
        "speed": "~3-5s",
        "quality": 4,
        "recommended": False,
        "speciality": "極速生成",
        "best_for": ["快速原型", "批量生成", "測試創意"],
        "icon": "⚡",
        "color": "#f59e0b",
        "gradient": "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
    },
    "flux-krea-realism": {
        "name": "FLUX Krea Realism",
        "model_id": "flux-realism",
        "description": "專注寫實風格，適合需要高度真實感的圖像創作",
        "pricing": "免費",
        "speed": "~7-9s",
        "quality": 5,
        "recommended": False,
        "speciality": "寫實專精",
        "best_for": ["寫實人像", "產品攝影", "紀錄風格"],
        "icon": "📸",
        "color": "#059669",
        "gradient": "linear-gradient(135deg, #059669 0%, #047857 100%)"
    },
    "flux-krea-anime": {
        "name": "FLUX Krea Anime",
        "model_id": "flux-anime",
        "description": "動漫風格優化，專門生成高質量的動漫插畫風格圖像",
        "pricing": "免費",
        "speed": "~6-8s",
        "quality": 4,
        "recommended": False,
        "speciality": "動漫風格",
        "best_for": ["動漫角色", "插畫創作", "二次元風格"],
        "icon": "🎌",
        "color": "#ec4899",
        "gradient": "linear-gradient(135deg, #ec4899 0%, #be185d 100%)"
    },
    "flux-krea-artistic": {
        "name": "FLUX Krea Artistic",
        "model_id": "flux-artistic",
        "description": "藝術創作優化，強化創意表現和藝術風格渲染",
        "pricing": "免費",
        "speed": "~8-10s",
        "quality": 5,
        "recommended": False,
        "speciality": "藝術創作",
        "best_for": ["抽象藝術", "創意設計", "概念藝術"],
        "icon": "🎨",
        "color": "#dc2626",
        "gradient": "linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)"
    }
}

# NavyAI 模型庫（簡化版）
NAVYAI_MODELS = {
    "dalle": {
        "category_name": "🖼️ DALL-E",
        "description": "OpenAI 創意圖像生成，擁有強大的文本理解能力",
        "models": [
            {
                "id": "dall-e-3",
                "name": "DALL-E 3",
                "description": "OpenAI 最新創意版本，細節豐富，文本理解強",
                "pricing": "$0.040/image",
                "speed": "~10s",
                "quality": 5,
                "recommended": True,
                "api_model": "dall-e-3",
                "icon": "✨",
                "color": "#10b981",
                "gradient": "linear-gradient(135deg, #10b981 0%, #059669 100%)"
            },
            {
                "id": "dall-e-2",
                "name": "DALL-E 2",
                "description": "經典版本，穩定可靠的圖像生成能力",
                "pricing": "$0.020/image",
                "speed": "~8s",
                "quality": 4,
                "recommended": False,
                "api_model": "dall-e-2",
                "icon": "🎯",
                "color": "#6366f1",
                "gradient": "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)"
            }
        ]
    }
}

# FLUX Krea 預設
FLUX_KREA_PRESETS = {
    "portrait": {
        "name": "🖼️ 人像攝影",
        "prompt_prefix": "professional portrait photography, ",
        "prompt_suffix": ", natural lighting, realistic skin texture, detailed eyes, high resolution",
        "color": "#f59e0b",
        "description": "專業人像攝影風格，自然光照和細節優化"
    },
    "landscape": {
        "name": "🌄 風景攝影",
        "prompt_prefix": "beautiful landscape photography, ",
        "prompt_suffix": ", golden hour lighting, natural colors, scenic view, high detail",
        "color": "#059669",
        "description": "風景攝影風格，黃金時段光照和自然色彩"
    },
    "artistic": {
        "name": "🎨 藝術創作",
        "prompt_prefix": "artistic composition, ",
        "prompt_suffix": ", creative lighting, artistic style, detailed artwork, masterpiece",
        "color": "#dc2626",
        "description": "藝術創作風格，創意照明和藝術化渲染"
    },
    "realistic": {
        "name": "📸 寫實風格",
        "prompt_prefix": "photorealistic, ",
        "prompt_suffix": ", natural appearance, realistic details, authentic style, lifelike",
        "color": "#6b7280",
        "description": "寫實風格，真實自然的外觀和細節"
    }
}

# 數據庫管理器
class KoyebOptimizedProviderManager:
    def __init__(self):
        self.db_path = "/tmp/koyeb_providers.db" if KOYEB_ENV else "koyeb_providers.db"
        self.init_database()
    
    def init_database(self):
        """數據庫初始化"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # API 密鑰表
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
            
            # 圖像歷史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS image_history (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    image_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
    
    def save_api_key(self, provider, key_name, api_key):
        """保存 API 密鑰"""
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
            return key_id
        except Exception as e:
            logger.error(f"API key save failed: {str(e)}")
            return ""
    
    def get_active_api_key(self, provider):
        """獲取活動密鑰"""
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
                return {'id': row[0], 'key_name': row[1], 'api_key': row[2], 'created_at': row[3]}
            return None
        except Exception as e:
            logger.error(f"Key retrieval failed: {str(e)}")
            return None
    
    def save_image(self, provider, model_name, prompt, image_data):
        """保存生成的圖像到歷史"""
        try:
            image_id = str(uuid.uuid4())
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO image_history (id, provider, model_name, prompt, image_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_id, provider, model_name, prompt, image_data))
            conn.commit()
            conn.close()
            return image_id
        except Exception as e:
            logger.error(f"Image save failed: {str(e)}")
            return None

@st.cache_resource
def get_provider_manager():
    return KoyebOptimizedProviderManager()

provider_manager = get_provider_manager()

# 圖像生成函數
def generate_flux_krea_image(prompt, model_id="flux-krea", preset="realistic", size="1024x1024"):
    """FLUX Krea 圖像生成"""
    imports = get_heavy_imports()
    if not imports.get('requests') or not imports.get('base64'):
        return False, "缺少必要的模組 (requests, base64)"
    
    try:
        prompt = safe_text(prompt, max_length=500)
        preset_config = FLUX_KREA_PRESETS.get(preset, FLUX_KREA_PRESETS["realistic"])
        optimized_prompt = f"{preset_config['prompt_prefix']}{prompt}{preset_config['prompt_suffix']}"
        
        import urllib.parse
        encoded_prompt = urllib.parse.quote(optimized_prompt)
        width, height = map(int, size.split('x'))
        
        url_params = [
            f"model={model_id}",
            f"width={width}",
            f"height={height}",
            "nologo=true",
            "enhance=true"
        ]
        
        base_url = "https://image.pollinations.ai/prompt"
        full_url = f"{base_url}/{encoded_prompt}?{'&'.join(url_params)}"
        
        logger.info(f"FLUX Krea API call: {model_id}")
        response = imports['requests'].get(full_url, timeout=45)
        
        if response.status_code == 200:
            encoded_image = imports['base64'].b64encode(response.content).decode()
            image_url = f"data:image/png;base64,{encoded_image}"
            logger.info("FLUX Krea generation successful")
            return True, image_url
        else:
            return False, f"HTTP {response.status_code}"
            
    except Exception as e:
        error_msg = safe_text(str(e))
        logger.error(f"FLUX Krea generation error: {error_msg}")
        return False, error_msg

def generate_navyai_image_real(api_key, model_id, prompt, **params):
    """NavyAI 真實 API 調用"""
    imports = get_heavy_imports()
    if not imports.get('OpenAI'):
        logger.warning("OpenAI not available, using fallback")
        return generate_navyai_image_fallback(api_key, model_id, prompt, **params)
    
    try:
        prompt = safe_text(prompt, max_length=1000)
        api_model = params.get('api_model', 'dall-e-3')
        size = params.get('size', '1024x1024')
        num_images = min(params.get('num_images', 1), 4)
        
        logger.info(f"NavyAI API call: model={api_model}")
        
        client = imports['OpenAI'](
            api_key=api_key,
            base_url="https://api.navy/v1"
        )
        
        response = client.images.generate(
            model=api_model,
            prompt=prompt,
            n=num_images,
            size=size,
            quality="standard",
            response_format="b64_json"
        )
        
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
    """NavyAI 模擬生成（回退版本）"""
    imports = get_heavy_imports()
    if not imports.get('Image') or not imports.get('base64') or not imports.get('BytesIO'):
        return False, "缺少圖像處理模組 (PIL, base64, BytesIO)"
    
    try:
        logger.info("Using NavyAI fallback mode")
        time.sleep(4)  # 模擬 API 調用時間
        
        prompt = safe_text(prompt, max_length=500)
        width, height = map(int, params.get('size', '1024x1024').split('x'))
        
        img = imports['Image'].new('RGB', (width, height))
        draw = imports['ImageDraw'].Draw(img)
        
        # 創建更美觀的漸變背景
        for y in range(height):
            r = int(30 + (120 * y / height))
            g = int(60 + (140 * y / height))
            b = int(120 + (135 * y / height))
            for x in range(width):
                draw.point((x, y), (r, g, b))
        
        try:
            font_large = imports['ImageFont'].load_default()
            font_small = imports['ImageFont'].load_default()
        except:
            font_large = font_small = None
        
        # 添加內容
        draw.text((50, 50), "NavyAI Demo Generation", fill=(255, 255, 255), font=font_large)
        draw.text((50, 100), f"Model: {model_id}", fill=(255, 255, 255), font=font_large)
        
        # 提示詞預覽
        prompt_lines = [prompt[i:i+50] for i in range(0, min(len(prompt), 150), 50)]
        y_offset = 150
        for line in prompt_lines:
            draw.text((50, y_offset), f"Prompt: {line}", fill=(255, 255, 255), font=font_small)
            y_offset += 30
        
        # 狀態信息
        draw.text((50, height - 120), "Fallback Mode - Demo Generation", fill=(255, 255, 0), font=font_small)
        draw.text((50, height - 90), "Real API available with valid key", fill=(255, 255, 255), font=font_small)
        draw.text((50, height - 60), "AI Image Studio Pro", fill=(255, 255, 255), font=font_small)
        
        # 轉換為 base64
        buffer = imports['BytesIO']()
        img.save(buffer, format='PNG')
        encoded_image = imports['base64'].b64encode(buffer.getvalue()).decode()
        
        return True, f"data:image/png;base64,{encoded_image}"
        
    except Exception as e:
        error_msg = safe_text(str(e))
        logger.error(f"NavyAI fallback generation error: {error_msg}")
        return False, error_msg

# 現代化 UI 組件
def show_modern_hero():
    """顯示現代化英雄區塊"""
    st.markdown("""
    <div class="hero-section fade-in">
        <div class="hero-title">🎨 AI Image Studio Pro</div>
        <div class="hero-subtitle">
            專業級 AI 圖像生成平台 • FLUX Krea 美學優化 • NavyAI 統一接口 • 全球高性能部署
        </div>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-top: 2rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.75rem 1.5rem; border-radius: 25px; color: white; font-weight: 600; backdrop-filter: blur(10px);">🎭 6種 FLUX 模型</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.75rem 1.5rem; border-radius: 25px; color: white; font-weight: 600; backdrop-filter: blur(10px);">⚓ 真實 API 調用</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.75rem 1.5rem; border-radius: 25px; color: white; font-weight: 600; backdrop-filter: blur(10px);">🚀 Koyeb 部署</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.75rem 1.5rem; border-radius: 25px; color: white; font-weight: 600; backdrop-filter: blur(10px);">✨ 現代化界面</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_system_status():
    """顯示系統狀態"""
    imports = get_heavy_imports()
    
    st.markdown("### 🔧 系統狀態監控")
    
    # 創建狀態卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "✅ 正常" if imports.get('requests') else "❌ 錯誤"
        color = "success" if imports.get('requests') else "error"
        st.markdown(f"""
        <div class="modern-card text-center slide-up" style="animation-delay: 0.1s;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🌐</div>
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">網絡請求</div>
            <div class="status-indicator {color}">{status}</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">HTTP 客戶端</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = "✅ 正常" if imports.get('Image') else "❌ 錯誤"
        color = "success" if imports.get('Image') else "error"
        st.markdown(f"""
        <div class="modern-card text-center slide-up" style="animation-delay: 0.2s;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🖼️</div>
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">圖像處理</div>
            <div class="status-indicator {color}">{status}</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">PIL 圖像庫</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = "✅ 正常" if imports.get('OpenAI') else "⚠️ 回退"
        color = "success" if imports.get('OpenAI') else "warning"
        st.markdown(f"""
        <div class="modern-card text-center slide-up" style="animation-delay: 0.3s;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🤖</div>
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">OpenAI 接口</div>
            <div class="status-indicator {color}">{status}</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">API 客戶端</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        env_status = "🌍 生產環境" if KOYEB_ENV else "💻 開發環境"
        st.markdown(f"""
        <div class="modern-card text-center slide-up" style="animation-delay: 0.4s;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🚀</div>
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">部署狀態</div>
            <div class="status-indicator success">{env_status}</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">Koyeb 平台</div>
        </div>
        """, unsafe_allow_html=True)

def show_provider_selection():
    """顯示服務提供商選擇"""
    st.markdown("### 🎯 選擇 AI 圖像生成服務")
    st.markdown("選擇最適合您需求的 AI 圖像生成服務，開始您的創作之旅")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card slide-up" style="animation-delay: 0.1s;">
            <div class="feature-icon" style="background: linear-gradient(135deg, #3b82f6, #1d4ed8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🎭</div>
            <div class="feature-title">FLUX Krea AI Studio</div>
            <div class="feature-desc">
                <strong>🎨 6種專業模型選擇</strong><br><br>
                • <strong>Dev & Pro</strong>：平衡性能與最高質量<br>
                • <strong>Schnell</strong>：極速生成，快如閃電<br>
                • <strong>Realism</strong>：寫實風格專精<br>
                • <strong>Anime</strong>：動漫插畫專家<br>
                • <strong>Artistic</strong>：藝術創作優化<br><br>
                ✅ 完全免費使用<br>
                🚀 Koyeb 高性能部署<br>
                🎯 美學算法優化
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎭 啟動 FLUX Krea Studio", type="primary", use_container_width=True):
            st.session_state.selected_provider = "FLUX Krea AI"
            st.session_state.current_page = "flux_krea"
            st.success("✅ FLUX Krea AI Studio 已啟動")
            st.balloons()
            time.sleep(1)
            rerun_app()
    
    with col2:
        st.markdown("""
        <div class="feature-card slide-up" style="animation-delay: 0.2s;">
            <div class="feature-icon" style="background: linear-gradient(135deg, #10b981, #059669); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">⚓</div>
            <div class="feature-title">NavyAI 統一接口</div>
            <div class="feature-desc">
                <strong>🔗 真實 OpenAI 兼容 API</strong><br><br>
                • <strong>DALL-E 3</strong>：OpenAI 最新模型<br>
                • <strong>DALL-E 2</strong>：經典穩定版本<br>
                • <strong>統一接口</strong>：一個 API 多個模型<br><br>
                🌍 真實雲端生成<br>
                🛡️ 自動回退保護<br>
                💰 按使用付費<br>
                📊 專業級質量
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("⚓ 配置 NavyAI 接口", use_container_width=True):
            st.session_state.current_page = "navyai_setup"
            rerun_app()

def show_flux_krea_studio():
    """FLUX Krea Studio 界面"""
    # 頂部導航
    col_nav1, col_nav2, col_nav3 = st.columns([1, 6, 1])
    with col_nav1:
        if st.button("← 回到主頁", use_container_width=True):
            go_to_homepage()
    with col_nav2:
        st.markdown("### 🎭 FLUX Krea AI Studio - 專業美學圖像生成平台")
    with col_nav3:
        if st.button("🖼️ 作品集", use_container_width=True):
            st.session_state.show_gallery = True
    
    # 檢查依賴
    imports = get_heavy_imports()
    if not imports.get('requests') or not imports.get('base64'):
        st.error("⚠️ 系統依賴不完整，FLUX Krea Studio 暫時不可用")
        st.info("請檢查系統配置，確保 requests 和 base64 模組正常載入")
        
        col_back, col_status = st.columns([1, 1])
        with col_back:
            if st.button("🏠 返回主頁", type="primary", use_container_width=True):
                go_to_homepage()
        with col_status:
            if st.button("🔧 檢查系統狀態", use_container_width=True):
                st.session_state.current_page = 'home'
                rerun_app()
        return
    
    # 模型選擇區
    st.markdown("#### 🤖 選擇 FLUX Krea 專業模型")
    st.markdown("每個模型都經過專門優化，適合不同的創作需求和風格偏好")
    
    # 推薦模型
    st.markdown("##### ⭐ 推薦模型 - 最受歡迎的專業選擇")
    recommended_models = {k: v for k, v in FLUX_KREA_MODELS.items() if v['recommended']}
    
    cols = st.columns(len(recommended_models))
    selected_model = None
    
    for i, (model_key, model_info) in enumerate(recommended_models.items()):
        with cols[i]:
            # 檢查是否被選中
            is_selected = st.session_state.get('selected_flux_krea_model', {}).get('name') == model_info['name']
            card_class = "model-card selected" if is_selected else "model-card"
            
            st.markdown(f"""
            <div class="{card_class} slide-up" style="animation-delay: {0.1 + i*0.1}s;">
                <div class="model-icon" style="background: {model_info['gradient']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {model_info['icon']}
                </div>
                <div class="model-name">{model_info['name']}</div>
                <div class="model-desc">{model_info['description']}</div>
                <div class="model-specs">
                    ⚡ 速度: {model_info['speed']}<br>
                    ⭐ 質量: {'★' * model_info['quality']}<br>
                    🎯 專長: {model_info['speciality']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"選擇 {model_info['name']}", key=f"rec_flux_{model_key}", use_container_width=True, type="primary"):
                st.session_state.selected_flux_krea_model = model_info
                selected_model = model_info
                st.success(f"✅ 已選擇 {model_info['name']}")
                time.sleep(0.5)
                rerun_app()
    
    # 專業模型
    st.markdown("##### 🛠️ 專業模型 - 特定領域專精")
    other_models = {k: v for k, v in FLUX_KREA_MODELS.items() if not v['recommended']}
    
    cols = st.columns(4)
    for i, (model_key, model_info) in enumerate(other_models.items()):
        with cols[i % 4]:
            is_selected = st.session_state.get('selected_flux_krea_model', {}).get('name') == model_info['name']
            card_class = "model-card selected" if is_selected else "model-card"
            
            st.markdown(f"""
            <div class="{card_class} slide-up" style="animation-delay: {0.3 + i*0.1}s;">
                <div class="model-icon" style="background: {model_info['gradient']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {model_info['icon']}
                </div>
                <div class="model-name">{model_info['name']}</div>
                <div class="model-desc">{model_info['description']}</div>
                <div class="model-specs">⚡ {model_info['speed']} | 🎯 {model_info['speciality']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"選擇", key=f"sel_flux_{model_key}", use_container_width=True):
                st.session_state.selected_flux_krea_model = model_info
                st.success(f"✅ 已選擇 {model_info['name']}")
                time.sleep(0.5)
                rerun_app()
    
    # 檢查會話中的選擇
    if hasattr(st.session_state, 'selected_flux_krea_model'):
        selected_model = st.session_state.selected_flux_krea_model
    
    if selected_model:
        st.markdown("---")
        
        # 已選擇模型顯示
        st.markdown(f"""
        <div class="modern-card fade-in" style="background: {selected_model['gradient']}; color: white;">
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                <div style="font-size: 3rem;">{selected_model['icon']}</div>
                <div>
                    <div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 0.5rem;">
                        ✅ 已選擇: {selected_model['name']}
                    </div>
                    <div style="opacity: 0.9; font-size: 1rem; margin-bottom: 0.5rem;">
                        {selected_model['description']}
                    </div>
                    <div style="display: flex; gap: 1rem; font-size: 0.9rem; opacity: 0.8;">
                        <span>🎯 {selected_model['speciality']}</span>
                        <span>⚡ {selected_model['speed']}</span>
                        <span>⭐ {'★' * selected_model['quality']}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 生成界面
        col_prompt, col_settings = st.columns([2, 1])
        
        with col_prompt:
            st.markdown("#### ✍️ 創作提示詞")
            st.markdown("描述您想要創作的圖像，越詳細越能體現模型的專業能力")
            
            prompt = st.text_area(
                "輸入您的創意描述:",
                height=140,
                placeholder=f"例如：A professional portrait of a confident woman, natural lighting, realistic skin texture...\n\n當前使用 {selected_model['name']}，擅長 {selected_model['speciality']}",
                help=f"💡 {selected_model['name']} 最適合: {', '.join(selected_model['best_for'])}"
            )
            
            # 智能模板建議
            st.markdown("##### 🎨 智能模板建議")
            st.markdown(f"基於 {selected_model['name']} 的特色，為您推薦以下模板：")
            
            # 根據模型類型提供模板
            if "realism" in selected_model['model_id']:
                templates = [
                    "A professional business portrait with natural lighting and realistic details",
                    "Product photography on white background with commercial studio lighting",
                    "Documentary style street photography capturing authentic human moments",
                    "Architectural interior photography with realistic lighting and textures"
                ]
            elif "anime" in selected_model['model_id']:
                templates = [
                    "Beautiful anime girl character with flowing hair, detailed eyes, vibrant colors",
                    "Fantasy anime warrior in magical forest with dynamic pose and epic lighting",
                    "Cute chibi character with pastel colors, kawaii style, adorable expression",
                    "Anime landscape with cherry blossoms, dreamy atmosphere, soft lighting"
                ]
            elif "artistic" in selected_model['model_id']:
                templates = [
                    "Abstract expressionist painting with bold brushstrokes and vibrant palette",
                    "Surreal dreamscape with floating objects and impossible architecture",
                    "Digital concept art with futuristic elements and creative composition",
                    "Contemporary art installation with conceptual design and artistic interpretation"
                ]
            else:
                templates = [
                    "Professional portrait photography with natural lighting and skin texture",
                    "Golden hour landscape photography with natural colors and atmosphere",
                    "Modern architectural interior with clean design and natural lighting",
                    "Candid street photography with authentic expression and urban setting"
                ]
            
            template_cols = st.columns(2)
            for i, template in enumerate(templates):
                with template_cols[i % 2]:
                    template_preview = template[:45] + "..." if len(template) > 45 else template
                    if st.button(f"💡 {template_preview}", key=f"template_{i}", use_container_width=True):
                        st.session_state.quick_template = template
                        st.success("✅ 模板已應用")
                        time.sleep(0.5)
                        rerun_app()
        
        with col_settings:
            st.markdown("#### 🎛️ 生成設置")
            
            # 美學預設選擇
            st.markdown("##### 🎨 美學風格預設")
            preset_options = list(FLUX_KREA_PRESETS.keys())
            preset_display = []
            
            for preset_key in preset_options:
                preset_info = FLUX_KREA_PRESETS[preset_key]
                preset_display.append(f"{preset_info['name']} - {preset_info['description']}")
            
            selected_preset_idx = st.selectbox(
                "選擇預設風格:",
                range(len(preset_display)),
                format_func=lambda x: preset_display[x],
                index=0,
                help="不同的預設會自動優化提示詞以獲得最佳效果"
            )
            selected_preset = preset_options[selected_preset_idx]
            
            # 生成參數
            st.markdown("##### 📐 圖像參數")
            size_options = ["512x512", "768x768", "1024x1024", "1152x896", "896x1152"]
            size_descriptions = ["正方形 (小)", "正方形 (中)", "正方形 (大)", "橫向", "縱向"]
            
            selected_size = st.selectbox(
                "圖像尺寸:",
                size_options,
                index=2,
                format_func=lambda x: f"{x} ({size_descriptions[size_options.index(x)]})",
                help="不同尺寸適合不同用途，1024x1024 是最佳品質選擇"
            )
            
            # 模型特性展示
            st.markdown(f"""
            <div class="modern-card dark-card">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem;">{selected_model['icon']}</div>
                    <div style="font-weight: 700; font-size: 1.2rem; margin-bottom: 0.5rem;">
                        {selected_model['name']}
                    </div>
                    <div style="margin-bottom: 1rem; opacity: 0.9;">
                        {selected_model['speciality']}
                    </div>
                    <div style="font-size: 0.95rem; opacity: 0.8; line-height: 1.6;">
                        <div>質量等級: {'★' * selected_model['quality']}</div>
                        <div>生成速度: {selected_model['speed']}</div>
                        <div style="margin-top: 0.5rem;">
                            <strong>最適合場景:</strong><br>
                            {' • '.join(selected_model['best_for'])}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 高級設置
            with st.expander("🔧 高級設置", expanded=False):
                st.markdown("**實驗性功能 (即將推出)**")
                st.slider("創意強度", 0.1, 2.0, 1.0, 0.1, disabled=True)
                st.slider("細節增強", 0.5, 2.0, 1.2, 0.1, disabled=True)
                st.checkbox("HDR 處理", disabled=True)
                st.checkbox("色彩增強", disabled=True)
        
        # 檢查快速模板
        if hasattr(st.session_state, 'quick_template'):
            prompt = st.session_state.quick_template
            del st.session_state.quick_template
            rerun_app()
        
        st.markdown("---")
        
        # 生成按鈕區域
        can_generate = prompt.strip() and selected_model
        
        col_generate, col_clear, col_back = st.columns([4, 1, 1])
        with col_generate:
            generate_btn_text = f"🎨 {selected_model['icon']} 開始專業創作"
            if not can_generate:
                generate_btn_text = "📝 請先輸入創作提示詞"
            
            if st.button(
                generate_btn_text,
                type="primary",
                disabled=not can_generate,
                use_container_width=True
            ):
                if can_generate:
                    generate_flux_krea_main(selected_model, prompt, selected_preset, selected_size)
        
        with col_clear:
            if st.button("🔄 重置", use_container_width=True):
                if 'selected_flux_krea_model' in st.session_state:
                    del st.session_state.selected_flux_krea_model
                st.success("✅ 已重置選擇")
                time.sleep(0.5)
                rerun_app()
        
        with col_back:
            if st.button("🏠 主頁", use_container_width=True):
                go_to_homepage()
    
    else:
        # 未選擇模型的狀態
        st.markdown("""
        <div class="modern-card text-center fade-in" style="padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 2rem;">🤖</div>
            <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem; color: #1e293b;">
                請選擇一個 FLUX Krea 專業模型
            </div>
            <div style="color: #64748b; font-size: 1.1rem; line-height: 1.6; max-width: 500px; margin: 0 auto;">
                每個模型都經過專門優化，擁有獨特的專長領域。<br>
                選擇最適合您創作需求的模型，開始專業級圖像生成之旅。
            </div>
            <div style="margin-top: 2rem; color: #7c3aed; font-weight: 600;">
                ⬆️ 在上方選擇您喜歡的模型
            </div>
        </div>
        """, unsafe_allow_html=True)

def generate_flux_krea_main(selected_model, prompt, preset, size):
    """FLUX Krea 主生成流程"""
    # 現代化進度界面
    progress_container = st.empty()
    
    with progress_container.container():
        st.markdown(f"""
        <div class="progress-container fade-in">
            <div class="progress-icon" style="background: {selected_model['gradient']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {selected_model['icon']}
            </div>
            <div class="progress-title">
                {selected_model['name']} 正在創作中...
            </div>
            <div class="progress-desc">
                請稍候，AI 正在運用 {selected_model['speciality']} 為您生成專業級圖像
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stages = [
            f"{selected_model['icon']} 初始化 {selected_model['name']} 美學引擎...",
            f"✨ 應用 {selected_model['speciality']} 專業優化...",
            f"🎨 處理創意提示詞與美學預設...",
            f"🌈 生成專業級色彩與光影方案...",
            f"🔮 {selected_model['name']} 深度渲染處理中...",
            f"✨ 美學細節優化與後處理...",
            f"🎉 創作完成！{selected_model['name']} 專業作品已生成"
        ]
        
        for i, stage in enumerate(stages):
            status_text.info(stage)
            # 根據模型速度調整進度時間
            if "schnell" in selected_model['model_id']:
                time.sleep(0.5)  # 快速模型
            elif "pro" in selected_model['model_id']:
                time.sleep(1.0)  # 專業模型需要更多時間
            else:
                time.sleep(0.8)  # 標準時間
            progress_bar.progress((i + 1) / len(stages))
    
    # 執行圖像生成
    success, result = generate_flux_krea_image(prompt, selected_model['model_id'], preset, size)
    
    progress_container.empty()
    
    if success:
        # 成功界面
        st.success(f"🎉 {selected_model['icon']} {selected_model['name']} 創作完成！")
        st.balloons()
        
        # 保存到歷史記錄
        image_id = provider_manager.save_image("FLUX Krea", selected_model['name'], prompt, result)
        
        # 作品展示區域
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0 1rem 0;">
            <h3 style="background: {selected_model['gradient']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.8rem; margin: 0;">
                🖼️ {selected_model['name']} 專業作品
            </h3>
            <p style="color: #64748b; margin-top: 0.5rem;">
                {selected_model['speciality']} • {FLUX_KREA_PRESETS[preset]['name']} • {size}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 圖像展示容器
        img_container = st.container()
        with img_container:
            st.image(
                result, 
                use_column_width=True, 
                caption=f"由 {selected_model['name']} 生成 • 風格: {FLUX_KREA_PRESETS[preset]['name']} • 提示詞: {prompt[:50]}..."
            )
        
        # 作品詳情與操作
        col_details, col_actions = st.columns([2, 1])
        
        with col_details:
            with st.expander(f"🔍 {selected_model['name']} 創作詳情", expanded=True):
                col_model_info, col_generation_info = st.columns(2)
                
                with col_model_info:
                    st.markdown("**🤖 模型信息**")
                    st.write(f"**名稱**: {selected_model['name']}")
                    st.write(f"**專長**: {selected_model['speciality']}")
                    st.write(f"**質量等級**: {'★' * selected_model['quality']}")
                    st.write(f"**生成速度**: {selected_model['speed']}")
                    st.write(f"**最適合**: {', '.join(selected_model['best_for'])}")
                
                with col_generation_info:
                    st.markdown("**🎨 生成信息**")
                    preset_config = FLUX_KREA_PRESETS[preset]
                    st.write(f"**美學預設**: {preset_config['name']}")
                    st.write(f"**圖像尺寸**: {size}")
                    st.write(f"**創作時間**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    if image_id:
                        st.write(f"**作品ID**: {image_id}")
                    st.write(f"**提示詞長度**: {len(prompt)} 字符")
        
        with col_actions:
            st.markdown("**🛠️ 作品操作**")
            
            if st.button("📥 下載高清作品", use_container_width=True, type="primary"):
                st.success("💡 請右鍵點擊圖像選擇「另存為」")
                st.info("💡 建議保存為 PNG 格式以保持最佳質量")
            
            if st.button("📋 複製提示詞", use_container_width=True):
                st.code(prompt, language="text")
                st.success("✅ 提示詞已顯示，可手動複製")
            
            if st.button("🔄 重新生成", use_container_width=True):
                st.info("🎨 使用相同設定重新生成...")
                time.sleep(1)
                generate_flux_krea_main(selected_model, prompt, preset, size)
            
            if st.button("✨ 新的創作", use_container_width=True):
                if 'selected_flux_krea_model' in st.session_state:
                    del st.session_state.selected_flux_krea_model
                st.success("✅ 已重置，可選擇新模型")
                time.sleep(0.5)
                rerun_app()
        
        # 底部操作欄
        st.markdown("---")
        col_bottom = st.columns(4)
        
        with col_bottom[0]:
            if st.button("🏠 回到主頁", use_container_width=True):
                go_to_homepage()
        
        with col_bottom[1]:
            if st.button("🎭 選擇新模型", use_container_width=True):
                if 'selected_flux_krea_model' in st.session_state:
                    del st.session_state.selected_flux_krea_model
                rerun_app()
        
        with col_bottom[2]:
            if st.button("🖼️ 查看作品集", use_container_width=True):
                st.session_state.show_gallery = True
        
        with col_bottom[3]:
            if st.button("📤 分享作品", use_container_width=True):
                st.info("💡 分享功能即將推出")
                
    else:
        # 失敗界面
        st.error(f"❌ {selected_model['name']} 生成失敗")
        
        st.markdown(f"""
        <div class="modern-card" style="border-left: 4px solid #dc2626;">
            <h4 style="color: #dc2626; margin-bottom: 1rem;">🚨 生成錯誤詳情</h4>
            <p><strong>錯誤信息</strong>: {result}</p>
            <p><strong>使用模型</strong>: {selected_model['name']}</p>
            <p><strong>提示詞長度</strong>: {len(prompt)} 字符</p>
            <p><strong>圖像尺寸</strong>: {size}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_retry, col_home, col_support = st.columns([2, 1, 1])
        with col_retry:
            if st.button("🔄 重試生成", type="primary", use_container_width=True):
                st.info("🔄 正在重新嘗試生成...")
                time.sleep(1)
                generate_flux_krea_main(selected_model, prompt, preset, size)
        
        with col_home:
            if st.button("🏠 返回主頁", use_container_width=True):
                go_to_homepage()
        
        with col_support:
            if st.button("📞 技術支援", use_container_width=True):
                st.info("如問題持續，請檢查網絡連接或稍後重試")

def show_navyai_setup():
    """NavyAI 設置界面"""
    # 頂部導航
    col_nav1, col_nav2 = st.columns([1, 6])
    with col_nav1:
        if st.button("← 回到主頁", use_container_width=True):
            go_to_homepage()
    with col_nav2:
        st.markdown("### ⚓ NavyAI 統一接口配置")
    
    # 英雄區塊
    st.markdown("""
    <div class="modern-card fade-in" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; text-align: center;">
        <div style="font-size: 4rem; margin-bottom: 1.5rem;">⚓</div>
        <div style="font-size: 2rem; font-weight: 700; margin-bottom: 1rem;">NavyAI 統一接口</div>
        <div style="font-size: 1.2rem; opacity: 0.95; line-height: 1.6; max-width: 600px; margin: 0 auto;">
            配置真實的 OpenAI 兼容 API，解鎖 DALL-E 等頂級模型的強大能力。<br>
            一個接口，多個世界級模型，專業品質保證。
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 功能特色
    st.markdown("#### 🌟 NavyAI 接口特色")
    
    feature_cols = st.columns(3)
    with feature_cols[0]:
        st.markdown("""
        <div class="feature-card slide-up" style="animation-delay: 0.1s;">
            <div class="feature-icon">🔗</div>
            <div class="feature-title">真實 API 調用</div>
            <div class="feature-desc">
                直接連接 OpenAI 等頂級 AI 服務商，<br>
                獲得最原汁原味的模型能力
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_cols[1]:
        st.markdown("""
        <div class="feature-card slide-up" style="animation-delay: 0.2s;">
            <div class="feature-icon">🛡️</div>
            <div class="feature-title">智能回退保護</div>
            <div class="feature-desc">
                API 異常時自動切換回退模式，<br>
                確保服務持續可用
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_cols[2]:
        st.markdown("""
        <div class="feature-card slide-up" style="animation-delay: 0.3s;">
            <div class="feature-icon">📊</div>
            <div class="feature-title">統一管理界面</div>
            <div class="feature-desc">
                一個界面管理多個模型，<br>
                簡化使用流程
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 設置表單
    st.markdown("---")
    st.markdown("#### 🔑 API 密鑰配置")
    
    with st.form("navyai_setup_form", clear_on_submit=False):
        col_form1, col_form2 = st.columns([2, 1])
        
        with col_form1:
            key_name = st.text_input(
                "密鑰名稱:",
                placeholder="NavyAI 主密鑰",
                value="NavyAI 主密鑰",
                help="為您的 API 密鑰設定一個易於識別的名稱"
            )
            
            api_key = st.text_input(
                "NavyAI API 密鑰:",
                type="password",
                placeholder="輸入您的 NavyAI API 密鑰...",
                help="格式通常為: navy_xxxxxxxx 或 sk-xxxxxxxx"
            )
            
            st.markdown("**🔐 安全說明**")
            st.info("• 您的 API 密鑰將安全存儲在本地數據庫中\n• 我們不會將您的密鑰傳送到任何第三方服務\n• 密鑰僅用於向 NavyAI 發送圖像生成請求")
        
        with col_form2:
            st.markdown("**📋 支援的模型**")
            st.markdown("""
            **🖼️ DALL-E 系列**
            • DALL-E 3 (推薦)
            • DALL-E 2 (經典)
            
            **⚡ 特色功能**
            • 高解析度輸出
            • 創意文本理解
            • 多尺寸支援
            • 快速生成
            """)
        
        # 服務對比
        st.markdown("---")
        st.markdown("#### 📊 服務詳細對比")
        
        comparison_cols = st.columns(2)
        
        with comparison_cols[0]:
            st.markdown("""
            <div class="feature-card" style="background: linear-gradient(145deg, #ecfdf5 0%, #f0fdf4 100%); border: 2px solid #10b981;">
                <div class="feature-icon" style="color: #10b981;">⚓</div>
                <div class="feature-title" style="color: #10b981;">NavyAI 統一接口</div>
                <div class="feature-desc">
                    <strong>💰 付費服務</strong><br><br>
                    ✅ DALL-E 3 最新模型<br>
                    ✅ 真實雲端 API 調用<br>
                    ✅ 最高生成質量<br>
                    ✅ 商業級穩定性<br>
                    ✅ 多尺寸支援<br>
                    ✅ 快速響應速度<br><br>
                    <strong>💳 按使用量計費</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with comparison_cols[1]:
            st.markdown("""
            <div class="feature-card" style="background: linear-gradient(145deg, #eff6ff 0%, #f0f9ff 100%); border: 2px solid #3b82f6;">
                <div class="feature-icon" style="color: #3b82f6;">🎭</div>
                <div class="feature-title" style="color: #3b82f6;">FLUX Krea Studio</div>
                <div class="feature-desc">
                    <strong>🆓 免費服務</strong><br><br>
                    ✅ 6種專業模型選擇<br>
                    ✅ 美學算法優化<br>
                    ✅ 高質量輸出<br>
                    ✅ 完全免費使用<br>
                    ✅ 多種風格支援<br>
                    ✅ Koyeb 高性能部署<br><br>
                    <strong>💎 永久免費</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 提交按鈕
        st.markdown("---")
        col_submit, col_cancel = st.columns([3, 1])
        
        with col_submit:
            submitted = st.form_submit_button(
                "💾 保存配置並啟用 NavyAI 接口", 
                type="primary", 
                use_container_width=True
            )
        
        with col_cancel:
            if st.form_submit_button("🏠 返回主頁", use_container_width=True):
                go_to_homepage()
        
        # 處理表單提交
        if submitted:
            if not api_key.strip():
                st.error("❌ 請輸入有效的 API 密鑰")
            else:
                with st.spinner("正在驗證並保存 API 密鑰..."):
                    key_id = provider_manager.save_api_key("
