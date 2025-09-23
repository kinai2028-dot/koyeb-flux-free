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
    }
    
    .feature-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1e293b;
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
    
    .model-card:hover {
        border-color: #3b82f6;
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(59, 130, 246, 0.1), 0 10px 10px -5px rgba(59, 130, 246, 0.04);
    }
    
    .model-card.selected {
        border-color: #10b981;
        background: linear-gradient(145deg, #ecfdf5 0%, #f0fdf4 100%);
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(16, 185, 129, 0.15);
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
    
    /* 動畫效果 */
    @keyframes pulse {
        0%, 100% { 
            transform: scale(1); 
        }
        50% { 
            transform: scale(1.05); 
        }
    }
    
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
        "color": "#3b82f6"
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
        "color": "#7c3aed"
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
        "color": "#f59e0b"
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
        "color": "#059669"
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
        "color": "#ec4899"
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
        "color": "#dc2626"
    }
}

# FLUX Krea 預設
FLUX_KREA_PRESETS = {
    "portrait": {
        "name": "🖼️ 人像攝影",
        "prompt_prefix": "professional portrait photography, ",
        "prompt_suffix": ", natural lighting, realistic skin texture, detailed eyes, high resolution",
        "color": "#f59e0b"
    },
    "landscape": {
        "name": "🌄 風景攝影",
        "prompt_prefix": "beautiful landscape photography, ",
        "prompt_suffix": ", golden hour lighting, natural colors, scenic view, high detail",
        "color": "#059669"
    },
    "artistic": {
        "name": "🎨 藝術創作",
        "prompt_prefix": "artistic composition, ",
        "prompt_suffix": ", creative lighting, artistic style, detailed artwork, masterpiece",
        "color": "#dc2626"
    },
    "realistic": {
        "name": "📸 寫實風格",
        "prompt_prefix": "photorealistic, ",
        "prompt_suffix": ", natural appearance, realistic details, authentic style, lifelike",
        "color": "#6b7280"
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
            <span style="background: rgba(255,255,255,0.2); padding: 0.75rem 1.5rem; border-radius: 25px; color: white; font-weight: 600;">🎭 6種 FLUX 模型</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.75rem 1.5rem; border-radius: 25px; color: white; font-weight: 600;">⚓ 真實 API 調用</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.75rem 1.5rem; border-radius: 25px; color: white; font-weight: 600;">🚀 Koyeb 部署</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.75rem 1.5rem; border-radius: 25px; color: white; font-weight: 600;">✨ 現代化界面</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_system_status():
    """顯示系統狀態"""
    imports = get_heavy_imports()
    
    st.markdown("### 🔧 系統狀態監控")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "✅ 正常" if imports.get('requests') else "❌ 錯誤"
        color = "success" if imports.get('requests') else "error"
        st.markdown(f"""
        <div class="modern-card text-center slide-up" style="animation-delay: 0.1s;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🌐</div>
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">網絡請求</div>
            <div class="status-indicator {color}">{status}</div>
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
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        env_status = "🌍 生產環境" if KOYEB_ENV else "💻 開發環境"
        st.markdown(f"""
        <div class="modern-card text-center slide-up" style="animation-delay: 0.4s;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🚀</div>
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">部署狀態</div>
            <div class="status-indicator success">{env_status}</div>
        </div>
        """, unsafe_allow_html=True)

def show_provider_selection():
    """顯示服務提供商選擇"""
    st.markdown("### 🎯 選擇 AI 圖像生成服務")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card slide-up" style="animation-delay: 0.1s;">
            <div class="feature-icon">🎭</div>
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
            <div class="feature-icon">⚓</div>
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
        col_back, _ = st.columns([1, 1])
        with col_back:
            if st.button("🏠 返回主頁", type="primary", use_container_width=True):
                go_to_homepage()
        return
    
    # 模型選擇區
    st.markdown("#### 🤖 選擇 FLUX Krea 專業模型")
    
    # 推薦模型
    st.markdown("##### ⭐ 推薦模型")
    recommended_models = {k: v for k, v in FLUX_KREA_MODELS.items() if v['recommended']}
    
    cols = st.columns(len(recommended_models))
    selected_model = None
    
    for i, (model_key, model_info) in enumerate(recommended_models.items()):
        with cols[i]:
            is_selected = st.session_state.get('selected_flux_krea_model', {}).get('name') == model_info['name']
            card_class = "model-card selected" if is_selected else "model-card"
            
            st.markdown(f"""
            <div class="{card_class} slide-up" style="animation-delay: {0.1 + i*0.1}s;">
                <div class="model-icon" style="color: {model_info['color']};">{model_info['icon']}</div>
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
    st.markdown("##### 🛠️ 專業模型")
    other_models = {k: v for k, v in FLUX_KREA_MODELS.items() if not v['recommended']}
    
    cols = st.columns(4)
    for i, (model_key, model_info) in enumerate(other_models.items()):
        with cols[i % 4]:
            is_selected = st.session_state.get('selected_flux_krea_model', {}).get('name') == model_info['name']
            card_class = "model-card selected" if is_selected else "model-card"
            
            st.markdown(f"""
            <div class="{card_class} slide-up" style="animation-delay: {0.3 + i*0.1}s;">
                <div class="model-icon" style="color: {model_info['color']};">{model_info['icon']}</div>
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
        <div class="modern-card fade-in" style="background: linear-gradient(135deg, {selected_model['color']} 0%, {selected_model['color']}80 100%); color: white;">
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
            
            prompt = st.text_area(
                "輸入您的創意描述:",
                height=140,
                placeholder=f"例如：A professional portrait of a confident woman, natural lighting...\n\n當前使用 {selected_model['name']}，擅長 {selected_model['speciality']}",
                help=f"💡 {selected_model['name']} 最適合: {', '.join(selected_model['best_for'])}"
            )
            
            # 智能模板建議
            st.markdown("##### 🎨 智能模板建議")
            
            if "realism" in selected_model['model_id']:
                templates = [
                    "A professional business portrait with natural lighting and realistic details",
                    "Product photography on white background with commercial studio lighting"
                ]
            elif "anime" in selected_model['model_id']:
                templates = [
                    "Beautiful anime girl character with flowing hair, detailed eyes, vibrant colors",
                    "Fantasy anime warrior in magical forest with dynamic pose and epic lighting"
                ]
            elif "artistic" in selected_model['model_id']:
                templates = [
                    "Abstract expressionist painting with bold brushstrokes and vibrant palette",
                    "Surreal dreamscape with floating objects and impossible architecture"
                ]
            else:
                templates = [
                    "Professional portrait photography with natural lighting and skin texture",
                    "Golden hour landscape photography with natural colors and atmosphere"
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
            preset_names = [FLUX_KREA_PRESETS[p]["name"] for p in preset_options]
            
            selected_preset_idx = st.selectbox(
                "選擇預設風格:",
                range(len(preset_names)),
                format_func=lambda x: preset_names[x],
                index=0
            )
            selected_preset = preset_options[selected_preset_idx]
            
            # 生成參數
            st.markdown("##### 📐 圖像參數")
            size_options = ["512x512", "768x768", "1024x1024", "1152x896", "896x1152"]
            selected_size = st.selectbox("圖像尺寸:", size_options, index=2)
            
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
        
        # 檢查快速模板
        if hasattr(st.session_state, 'quick_template'):
            prompt = st.session_state.quick_template
            del st.session_state.quick_template
            rerun_app()
        
        st.markdown("---")
        
        # 生成按鈕
        can_generate = prompt.strip() and selected_model
        
        col_generate, col_clear, col_back = st.columns([4, 1, 1])
        with col_generate:
            if st.button(
                f"🎨 {selected_model['icon']} 開始專業創作",
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
                rerun_app()
        
        with col_back:
            if st.button("🏠 主頁", use_container_width=True):
                go_to_homepage()
    
    else:
        st.markdown("""
        <div class="modern-card text-center fade-in" style="padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 2rem;">🤖</div>
            <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem; color: #1e293b;">
                請選擇一個 FLUX Krea 專業模型
            </div>
            <div style="color: #64748b; font-size: 1.1rem; line-height: 1.6; max-width: 500px; margin: 0 auto;">
                每個模型都經過專門優化，擁有獨特的專長領域。
            </div>
        </div>
        """, unsafe_allow_html=True)

def generate_flux_krea_main(selected_model, prompt, preset, size):
    """FLUX Krea 主生成流程"""
    progress_container = st.empty()
    
    with progress_container.container():
        st.markdown(f"""
        <div class="progress-container fade-in">
            <div class="progress-icon" style="color: {selected_model['color']};">
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
            f"🎉 創作完成！{selected_model['name']} 專業作品已生成"
        ]
        
        for i, stage in enumerate(stages):
            status_text.info(stage)
            time.sleep(0.8)
            progress_bar.progress((i + 1) / len(stages))
    
    success, result = generate_flux_krea_image(prompt, selected_model['model_id'], preset, size)
    
    progress_container.empty()
    
    if success:
        st.success(f"🎉 {selected_model['icon']} {selected_model['name']} 創作完成！")
        st.balloons()
        
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0 1rem 0;">
            <h3 style="color: {selected_model['color']}; font-size: 1.8rem; margin: 0;">
                🖼️ {selected_model['name']} 專業作品
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.image(
            result, 
            use_column_width=True, 
            caption=f"由 {selected_model['name']} 生成 • 風格: {FLUX_KREA_PRESETS[preset]['name']}"
        )
        
        col_download, col_regen, col_home = st.columns([2, 2, 1])
        
        with col_download:
            if st.button("📥 下載作品", use_container_width=True):
                st.info("💡 右鍵點擊圖像選擇「另存為」")
        
        with col_regen:
            if st.button("🔄 重新生成", use_container_width=True):
                generate_flux_krea_main(selected_model, prompt, preset, size)
        
        with col_home:
            if st.button("🏠 主頁", use_container_width=True):
                go_to_homepage()
                
    else:
        st.error(f"❌ {selected_model['name']} 生成失敗: {result}")
        col_retry, col_home = st.columns([3, 1])
        with col_retry:
            if st.button("🔄 重試生成", type="primary", use_container_width=True):
                generate_flux_krea_main(selected_model, prompt, preset, size)
        with col_home:
            if st.button("🏠 主頁", use_container_width=True):
                go_to_homepage()

# 初始化會話狀態
def init_session_state():
    """初始化會話狀態"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'

def main():
    """主程式"""
    try:
        load_custom_css()
        init_session_state()
        
        # 側邊欄
        with st.sidebar:
            st.markdown("### 🎨 AI Image Studio Pro")
            st.markdown("---")
            
            if st.button("🏠 主頁", use_container_width=True):
                st.session_state.current_page = 'home'
                rerun_app()
            
            if st.button("🎭 FLUX Krea Studio", use_container_width=True):
                st.session_state.current_page = 'flux_krea'
                rerun_app()
            
            st.markdown("---")
            st.markdown("### ℹ️ 系統信息")
            if KOYEB_ENV:
                st.success("🌍 Koyeb 生產環境")
            else:
                st.info("💻 本地開發環境")
            
            st.caption(f"端口: {PORT}")
            st.caption("版本: v8.0 修復版")
        
        # 主內容區域
        current_page = st.session_state.get('current_page', 'home')
        
        if current_page == 'home':
            show_modern_hero()
            show_system_status()
            st.markdown("---")
            show_provider_selection()
        
        elif current_page == 'flux_krea':
            show_flux_krea_studio()
        
        else:
            st.session_state.current_page = 'home'
            rerun_app()
        
        # 頁腳
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); border-radius: 12px; margin-top: 3rem;">
            <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem; color: #1e293b;">
                🚀 AI Image Studio Pro - Powered by Koyeb
            </div>
            <div style="color: #64748b; margin-bottom: 1rem;">
                專業級 AI 圖像生成平台 • FLUX Krea 美學優化 • 全球 CDN 加速
            </div>
            <div style="font-size: 0.9rem; color: #94a3b8;">
                運行環境: {'🌍 Koyeb Production' if KOYEB_ENV else '💻 Local Development'} | 
                版本: v8.0 修復版 | 
                構建時間: 2025-09-23
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"應用運行錯誤: {safe_text(str(e))}")
        logger.error(f"Main app error: {str(e)}")

if __name__ == "__main__":
    main()
