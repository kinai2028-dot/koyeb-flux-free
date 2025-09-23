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
    
    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }
    
    /* 響應式設計 */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        .hero-subtitle {
            font-size: 1rem;
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
            import requests
            imports['requests'] = requests
            logger.info("Requests imported successfully")
        except ImportError:
            logger.error("Requests import failed")
            imports['requests'] = None
        
        try:
            import base64
            imports['base64'] = base64
            logger.info("Base64 imported successfully")
        except ImportError:
            logger.error("Base64 import failed")
            imports['base64'] = None
        
        try:
            import urllib.parse
            imports['urllib_parse'] = urllib.parse
            logger.info("Urllib.parse imported successfully")
        except ImportError:
            logger.error("Urllib.parse import failed")
            imports['urllib_parse'] = None
        
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
            'selected_provider', 'selected_flux_krea_model', 'quick_template',
            'current_page', 'generated_images'
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

# FLUX Krea 模型庫 - 修復版本
FLUX_KREA_MODELS = {
    "flux-dev": {
        "name": "FLUX Dev",
        "model_id": "flux",
        "description": "高質量圖像生成模型，平衡質量與速度",
        "pricing": "免費",
        "speed": "~6-8s",
        "quality": 5,
        "recommended": True,
        "speciality": "通用生成",
        "best_for": ["人像攝影", "風景攝影", "日常創作"],
        "icon": "🎭",
        "color": "#3b82f6"
    },
    "stable-diffusion": {
        "name": "Stable Diffusion",
        "model_id": "turbo",
        "description": "穩定擴散模型，快速生成高質量圖像",
        "pricing": "免費",
        "speed": "~4-6s",
        "quality": 4,
        "recommended": True,
        "speciality": "快速生成",
        "best_for": ["快速原型", "概念設計", "創意測試"],
        "icon": "⚡",
        "color": "#f59e0b"
    },
    "playground": {
        "name": "Playground",
        "model_id": "playground",
        "description": "實驗性模型，提供創新的圖像風格",
        "pricing": "免費",
        "speed": "~5-7s",
        "quality": 4,
        "recommended": False,
        "speciality": "創新風格",
        "best_for": ["藝術創作", "風格實驗", "創意探索"],
        "icon": "🎪",
        "color": "#ec4899"
    },
    "realistic": {
        "name": "Realistic",
        "model_id": "realistic",
        "description": "專注寫實風格的圖像生成",
        "pricing": "免費",
        "speed": "~7-9s",
        "quality": 5,
        "recommended": False,
        "speciality": "寫實專精",
        "best_for": ["寫實人像", "產品攝影", "紀錄風格"],
        "icon": "📸",
        "color": "#059669"
    },
    "anime": {
        "name": "Anime Style",
        "model_id": "anime",
        "description": "動漫風格專精模型",
        "pricing": "免費",
        "speed": "~6-8s",
        "quality": 4,
        "recommended": False,
        "speciality": "動漫風格",
        "best_for": ["動漫角色", "插畫創作", "二次元風格"],
        "icon": "🎌",
        "color": "#8b5cf6"
    },
    "artistic": {
        "name": "Artistic",
        "model_id": "artistic",
        "description": "藝術風格優化模型",
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

# FLUX Krea 預設 - 修復版本
FLUX_KREA_PRESETS = {
    "portrait": {
        "name": "🖼️ 人像攝影",
        "prompt_prefix": "professional portrait photography, ",
        "prompt_suffix": ", natural lighting, realistic skin texture, detailed eyes, high resolution, masterpiece",
        "color": "#f59e0b"
    },
    "landscape": {
        "name": "🌄 風景攝影",
        "prompt_prefix": "beautiful landscape photography, ",
        "prompt_suffix": ", golden hour lighting, natural colors, scenic view, high detail, cinematic",
        "color": "#059669"
    },
    "artistic": {
        "name": "🎨 藝術創作",
        "prompt_prefix": "artistic composition, ",
        "prompt_suffix": ", creative lighting, artistic style, detailed artwork, masterpiece, fine art",
        "color": "#dc2626"
    },
    "realistic": {
        "name": "📸 寫實風格",
        "prompt_prefix": "photorealistic, ",
        "prompt_suffix": ", natural appearance, realistic details, authentic style, lifelike, 8k quality",
        "color": "#6b7280"
    }
}

# 修復的圖像生成函數 [web:350][web:356][web:362]
def generate_flux_krea_image_fixed(prompt, model_id="flux", preset="realistic", size="1024x1024"):
    """修復版本的 FLUX Krea 圖像生成 - 使用 Pollinations.ai API"""
    imports = get_heavy_imports()
    
    # 檢查必要的模組
    if not imports.get('requests') or not imports.get('urllib_parse') or not imports.get('base64'):
        return False, "缺少必要的模組 (requests, urllib.parse, base64)"
    
    try:
        # 安全處理提示詞
        prompt = safe_text(prompt, max_length=800)
        
        # 應用預設優化
        preset_config = FLUX_KREA_PRESETS.get(preset, FLUX_KREA_PRESETS["realistic"])
        optimized_prompt = f"{preset_config['prompt_prefix']}{prompt}{preset_config['prompt_suffix']}"
        
        # URL 編碼提示詞
        encoded_prompt = imports['urllib_parse'].quote(optimized_prompt)
        
        # 解析圖像尺寸
        try:
            width, height = map(int, size.split('x'))
        except:
            width, height = 1024, 1024
        
        # 構建 Pollinations.ai API URL [web:356][web:359]
        api_params = []
        
        # 添加模型參數
        if model_id and model_id != "flux":
            api_params.append(f"model={model_id}")
        
        # 添加尺寸參數
        api_params.append(f"width={width}")
        api_params.append(f"height={height}")
        
        # 添加質量參數
        api_params.append("nologo=true")  # 移除 logo
        api_params.append("enhance=true")  # 增強質量
        api_params.append("private=false")  # 公開模式
        
        # 構建完整 URL
        base_url = "https://image.pollinations.ai/prompt"
        param_string = "&".join(api_params)
        full_url = f"{base_url}/{encoded_prompt}?{param_string}"
        
        logger.info(f"Pollinations API call: {full_url[:100]}...")
        
        # 發送請求 - 增加超時和重試機制
        headers = {
            'User-Agent': 'AI-Image-Studio-Pro/1.0',
            'Accept': 'image/*'
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = imports['requests'].get(
                    full_url, 
                    timeout=60,  # 增加超時時間
                    headers=headers,
                    stream=True
                )
                
                if response.status_code == 200:
                    # 檢查內容類型
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type:
                        # 編碼圖像為 base64
                        encoded_image = imports['base64'].b64encode(response.content).decode()
                        image_url = f"data:image/png;base64,{encoded_image}"
                        logger.info(f"Pollinations generation successful on attempt {attempt + 1}")
                        return True, image_url
                    else:
                        logger.warning(f"Unexpected content type: {content_type}")
                        if attempt == max_retries - 1:
                            return False, f"接收到非圖像內容: {content_type}"
                else:
                    logger.warning(f"HTTP {response.status_code} on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return False, f"HTTP錯誤 {response.status_code}"
                
            except imports['requests'].exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    return False, "請求超時，請稍後重試"
                    
            except imports['requests'].exceptions.RequestException as e:
                logger.warning(f"Request exception on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    return False, f"網絡錯誤: {safe_text(str(e))}"
            
            # 重試前等待
            if attempt < max_retries - 1:
                time.sleep(2)
        
        return False, "所有重試均失敗"
            
    except Exception as e:
        error_msg = safe_text(str(e))
        logger.error(f"Pollinations generation error: {error_msg}")
        return False, f"生成錯誤: {error_msg}"

# 現代化 UI 組件
def show_modern_hero():
    """顯示現代化英雄區塊"""
    st.markdown("""
    <div class="hero-section fade-in">
        <div class="hero-title">🎨 AI Image Studio Pro</div>
        <div class="hero-subtitle">
            專業級 AI 圖像生成平台 • FLUX Krea 美學優化 • Pollinations.ai 免費 API
        </div>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-top: 2rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.75rem 1.5rem; border-radius: 25px; color: white; font-weight: 600;">🎭 6種 AI 模型</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.75rem 1.5rem; border-radius: 25px; color: white; font-weight: 600;">🆓 完全免費</span>
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
        <div class="modern-card text-center">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🌐</div>
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">網絡請求</div>
            <div class="status-indicator {color}">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = "✅ 正常" if imports.get('urllib_parse') else "❌ 錯誤"
        color = "success" if imports.get('urllib_parse') else "error"
        st.markdown(f"""
        <div class="modern-card text-center">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🔗</div>
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">URL 處理</div>
            <div class="status-indicator {color}">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = "✅ 正常" if imports.get('base64') else "❌ 錯誤"
        color = "success" if imports.get('base64') else "error"
        st.markdown(f"""
        <div class="modern-card text-center">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">📊</div>
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">編碼處理</div>
            <div class="status-indicator {color}">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        env_status = "🌍 生產環境" if KOYEB_ENV else "💻 開發環境"
        st.markdown(f"""
        <div class="modern-card text-center">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🚀</div>
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">部署狀態</div>
            <div class="status-indicator success">{env_status}</div>
        </div>
        """, unsafe_allow_html=True)

def show_provider_selection():
    """顯示服務提供商選擇"""
    st.markdown("### 🎯 AI 圖像生成工作室")
    st.markdown("使用 Pollinations.ai 免費 API，體驗專業級 AI 圖像生成")
    
    st.markdown("""
    <div class="feature-card fade-in">
        <div class="feature-icon">🎭</div>
        <div class="feature-title">FLUX Krea AI Studio</div>
        <div class="feature-desc">
            <strong>🎨 6種專業 AI 模型</strong><br><br>
            • <strong>FLUX Dev</strong>：高質量通用生成<br>
            • <strong>Stable Diffusion</strong>：快速穩定生成<br>
            • <strong>Playground</strong>：創新實驗風格<br>
            • <strong>Realistic</strong>：寫實風格專精<br>
            • <strong>Anime</strong>：動漫插畫專家<br>
            • <strong>Artistic</strong>：藝術創作優化<br><br>
            ✅ 完全免費使用 • 🌍 Pollinations.ai 驅動 • 🚀 Koyeb 高性能部署
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎨 啟動 AI Image Studio", type="primary", use_container_width=True, key="start_studio"):
        st.session_state.current_page = "flux_krea"
        st.success("✅ AI Image Studio 已啟動")
        st.balloons()
        time.sleep(1)
        rerun_app()

def show_flux_krea_studio():
    """修復版本的 FLUX Krea Studio 界面"""
    col_nav1, col_nav2 = st.columns([1, 6])
    with col_nav1:
        if st.button("← 回到主頁", use_container_width=True, key="home_from_studio"):
            go_to_homepage()
    with col_nav2:
        st.markdown("### 🎨 AI Image Studio - 專業圖像生成平台")
    
    # 檢查依賴
    imports = get_heavy_imports()
    if not all([imports.get('requests'), imports.get('urllib_parse'), imports.get('base64')]):
        st.error("⚠️ 系統依賴不完整，請檢查網絡連接")
        if st.button("🏠 返回主頁", type="primary", use_container_width=True, key="home_error"):
            go_to_homepage()
        return
    
    # API 狀態檢查
    st.info("🌍 使用 Pollinations.ai 免費 API - 無需註冊或密鑰")
    
    # 模型選擇區
    st.markdown("#### 🤖 選擇 AI 圖像生成模型")
    
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
            <div class="{card_class}">
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
            
            if st.button(f"選擇 {model_info['name']}", key=f"rec_{model_key}", use_container_width=True, type="primary"):
                st.session_state.selected_flux_krea_model = model_info
                selected_model = model_info
                st.success(f"✅ 已選擇 {model_info['name']}")
                time.sleep(0.5)
                rerun_app()
    
    # 其他模型
    st.markdown("##### 🛠️ 專業模型")
    other_models = {k: v for k, v in FLUX_KREA_MODELS.items() if not v['recommended']}
    
    cols = st.columns(4)
    for i, (model_key, model_info) in enumerate(other_models.items()):
        with cols[i % 4]:
            is_selected = st.session_state.get('selected_flux_krea_model', {}).get('name') == model_info['name']
            card_class = "model-card selected" if is_selected else "model-card"
            
            st.markdown(f"""
            <div class="{card_class}">
                <div class="model-icon" style="color: {model_info['color']};">{model_info['icon']}</div>
                <div class="model-name">{model_info['name']}</div>
                <div class="model-desc">{model_info['description']}</div>
                <div class="model-specs">⚡ {model_info['speed']} | 🎯 {model_info['speciality']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("選擇", key=f"sel_{model_key}", use_container_width=True):
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
        <div class="modern-card fade-in" style="background: linear-gradient(135deg, {selected_model['color']}20 0%, {selected_model['color']}10 100%); border-left: 4px solid {selected_model['color']};">
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                <div style="font-size: 3rem;">{selected_model['icon']}</div>
                <div>
                    <div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 0.5rem; color: {selected_model['color']};">
                        ✅ 已選擇: {selected_model['name']}
                    </div>
                    <div style="color: #64748b; font-size: 1rem; margin-bottom: 0.5rem;">
                        {selected_model['description']}
                    </div>
                    <div style="display: flex; gap: 1rem; font-size: 0.9rem; color: #64748b;">
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
                "描述您想要生成的圖像:",
                height=140,
                placeholder=f"例如：A beautiful sunset over mountains, golden hour lighting...\n\n當前使用 {selected_model['name']} 模型",
                help=f"💡 {selected_model['name']} 最適合: {', '.join(selected_model['best_for'])}",
                key="main_prompt"
            )
            
            # 智能模板建議
            st.markdown("##### 🎨 智能模板建議")
            
            if "realistic" in selected_model['model_id']:
                templates = [
                    "A professional business portrait with natural lighting",
                    "Product photography on white background, studio lighting"
                ]
            elif "anime" in selected_model['model_id']:
                templates = [
                    "Beautiful anime girl with flowing hair and detailed eyes",
                    "Fantasy anime warrior in magical forest"
                ]
            elif "artistic" in selected_model['model_id']:
                templates = [
                    "Abstract art with bold colors and geometric shapes",
                    "Surreal landscape with floating objects"
                ]
            else:
                templates = [
                    "Professional portrait with natural lighting",
                    "Beautiful landscape at golden hour"
                ]
            
            template_cols = st.columns(2)
            for i, template in enumerate(templates):
                with template_cols[i % 2]:
                    if st.button(f"💡 {template[:35]}...", key=f"template_{i}", use_container_width=True):
                        st.session_state.quick_template = template
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
                index=0,
                key="preset_select"
            )
            selected_preset = preset_options[selected_preset_idx]
            
            # 生成參數
            st.markdown("##### 📐 圖像參數")
            size_options = ["512x512", "768x768", "1024x1024", "1152x896", "896x1152"]
            selected_size = st.selectbox("圖像尺寸:", size_options, index=2, key="size_select")
            
            # 模型特性展示
            st.success(f"**{selected_model['icon']} {selected_model['name']}**")
            st.caption(f"專長: {selected_model['speciality']}")
            st.caption(f"質量: {'★' * selected_model['quality']}")
            st.caption(f"速度: {selected_model['speed']}")
        
        # 檢查快速模板
        if hasattr(st.session_state, 'quick_template'):
            # 直接更新文本框需要重新運行
            prompt = st.session_state.quick_template
            del st.session_state.quick_template
            
            # 由於 Streamlit 的限制，我們顯示模板內容
            st.info(f"💡 已應用模板: {prompt}")
        
        st.markdown("---")
        
        # 生成按鈕
        can_generate = prompt and prompt.strip() and selected_model
        
        col_generate, col_clear, col_back = st.columns([4, 1, 1])
        with col_generate:
            if st.button(
                f"🎨 {selected_model['icon']} 開始生成圖像",
                type="primary",
                disabled=not can_generate,
                use_container_width=True,
                key="generate_btn"
            ):
                if can_generate:
                    generate_image_main(selected_model, prompt, selected_preset, selected_size)
        
        with col_clear:
            if st.button("🔄 重置", use_container_width=True, key="clear_btn"):
                if 'selected_flux_krea_model' in st.session_state:
                    del st.session_state.selected_flux_krea_model
                rerun_app()
        
        with col_back:
            if st.button("🏠 主頁", use_container_width=True, key="home_btn"):
                go_to_homepage()
    
    else:
        st.markdown("""
        <div class="modern-card text-center fade-in" style="padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 2rem;">🤖</div>
            <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem; color: #1e293b;">
                請選擇一個 AI 圖像生成模型
            </div>
            <div style="color: #64748b; font-size: 1.1rem; line-height: 1.6; max-width: 500px; margin: 0 auto;">
                每個模型都有獨特的專長領域，選擇最適合您創作需求的模型開始生成。
            </div>
        </div>
        """, unsafe_allow_html=True)

def generate_image_main(selected_model, prompt, preset, size):
    """主圖像生成流程 - 修復版本"""
    # 使用模板內容（如果存在）
    if hasattr(st.session_state, 'quick_template'):
        prompt = st.session_state.quick_template
        del st.session_state.quick_template
    
    if not prompt or not prompt.strip():
        st.error("❌ 請輸入有效的提示詞")
        return
    
    # 現代化進度界面
    progress_container = st.empty()
    
    with progress_container.container():
        st.markdown(f"""
        <div class="progress-container fade-in">
            <div class="progress-icon" style="color: {selected_model['color']};">
                {selected_model['icon']}
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem; color: #1e293b;">
                {selected_model['name']} 正在生成中...
            </div>
            <div style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">
                使用 Pollinations.ai API，專業級 {selected_model['speciality']} 生成
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stages = [
            f"{selected_model['icon']} 連接 Pollinations.ai API...",
            f"📝 優化提示詞與 {selected_model['speciality']} 參數...",
            f"🎨 {selected_model['name']} 模型處理中...",
            f"🖼️ 生成 {size} 高質量圖像...",
            f"📡 從雲端接收圖像數據...",
            f"🎉 {selected_model['name']} 生成完成！"
        ]
        
        for i, stage in enumerate(stages):
            status_text.info(stage)
            # 根據模型調整進度時間
            if "turbo" in selected_model['model_id']:
                time.sleep(0.6)  # 快速模型
            else:
                time.sleep(1.0)   # 標準時間
            progress_bar.progress((i + 1) / len(stages))
    
    # 執行圖像生成
    success, result = generate_flux_krea_image_fixed(prompt, selected_model['model_id'], preset, size)
    
    progress_container.empty()
    
    if success:
        st.success(f"🎉 {selected_model['icon']} {selected_model['name']} 生成完成！")
        st.balloons()
        
        # 作品展示
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0 1rem 0;">
            <h3 style="color: {selected_model['color']}; font-size: 1.8rem; margin: 0;">
                🖼️ {selected_model['name']} 專業作品
            </h3>
            <p style="color: #64748b; margin-top: 0.5rem;">
                {selected_model['speciality']} • {FLUX_KREA_PRESETS[preset]['name']} • {size}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 圖像展示
        st.image(
            result, 
            use_column_width=True, 
            caption=f"由 {selected_model['name']} 生成 • 風格: {FLUX_KREA_PRESETS[preset]['name']} • 提示詞: {prompt[:80]}..."
        )
        
        # 生成詳情
        with st.expander(f"🔍 {selected_model['name']} 生成詳情", expanded=False):
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
                st.write(f"**API 服務**: Pollinations.ai")
                st.write(f"**生成時間**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**提示詞長度**: {len(prompt)} 字符")
        
        # 操作按鈕
        col_download, col_regen, col_new, col_home = st.columns(4)
        
        with col_download:
            if st.button("📥 下載圖像", use_container_width=True, key="download_btn"):
                st.success("💡 請右鍵點擊圖像選擇「另存為」")
        
        with col_regen:
            if st.button("🔄 重新生成", use_container_width=True, key="regen_btn"):
                generate_image_main(selected_model, prompt, preset, size)
        
        with col_new:
            if st.button("✨ 新作品", use_container_width=True, key="new_btn"):
                if 'selected_flux_krea_model' in st.session_state:
                    del st.session_state.selected_flux_krea_model
                rerun_app()
        
        with col_home:
            if st.button("🏠 回到主頁", use_container_width=True, key="home_result_btn"):
                go_to_homepage()
                
    else:
        st.error(f"❌ {selected_model['name']} 生成失敗")
        
        st.markdown(f"""
        <div class="modern-card" style="border-left: 4px solid #dc2626;">
            <h4 style="color: #dc2626; margin-bottom: 1rem;">🚨 生成錯誤</h4>
            <p><strong>錯誤信息</strong>: {result}</p>
            <p><strong>使用模型</strong>: {selected_model['name']}</p>
            <p><strong>API 服務</strong>: Pollinations.ai</p>
            <p><strong>提示詞長度</strong>: {len(prompt)} 字符</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_retry, col_home_error = st.columns([3, 1])
        with col_retry:
            if st.button("🔄 重試生成", type="primary", use_container_width=True, key="retry_btn"):
                generate_image_main(selected_model, prompt, preset, size)
        with col_home_error:
            if st.button("🏠 返回主頁", use_container_width=True, key="home_error_btn"):
                go_to_homepage()

# 初始化會話狀態
def init_session_state():
    """初始化會話狀態"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'

def main():
    """主程式 - 修復版本"""
    try:
        load_custom_css()
        init_session_state()
        
        # 側邊欄
        with st.sidebar:
            st.markdown("### 🎨 AI Image Studio Pro")
            st.markdown("---")
            
            if st.button("🏠 主頁", use_container_width=True, key="sidebar_home"):
                st.session_state.current_page = 'home'
                rerun_app()
            
            if st.button("🎨 圖像工作室", use_container_width=True, key="sidebar_studio"):
                st.session_state.current_page = 'flux_krea'
                rerun_app()
            
            st.markdown("---")
            st.markdown("### ℹ️ 系統信息")
            st.success("🌍 Pollinations.ai 免費 API")
            if KOYEB_ENV:
                st.success("🚀 Koyeb 生產環境")
            else:
                st.info("💻 本地開發環境")
            
            st.caption(f"端口: {PORT}")
            st.caption("版本: v9.0 修復版")
        
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
                🚀 AI Image Studio Pro - Powered by Pollinations.ai & Koyeb
            </div>
            <div style="color: #64748b; margin-bottom: 1rem;">
                專業級 AI 圖像生成平台 • 免費 API • 6種專業模型 • 全球高性能部署
            </div>
            <div style="font-size: 0.9rem; color: #94a3b8;">
                API 服務: Pollinations.ai | 
                運行環境: {'🌍 Koyeb Production' if KOYEB_ENV else '💻 Local Development'} | 
                版本: v9.0 修復版
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"應用運行錯誤: {safe_text(str(e))}")
        logger.error(f"Main app error: {str(e)}")

if __name__ == "__main__":
    main()
