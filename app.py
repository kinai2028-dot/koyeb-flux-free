import streamlit as st
import os
import logging
import time
import sqlite3
import uuid
import json
import random
from functools import lru_cache
import urllib.parse

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
    handlers=[logging.StreamHandler()]
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
            logger.info("Requests imported successfully")
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
            logger.info("Base64 imported successfully")
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

# 修復的 FLUX Krea 模型庫 - 正確的模型 ID
FLUX_KREA_MODELS = {
    "flux-krea-dev": {
        "name": "FLUX Krea Dev",
        "model_id": "flux",  # 修復：使用正確的 model ID
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
        "model_id": "flux-pro",  # 修復：使用正確的 model ID
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
        "model_id": "flux-schnell",  # 修復：使用正確的 model ID
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
        "model_id": "flux-realism",  # 修復：使用正確的 model ID
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
        "model_id": "flux-anime",  # 修復：使用正確的 model ID
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
        "model_id": "flux",  # 修復：使用基本 flux 模型
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

# 修復的 FLUX Krea 生成函數
def generate_flux_krea_image(prompt, model_id="flux", preset="realistic", size="1024x1024"):
    """修復的 FLUX Krea 圖像生成"""
    imports = get_heavy_imports()
    
    # 檢查必要的導入
    if not imports.get('requests') or not imports.get('base64'):
        return False, "缺少必要的模組 (requests, base64)"
    
    try:
        # 安全處理提示詞
        prompt = str(prompt).strip()
        if len(prompt) > 500:
            prompt = prompt[:500]
        
        # 應用預設優化
        preset_config = {
            "realistic": {
                "prompt_prefix": "photorealistic, high quality, ",
                "prompt_suffix": ", detailed, sharp focus, professional photography"
            },
            "portrait": {
                "prompt_prefix": "professional portrait, ",
                "prompt_suffix": ", natural lighting, detailed eyes, high resolution"
            },
            "landscape": {
                "prompt_prefix": "beautiful landscape, ",
                "prompt_suffix": ", scenic view, natural colors, high detail"
            },
            "artistic": {
                "prompt_prefix": "artistic composition, ",
                "prompt_suffix": ", creative style, masterpiece quality"
            }
        }.get(preset, {
            "prompt_prefix": "", 
            "prompt_suffix": ", high quality"
        })
        
        # 優化提示詞
        optimized_prompt = f"{preset_config['prompt_prefix']}{prompt}{preset_config['prompt_suffix']}"
        
        # URL 編碼
        encoded_prompt = urllib.parse.quote(optimized_prompt)
        
        # 解析尺寸
        try:
            width, height = map(int, size.split('x'))
        except:
            width, height = 1024, 1024
        
        # 構建 API URL - 修復版本
        base_url = "https://image.pollinations.ai/prompt"
        url_params = [
            f"model={model_id}",
            f"width={width}",
            f"height={height}",
            "nologo=true",
            "enhance=true"  # 添加增強參數
        ]
        
        full_url = f"{base_url}/{encoded_prompt}?{'&'.join(url_params)}"
        
        logger.info(f"FLUX Krea API call: {model_id}, size: {size}")
        
        # 發送請求 - 增加超時和重試
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = imports['requests'].get(full_url, timeout=45, headers=headers)
        response.raise_for_status()
        
        if response.status_code == 200 and response.content:
            # 編碼圖像
            encoded_image = imports['base64'].b64encode(response.content).decode()
            image_url = f"data:image/png;base64,{encoded_image}"
            logger.info("FLUX Krea generation successful")
            return True, image_url
        else:
            error_msg = f"HTTP {response.status_code} - 無內容返回"
            logger.error(f"FLUX Krea API error: {error_msg}")
            return False, error_msg
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"FLUX Krea generation error: {error_msg}")
        return False, error_msg

# 修復的 NavyAI 生成函數
def generate_navyai_image_real(api_key, model_id, prompt, **params):
    """修復的 NavyAI 真實 API 生成"""
    imports = get_heavy_imports()
    
    # 檢查 OpenAI 是否可用
    if not imports.get('OpenAI'):
        logger.warning("OpenAI not available, using fallback")
        return generate_navyai_image_fallback(api_key, model_id, prompt, **params)
    
    try:
        # 安全處理參數
        prompt = str(prompt).strip()
        if len(prompt) > 1000:
            prompt = prompt[:1000]
            
        api_model = params.get('api_model', 'dall-e-3')
        size = params.get('size', '1024x1024')
        num_images = min(int(params.get('num_images', 1)), 4)
        
        logger.info(f"NavyAI API call: model={api_model}")
        
        # 創建 OpenAI 客戶端
        client = imports['OpenAI'](
            api_key=api_key,
            base_url="https://api.navy/v1",
            timeout=60  # 增加超時時間
        )
        
        # API 調用
        response = client.images.generate(
            model=api_model,
            prompt=prompt,
            n=num_images,
            size=size,
            quality="standard"
        )
        
        # 處理回應
        if response.data and len(response.data) > 0:
            image_data = response.data[0]
            
            # 檢查不同的回應格式
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                image_url = f"data:image/png;base64,{image_data.b64_json}"
                logger.info("NavyAI API call successful (b64_json)")
                return True, image_url
            elif hasattr(image_data, 'url') and image_data.url:
                # 如果返回的是 URL，下載圖像
                if imports.get('requests'):
                    img_response = imports['requests'].get(image_data.url, timeout=30)
                    if img_response.status_code == 200:
                        encoded_image = imports['base64'].b64encode(img_response.content).decode()
                        image_url = f"data:image/png;base64,{encoded_image}"
                        logger.info("NavyAI API call successful (url)")
                        return True, image_url
                return False, "無法下載圖像"
            else:
                logger.error("NavyAI API response format unknown")
                return generate_navyai_image_fallback(api_key, model_id, prompt, **params)
        else:
            logger.error("NavyAI API response empty")
            return generate_navyai_image_fallback(api_key, model_id, prompt, **params)
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"NavyAI API error: {error_msg}")
        return generate_navyai_image_fallback(api_key, model_id, prompt, **params)

def generate_navyai_image_fallback(api_key, model_id, prompt, **params):
    """NavyAI 回退生成 - 修復版本"""
    imports = get_heavy_imports()
    
    # 檢查必要的模組
    if not imports.get('Image') or not imports.get('base64') or not imports.get('BytesIO'):
        return False, "缺少圖像處理模組"
    
    try:
        logger.info("Using NavyAI fallback mode")
        
        # 模擬生成時間
        time.sleep(2)
        
        # 安全處理參數
        prompt = str(prompt).strip()
        if len(prompt) > 500:
            prompt = prompt[:500] + "..."
            
        try:
            width, height = map(int, params.get('size', '1024x1024').split('x'))
        except:
            width, height = 1024, 1024
        
        # 創建演示圖像
        img = imports['Image'].new('RGB', (width, height), color='#f0f8ff')
        draw = imports['ImageDraw'].Draw(img)
        
        # 創建漸變背景
        for y in range(height):
            r = int(240 + (15 * y / height))
            g = int(248 + (7 * y / height))  
            b = int(255)
            draw.line([(0, y), (width, y)], fill=(min(255, r), min(255, g), b))
        
        # 添加文字（使用默認字體）
        try:
            font = imports['ImageFont'].load_default()
        except:
            font = None
        
        # 添加標題和信息
        draw.text((50, 50), "NavyAI Demo Generation", fill=(50, 50, 150), font=font)
        draw.text((50, 100), f"Model: {model_id}", fill=(100, 100, 100), font=font)
        
        # 添加提示詞預覽
        prompt_lines = [prompt[i:i+40] for i in range(0, len(prompt), 40)]
        for i, line in enumerate(prompt_lines[:3]):
            draw.text((50, 150 + i*30), f"Prompt: {line}", fill=(80, 80, 80), font=font)
        
        # 添加狀態信息
        draw.text((50, height - 100), "Demo Mode - NavyAI Fallback", fill=(255, 100, 0), font=font)
        draw.text((50, height - 70), "Koyeb High-Performance Deploy", fill=(0, 100, 200), font=font)
        draw.text((50, height - 40), "Configure real API key for actual generation", fill=(150, 150, 150), font=font)
        
        # 轉換為 base64
        buffer = imports['BytesIO']()
        img.save(buffer, format='PNG')
        encoded_image = imports['base64'].b64encode(buffer.getvalue()).decode()
        
        logger.info("NavyAI fallback generation successful")
        return True, f"data:image/png;base64,{encoded_image}"
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"NavyAI fallback generation error: {error_msg}")
        return False, error_msg

# 其餘代碼保持不變，但需要確保主函數正確調用
def main():
    """主程式 - 修復版本"""
    try:
        # 檢查依賴是否載入成功
        imports = get_heavy_imports()
        
        if KOYEB_ENV:
            st.success("🚀 應用正在 Koyeb 高性能平台運行")
        
        # 顯示應用標題
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%); border-radius: 10px; margin-bottom: 1.5rem;">
            <h1 style="color: white; margin: 0; font-size: 2.2rem;">🎨 AI 圖像生成器 Pro</h1>
            <h2 style="color: #dbeafe; margin: 0.3rem 0; font-size: 1.1rem;">FLUX Krea 6種模型 + NavyAI 真實API調用</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # 檢查依賴狀態
        st.markdown("### 🔧 系統狀態檢查")
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
            
            # 簡化的功能選擇
            st.markdown("### 🎯 選擇 AI 圖像生成服務")
            
            col_flux, col_navy = st.columns(2)
            
            with col_flux:
                st.markdown("""
                #### 🎭 FLUX Krea AI (免費)
                - ✅ 6種 FLUX Krea 模型
                - 🎨 多種美學預設
                - ⚡ 即時生成
                - 🆓 完全免費
                """)
                
                if st.button("🎭 使用 FLUX Krea", type="primary", use_container_width=True):
                    test_flux_generation()
            
            with col_navy:
                st.markdown("""
                #### ⚓ NavyAI (真實API)
                - 🖼️ DALL-E 2/3
                - 🔗 真實雲端生成
                - 🛡️ 自動回退保護
                - 💰 按使用付費
                """)
                
                api_key = st.text_input("NavyAI API Key:", type="password", placeholder="輸入您的 API 密鑰")
                if st.button("⚓ 使用 NavyAI", use_container_width=True, disabled=not api_key):
                    if api_key:
                        test_navy_generation(api_key)
        else:
            st.error("⚠️ 部分功能不可用，請檢查依賴安裝")
            st.markdown("#### 📋 請確保 requirements.txt 包含：")
            st.code("""streamlit>=1.28.0
openai>=1.0.0
Pillow>=10.0.0
requests>=2.31.0""")
    
    except Exception as e:
        st.error(f"應用運行錯誤: {str(e)}")
        logger.error(f"Main app error: {str(e)}")

def test_flux_generation():
    """測試 FLUX Krea 生成"""
    st.markdown("### 🎭 FLUX Krea 測試生成")
    
    prompt = st.text_area("輸入提示詞:", value="a beautiful sunset over mountains", height=100)
    
    col_model, col_preset, col_size = st.columns(3)
    
    with col_model:
        model_options = list(FLUX_KREA_MODELS.keys())
        selected_model_key = st.selectbox("選擇模型:", model_options)
        selected_model = FLUX_KREA_MODELS[selected_model_key]
    
    with col_preset:
        preset = st.selectbox("美學預設:", ["realistic", "portrait", "landscape", "artistic"])
    
    with col_size:
        size = st.selectbox("尺寸:", ["512x512", "1024x1024"])
    
    if st.button("🎨 生成圖像", type="primary"):
        with st.spinner(f"使用 {selected_model['name']} 生成中..."):
            success, result = generate_flux_krea_image(prompt, selected_model['model_id'], preset, size)
            
            if success:
                st.success("✅ 生成成功！")
                st.image(result, caption=f"{selected_model['name']} - {prompt}", use_column_width=True)
            else:
                st.error(f"❌ 生成失敗: {result}")

def test_navy_generation(api_key):
    """測試 NavyAI 生成"""
    st.markdown("### ⚓ NavyAI 測試生成")
    
    prompt = st.text_area("輸入提示詞:", value="a cute cat wearing a wizard hat", height=100)
    
    col_model, col_size = st.columns(2)
    
    with col_model:
        api_model = st.selectbox("API 模型:", ["dall-e-3", "dall-e-2"])
    
    with col_size:
        if api_model == "dall-e-3":
            size_options = ["1024x1024", "1024x1792", "1792x1024"]
        else:
            size_options = ["256x256", "512x512", "1024x1024"]
        size = st.selectbox("尺寸:", size_options)
    
    if st.button("🎨 真實 API 生成", type="primary"):
        with st.spinner(f"使用 NavyAI {api_model} 生成中..."):
            success, result = generate_navyai_image_real(
                api_key, api_model, prompt, 
                api_model=api_model, size=size, num_images=1
            )
            
            if success:
                st.success("✅ NavyAI 生成成功！")
                st.image(result, caption=f"NavyAI {api_model} - {prompt}", use_column_width=True)
            else:
                st.error(f"❌ NavyAI 生成失敗: {result}")

if __name__ == "__main__":
    main()
