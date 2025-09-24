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
from urllib.parse import urlencode, quote
import hashlib
from cryptography.fernet import Fernet

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
    page_title="Flux AI 圖像生成器 Pro - 專業美學與藝術風格",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 藝術風格庫 - 根據最新AI藝術趨勢整理
ARTISTIC_STYLES = {
    "🎨 經典藝術風格": {
        "Oil Painting": "rich colors, visible brushstrokes, traditional oil painting technique",
        "Watercolor": "soft, fluid appearance, watercolor painting style, delicate transparency",
        "Pencil Sketch": "hand-drawn appearance, pencil sketch style, visible lines and shading",
        "Charcoal Drawing": "dramatic contrast, charcoal drawing style, deep blacks and soft grays",
        "Pastel Art": "soft, delicate appearance, pastel colors, subtle shading",
        "Acrylic Painting": "vibrant colors, bold brushstrokes, acrylic paint texture",
        "Gouache": "opaque watercolor style, matte finish, rich pigments"
    },
    "🖼️ 藝術運動風格": {
        "Impressionism": "loose brushwork, light and color emphasis, impressionist style",
        "Cubism": "geometric forms, multiple perspectives, cubist art style",
        "Surrealism": "dreamlike imagery, surreal elements, subconscious themes",
        "Pop Art": "bold colors, popular culture themes, commercial art style",
        "Abstract Expressionism": "abstract forms, emotional expression, gestural brushwork",
        "Art Nouveau": "organic forms, decorative elements, elegant curves",
        "Baroque": "ornate details, dramatic lighting, rich colors, grandeur",
        "Renaissance": "precise detail, realism, classical composition, sfumato technique"
    },
    "📸 攝影風格": {
        "Portrait Photography": "professional portrait style, dramatic lighting, shallow depth of field",
        "Landscape Photography": "natural landscape style, golden hour lighting, wide angle view",
        "Street Photography": "candid moments, urban environment, documentary style",
        "Fashion Photography": "high fashion style, dramatic poses, professional lighting",
        "Macro Photography": "extreme close-up, fine details, shallow focus",
        "Black and White": "monochrome photography, dramatic contrast, timeless appeal",
        "Vintage Photography": "retro aesthetic, film grain, aged appearance",
        "Cinematic": "movie-like quality, dramatic lighting, wide screen composition"
    },
    "🎭 現代數位風格": {
        "Digital Art": "clean digital illustration, smooth gradients, modern aesthetic",
        "Pixel Art": "retro gaming style, square pixels, limited color palette",
        "Voxel Art": "3D pixel art, cubic forms, isometric perspective",
        "Low Poly": "geometric faceted style, minimalist 3D forms",
        "Synthwave": "neon colors, retro-futuristic aesthetic, 1980s vibe",
        "Cyberpunk": "neon lights, dark urban setting, high-tech low-life aesthetic",
        "Steampunk": "Victorian era meets steam technology, brass and copper tones",
        "Vaporwave": "pastel colors, retro aesthetics, dreamy atmosphere"
    },
    "🌟 特殊效果風格": {
        "Psychedelic": "vibrant colors, swirling patterns, surreal imagery",
        "Glitch Art": "digital distortion, data corruption aesthetic, fragmented imagery",
        "Double Exposure": "overlay effects, transparent blending, artistic composition",
        "Long Exposure": "motion blur, light trails, smooth water effects",
        "HDR": "high dynamic range, enhanced colors, detailed shadows and highlights",
        "Tilt-Shift": "miniature effect, selective focus, toy-like appearance",
        "Cross Processing": "alternative color processing, vintage film look",
        "Lomography": "toy camera aesthetic, light leaks, saturated colors"
    },
    "🎬 電影風格": {
        "Film Noir": "high contrast, dramatic shadows, monochromatic mood",
        "Wes Anderson": "symmetrical composition, pastel colors, whimsical aesthetic",
        "Christopher Nolan": "dark, complex imagery, dramatic lighting",
        "Studio Ghibli": "animated film style, soft colors, magical realism",
        "Tim Burton": "gothic aesthetic, dark whimsy, exaggerated proportions",
        "Blade Runner": "dystopian future, neon-lit streets, rain-soaked atmosphere",
        "Mad Max": "post-apocalyptic, desert wasteland, gritty textures",
        "Matrix": "green tint, digital rain effect, high contrast"
    },
    "🗾 文化藝術風格": {
        "Ukiyo-e": "Japanese woodblock print style, flat colors, elegant lines",
        "Chinese Ink Painting": "traditional brush painting, flowing ink, minimalist composition",
        "Islamic Art": "geometric patterns, intricate designs, calligraphic elements",
        "Aboriginal Art": "dot painting, earth tones, spiritual symbols",
        "Mexican Folk Art": "vibrant colors, traditional patterns, cultural motifs",
        "Art Deco": "geometric patterns, luxury materials, streamlined forms",
        "Tribal Art": "primitive forms, bold patterns, earth pigments",
        "Celtic Art": "intricate knots, spiral patterns, mystical themes"
    }
}

# API 提供商配置
API_PROVIDERS = {
    "Navy": {
        "name": "Navy API",
        "base_url_default": "https://api.navy/v1",
        "key_prefix": "sk-",
        "description": "Navy 提供的 AI 圖像生成服務",
        "icon": "⚓"
    },
    "FLUX Krea AI Studio": {
        "name": "FLUX Krea AI Studio",
        "base_url_default": "https://api.krea.ai/v1",
        "key_prefix": "krea_",
        "description": "專業美學圖像生成平台，提供高品質的FLUX模型系列，解決AI生成圖像的「AI感」問題",
        "icon": "🎨",
        "auth_modes": ["api_key"],
        "models": ["flux-default", "flux-1.1-pro", "flux-1.1-pro-ultra", "flux-kontext-pro"],
        "registration_url": "https://www.krea.ai/apps/image/flux-krea",
        "features": ["style_references", "aspect_ratio_control", "batch_generation", "image_editing"],
        "specialties": ["anti_ai_aesthetic", "professional_photography", "style_control"]
    },
    "Pollinations.ai": {
        "name": "Pollinations.ai",
        "base_url_default": "https://image.pollinations.ai",
        "key_prefix": "",
        "description": "支援免費和認證模式的圖像生成 API",
        "icon": "🌸",
        "auth_modes": ["free", "referrer", "token"]
    },
    "Hugging Face": {
        "name": "Hugging Face Inference",
        "base_url_default": "https://api-inference.huggingface.co",
        "key_prefix": "hf_",
        "description": "Hugging Face Inference API",
        "icon": "🤗"
    },
    "Custom": {
        "name": "自定義 API",
        "base_url_default": "",
        "key_prefix": "",
        "description": "自定義的 API 端點",
        "icon": "🔧"
    }
}

# 基礎 Flux 模型配置
BASE_FLUX_MODELS = {
    "flux.1-schnell": {
        "name": "FLUX.1 Schnell",
        "description": "最快的生成速度，開源模型",
        "icon": "⚡",
        "type": "快速生成",
        "test_prompt": "A simple cat sitting on a table",
        "expected_size": "1024x1024",
        "priority": 1,
        "source": "base",
        "auth_required": False
    },
    "flux.1-dev": {
        "name": "FLUX.1 Dev",
        "description": "開發版本，平衡速度與質量",
        "icon": "🔧",
        "type": "開發版本",
        "test_prompt": "A beautiful landscape with mountains",
        "expected_size": "1024x1024",
        "priority": 2,
        "source": "base",
        "auth_required": False
    },
    "flux.1.1-pro": {
        "name": "FLUX.1.1 Pro",
        "description": "改進的旗艦模型，最佳品質",
        "icon": "👑",
        "type": "旗艦版本",
        "test_prompt": "Professional portrait of a person in business attire",
        "expected_size": "1024x1024",
        "priority": 3,
        "source": "base",
        "auth_required": False
    },
    "flux.1-kontext-pro": {
        "name": "FLUX.1 Kontext Pro",
        "description": "支持圖像編輯和上下文理解（需認證）",
        "icon": "🎯",
        "type": "編輯專用",
        "test_prompt": "Abstract geometric shapes in vibrant colors",
        "expected_size": "1024x1024",
        "priority": 4,
        "source": "base",
        "auth_required": True
    },
    # FLUX Krea AI Studio 專用模型
    "flux-krea-default": {
        "name": "FLUX Krea (Default)",
        "description": "專為Krea最佳化的快速高品質模型，適合風格參考和專業美學",
        "icon": "🎨",
        "type": "美學專用",
        "test_prompt": "A linocut illustration of a forest clearing, with soft natural light and warm earthy tones",
        "expected_size": "1024x1024",
        "priority": 5,
        "source": "krea_studio",
        "auth_required": True,
        "api_endpoint": "/images/generations",
        "supports_style_reference": True,
        "provider": "FLUX Krea AI Studio"
    },
    "flux-1.1-pro-krea": {
        "name": "FLUX 1.1 Pro (Krea)",
        "description": "Black Forest Labs進階高效模型，Krea優化版本，專業攝影級品質",
        "icon": "👑",
        "type": "專業版本",
        "test_prompt": "Professional portrait photography with dramatic lighting and shallow depth of field",
        "expected_size": "1024x1024",
        "priority": 6,
        "source": "krea_studio",
        "auth_required": True,
        "api_endpoint": "/images/generations",
        "supports_style_reference": True,
        "provider": "FLUX Krea AI Studio"
    },
    "flux-kontext-pro-krea": {
        "name": "FLUX Kontext Pro (Krea)",
        "description": "前沿圖像編輯模型，支援高級推理和風格轉換，解決AI感問題",
        "icon": "🧠",
        "type": "編輯專業",
        "test_prompt": "A surreal digital art piece with complex visual elements and natural aesthetics",
        "expected_size": "1024x1024",
        "priority": 7,
        "source": "krea_studio",
        "auth_required": True,
        "api_endpoint": "/images/generations",
        "supports_image_editing": True,
        "supports_style_transfer": True,
        "provider": "FLUX Krea AI Studio"
    },
    "flux-1.1-pro-ultra-krea": {
        "name": "FLUX 1.1 Pro Ultra (Krea)",
        "description": "Krea最頂級模型，極致品質與細節，專業商用級別",
        "icon": "💎",
        "type": "頂級版本",
        "test_prompt": "Ultra high-quality professional commercial photography with perfect lighting",
        "expected_size": "1024x1024",
        "priority": 8,
        "source": "krea_studio",
        "auth_required": True,
        "api_endpoint": "/images/generations",
        "supports_style_reference": True,
        "supports_ultra_quality": True,
        "provider": "FLUX Krea AI Studio"
    }
}

# 加密功能
def get_encryption_key():
    """獲取或生成加密密鑰"""
    key_file = '.app_key'
    if os.path.exists(key_file):
        with open(key_file, 'rb') as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        with open(key_file, 'wb') as f:
            f.write(key)
        return key

def encrypt_api_key(api_key: str) -> str:
    """加密API密鑰"""
    if not api_key:
        return ""
    key = get_encryption_key()
    f = Fernet(key)
    encrypted_key = f.encrypt(api_key.encode())
    return base64.b64encode(encrypted_key).decode()

def decrypt_api_key(encrypted_key: str) -> str:
    """解密API密鑰"""
    if not encrypted_key:
        return ""
    try:
        key = get_encryption_key()
        f = Fernet(key)
        decrypted_key = f.decrypt(base64.b64decode(encrypted_key.encode()))
        return decrypted_key.decode()
    except:
        return ""

def save_api_keys_to_file():
    """保存API密鑰到本地文件"""
    try:
        config = st.session_state.api_config
        keys_data = {
            'provider': config.get('provider', ''),
            'base_url': config.get('base_url', ''),
            'encrypted_key': encrypt_api_key(config.get('api_key', '')),
            'validated': config.get('validated', False),
            'saved_at': datetime.datetime.now().isoformat()
        }
        
        # 保存到JSON文件
        with open('.api_keys.json', 'w') as f:
            json.dump(keys_data, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"保存API密鑰失敗: {str(e)}")
        return False

def load_api_keys_from_file():
    """從本地文件載入API密鑰"""
    try:
        if os.path.exists('.api_keys.json'):
            with open('.api_keys.json', 'r') as f:
                keys_data = json.load(f)
            
            return {
                'provider': keys_data.get('provider', 'Navy'),
                'base_url': keys_data.get('base_url', 'https://api.navy/v1'),
                'api_key': decrypt_api_key(keys_data.get('encrypted_key', '')),
                'validated': keys_data.get('validated', False),
                'saved_at': keys_data.get('saved_at', '')
            }
    except Exception as e:
        st.warning(f"載入API密鑰失敗: {str(e)}")
    
    return None

def apply_artistic_style(prompt: str, style_desc: str) -> str:
    """將藝術風格應用到提示詞"""
    if not style_desc or style_desc == "無風格":
        return prompt
    
    # 優化的風格整合方式
    if "," in prompt:
        # 如果prompt已有逗號，在適當位置插入風格
        parts = prompt.split(",", 1)
        return f"{parts[0].strip()}, {style_desc}, {parts[1].strip()}"
    else:
        # 簡單prompt，直接添加風格
        return f"{prompt}, {style_desc}"

def show_artistic_styles():
    """顯示藝術風格選擇界面"""
    st.markdown("### 🎨 藝術風格選擇")
    
    # 風格分類選擇
    style_categories = list(ARTISTIC_STYLES.keys())
    selected_category = st.selectbox(
        "選擇風格分類",
        ["無風格"] + style_categories,
        key="style_category"
    )
    
    selected_style_desc = ""
    
    if selected_category != "無風格":
        styles_in_category = ARTISTIC_STYLES[selected_category]
        style_names = list(styles_in_category.keys())
        
        selected_style = st.selectbox(
            f"選擇 {selected_category} 風格",
            style_names,
            key="style_name"
        )
        
        selected_style_desc = styles_in_category[selected_style]
        
        # 顯示風格描述
        st.info(f"**{selected_style}**: {selected_style_desc}")
        
        # 風格強度調整
        style_strength = st.slider(
            "風格強度",
            0.1, 2.0, 1.0, 0.1,
            key="style_strength",
            help="調整風格在最終圖像中的影響程度"
        )
        
        if style_strength != 1.0:
            # 根據強度調整風格描述
            if style_strength < 1.0:
                selected_style_desc = f"subtle {selected_style_desc}"
            else:
                selected_style_desc = f"strong {selected_style_desc}, highly stylized"
    
    return selected_style_desc

def validate_krea_api_key(api_key: str, base_url: str) -> Tuple[bool, str]:
    """驗證Krea AI API密鑰"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 測試API連接
        test_url = f"{base_url}/models"
        response = requests.get(test_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            model_count = len(models_data) if isinstance(models_data, list) else 0
            return True, f"Krea AI API連接成功，發現 {model_count} 個可用模型"
        elif response.status_code == 401:
            return False, "API密鑰無效或已過期，請檢查您的Krea AI帳戶"
        elif response.status_code == 403:
            return False, "API密鑰權限不足，請升級您的Krea AI訂閱"
        elif response.status_code == 429:
            return False, "請求過於頻繁，請稍後再試"
        else:
            return False, f"HTTP {response.status_code}: Krea AI連接失敗"
            
    except requests.exceptions.Timeout:
        return False, "Krea AI API連接超時，請檢查網路連接"
    except requests.exceptions.ConnectionError:
        return False, "無法連接到Krea AI服務器，請檢查網路設置"
    except Exception as e:
        return False, f"Krea AI API驗證失敗: {str(e)}"

def generate_images_krea(api_key: str, base_url: str, **params) -> Tuple[bool, any]:
    """Krea AI Studio專用的圖像生成函數"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "FLUX-Krea-Client/1.0"
        }
        
        # 準備Krea API請求參數
        krea_params = {
            "model": params.get("model", "flux-default").replace("-krea", ""),
            "prompt": params.get("prompt", ""),
            "width": int(params.get("size", "1024x1024").split('x')[0]),
            "height": int(params.get("size", "1024x1024").split('x')[1]),
            "num_images": params.get("n", 1),
            "guidance_scale": params.get("guidance_scale", 7.5),
            "num_inference_steps": params.get("steps", 50),
            "seed": params.get("seed", random.randint(0, 1000000)),
            "safety_checker": True,
            "enhance_prompt": True  # Krea特有功能
        }
        
        # 支援風格參考（Krea特色功能）
        if params.get("style_reference_url"):
            krea_params["style_reference"] = {
                "image_url": params["style_reference_url"],
                "strength": params.get("style_strength", 0.7)
            }
        
        # 支援長寬比控制
        aspect_ratio = params.get("aspect_ratio")
        if aspect_ratio:
            aspect_ratios = {
                "1:1": (1024, 1024),
                "16:9": (1344, 768),
                "9:16": (768, 1344),
                "4:3": (1152, 896),
                "3:4": (896, 1152),
                "21:9": (1536, 640),  # 電影比例
                "3:2": (1152, 768),   # 攝影比例
                "2:3": (768, 1152)
            }
            if aspect_ratio in aspect_ratios:
                width, height = aspect_ratios[aspect_ratio]
                krea_params.update({"width": width, "height": height})
        
        # 發送請求到Krea API
        response = requests.post(
            f"{base_url}/images/generations",
            headers=headers,
            json=krea_params,
            timeout=180  # Krea可能需要更長時間
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # 模擬OpenAI響應格式
            class MockResponse:
                def __init__(self, images_data):
                    self.data = []
                    for img_data in images_data:
                        img_obj = type('obj', (object,), {
                            'url': img_data.get('url', ''),
                            'revised_prompt': img_data.get('enhanced_prompt', img_data.get('prompt', '')),
                            'krea_metadata': {
                                'model_version': img_data.get('model_version', ''),
                                'generation_id': img_data.get('id', ''),
                                'aesthetic_score': img_data.get('aesthetic_score', 0)
                            }
                        })()
                        self.data.append(img_obj)
            
            images = result.get('images', [])
            if not images:
                return False, "Krea API返回空的圖像列表"
                
            return True, MockResponse(images)
            
        elif response.status_code == 400:
            error_detail = response.json().get('error', {})
            error_msg = error_detail.get('message', '請求參數錯誤')
            return False, f"Krea API參數錯誤: {error_msg}"
        elif response.status_code == 402:
            return False, "Krea AI額度不足，請檢查您的訂閱狀態"
        elif response.status_code == 429:
            return False, "Krea AI請求過於頻繁，請稍後再試"
        elif response.status_code == 503:
            return False, "Krea AI服務暫時不可用，請稍後重試"
        else:
            error_msg = response.json().get('error', {}).get('message', '未知錯誤')
            return False, f"Krea API錯誤 ({response.status_code}): {error_msg}"
            
    except requests.exceptions.Timeout:
        return False, "Krea AI圖像生成超時，請嘗試減少圖像數量或降低品質"
    except requests.exceptions.ConnectionError:
        return False, "無法連接到Krea AI服務，請檢查網路連接"
    except json.JSONDecodeError:
        return False, "Krea API返回格式錯誤，請稍後重試"
    except Exception as e:
        return False, f"Krea AI圖像生成失敗: {str(e)}"

def show_krea_settings(selected_provider, st):
    """顯示Krea AI Studio專用設置選項"""
    if selected_provider == "FLUX Krea AI Studio":
        st.markdown("### 🎨 FLUX Krea AI Studio 設置")
        
        # API密鑰註冊說明
        with st.expander("🔑 API密鑰獲取指南", expanded=True):
            st.markdown("""
            **📋 註冊和設置步驟：**
            
            1. **訪問官網**：前往 [krea.ai](https://www.krea.ai) 註冊帳戶
            2. **郵箱驗證**：檢查郵箱並完成帳戶驗證
            3. **選擇方案**：
               - 🆓 **免費方案**：每月限額，基礎功能
               - 💎 **Pro方案**：無限生成，高級功能
               - 🏢 **企業方案**：API存取，商業授權
            4. **生成密鑰**：帳戶設置 → API密鑰 → 創建新密鑰
            5. **複製密鑰**：將API密鑰粘貼到下方輸入框
            
            **📞 技術支援：**
            - 📧 支援郵箱：support@krea.ai
            - 📱 Discord社群：[加入Krea社群](https://discord.gg/krea)
            - 📚 文檔：[API文檔](https://www.krea.ai/docs/api)
            """)
        
        # 專業功能介紹
        with st.expander("✨ Krea AI Studio 專業功能"):
            st.markdown("""
            **🎨 專業美學特色：**
            - **反AI美學**：專門解決生成圖像的「AI感」問題
            - **風格參考控制**：上傳參考圖像，精確控制風格轉換
            - **專業攝影模式**：模擬真實攝影效果和光影
            - **智能提示詞增強**：自動優化提示詞獲得更佳效果
            
            **🏆 模型優勢：**
            - **FLUX Default**：6秒快速生成，專為風格工作最佳化
            - **FLUX 1.1 Pro**：Black Forest Labs官方進階版本
            - **FLUX Kontext Pro**：支援圖像編輯和智能風格轉換
            - **FLUX Pro Ultra**：頂級品質，商業用途級別
            
            **🔧 技術特點：**
            - **高解析度支援**：最高4K輸出品質
            - **批量處理**：支援同時生成多張變體
            - **風格一致性**：確保系列圖像風格統一
            - **商業授權**：Pro訂閱包含商業使用權
            """)
        
        # 風格設置選項
        st.markdown("#### 🎭 高級風格控制")
        
        col1, col2 = st.columns(2)
        with col1:
            enable_style_ref = st.checkbox(
                "啟用風格參考圖像", 
                value=st.session_state.get('krea_style_ref', False),
                key="krea_style_ref",
                help="上傳參考圖像來控制生成風格"
            )
            if enable_style_ref:
                style_strength = st.slider(
                    "風格強度", 
                    0.1, 1.0, 0.7, 0.1,
                    key="krea_style_strength",
                    help="控制參考風格的影響程度"
                )
                style_url = st.text_input(
                    "風格參考圖像URL",
                    key="krea_style_url",
                    placeholder="https://example.com/reference-image.jpg"
                )
        
        with col2:
            aspect_ratio_mode = st.selectbox(
                "長寬比模式",
                ["標準尺寸", "攝影比例", "電影比例", "社交媒體", "自定義"],
                key="krea_aspect_mode",
                help="選擇適合的圖像比例"
            )
            
            aspect_ratio = "1:1"
            if aspect_ratio_mode == "攝影比例":
                aspect_ratio = st.selectbox(
                    "攝影比例",
                    ["3:2", "2:3", "4:3", "3:4"],
                    key="photo_ratio"
                )
            elif aspect_ratio_mode == "電影比例":
                aspect_ratio = st.selectbox(
                    "電影比例", 
                    ["21:9", "16:9", "9:16"],
                    key="cinema_ratio"
                )
            elif aspect_ratio_mode == "社交媒體":
                aspect_ratio = st.selectbox(
                    "社交媒體比例",
                    ["1:1", "9:16", "4:5"],
                    key="social_ratio"
                )
        
        # 品質設置
        st.markdown("#### ⚙️ 生成參數調整")
        col3, col4 = st.columns(2)
        with col3:
            guidance_scale = st.slider(
                "提示詞引導強度", 
                1.0, 20.0, 7.5, 0.5,
                key="krea_guidance",
                help="提高數值讓模型更嚴格遵循提示詞"
            )
        with col4:
            inference_steps = st.slider(
                "推理步數", 
                20, 100, 50, 5,
                key="krea_steps",
                help="更多步數通常產生更好品質，但速度較慢"
            )
        
        # 返回設置字典
        settings = {
            "enable_style_ref": enable_style_ref,
            "style_strength": style_strength if enable_style_ref else 0.7,
            "style_url": style_url if enable_style_ref else "",
            "aspect_ratio_mode": aspect_ratio_mode,
            "aspect_ratio": aspect_ratio,
            "guidance_scale": guidance_scale,
            "steps": inference_steps
        }
        
        return settings
    
    return {}

# [其他輔助函數保持不變，包括模型發現、驗證、生成等...]
def auto_discover_flux_models(client, provider: str, api_key: str, base_url: str) -> Dict[str, Dict]:
    """自動發現模型"""
    discovered_models = {}
    
    try:
        if provider == "FLUX Krea AI Studio":
            headers = {"Authorization": f"Bearer {api_key}"}
            models_url = f"{base_url}/models"
            response = requests.get(models_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                models_list = response.json()
                for model_data in models_list:
                    model_id = model_data.get('id', model_data.get('name', ''))
                    if 'flux' in model_id.lower():
                        model_info = {
                            "name": model_id.replace('-', ' ').title(),
                            "icon": "🎨",
                            "type": "Krea專用",
                            "description": f"Krea AI Studio模型: {model_id}",
                            "priority": 100,
                            "source": "krea_studio",
                            "auth_required": True,
                            "provider": "FLUX Krea AI Studio"
                        }
                        discovered_models[model_id] = model_info
        elif provider == "Pollinations.ai":
            models_url = f"{base_url}/models"
            response = requests.get(models_url, timeout=10)
            
            if response.status_code == 200:
                models_list = response.json()
                for model_name in models_list:
                    model_info = {
                        "name": model_name.replace('-', ' ').title(),
                        "icon": "🌸",
                        "type": "Pollinations",
                        "description": f"Pollinations模型: {model_name}",
                        "priority": 200,
                        "source": "pollinations",
                        "auth_required": False
                    }
                    discovered_models[model_name] = model_info
        
        return discovered_models
        
    except Exception as e:
        st.warning(f"模型自動發現失敗: {str(e)}")
        return {}

def merge_models() -> Dict[str, Dict]:
    """合併基礎模型和自動發現的模型"""
    discovered = st.session_state.get('discovered_models', {})
    merged_models = BASE_FLUX_MODELS.copy()

    for model_id, model_info in discovered.items():
        if model_id not in merged_models:
            merged_models[model_id] = model_info

    # 按 'priority' 排序
    sorted_models = sorted(merged_models.items(), key=lambda item: item[1].get('priority', 999))
    return dict(sorted_models)

def validate_api_key(api_key: str, base_url: str, provider: str) -> Tuple[bool, str]:
    """驗證 API 密鑰是否有效"""
    try:
        if provider == "FLUX Krea AI Studio":
            return validate_krea_api_key(api_key, base_url)
        elif provider == "Pollinations.ai":
            test_url = f"{base_url}/models"
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                return True, "Pollinations.ai 服務連接成功"
            else:
                return False, f"HTTP {response.status_code}: Pollinations.ai 連接失敗"
        elif provider == "Hugging Face":
            headers = {"Authorization": f"Bearer {api_key}"}
            test_url = f"{base_url}/models/black-forest-labs/FLUX.1-schnell"
            response = requests.get(test_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return True, "Hugging Face API 密鑰驗證成功"
            else:
                return False, f"HTTP {response.status_code}: 驗證失敗"
        else:
            # Navy 和 Custom API 使用 OpenAI 兼容格式
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

def generate_images_with_retry(client, provider: str, api_key: str, base_url: str, **params) -> Tuple[bool, any]:
    """帶重試機制的圖像生成"""
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                st.info(f"🔄 嘗試重新生成 (第 {attempt + 1}/{max_retries} 次)")
                
            if provider == "FLUX Krea AI Studio":
                config = st.session_state.get('api_config', {})
                
                # 添加Krea特定參數
                krea_params = params.copy()
                krea_params.update({
                    'guidance_scale': config.get('krea_guidance', 7.5),
                    'steps': config.get('krea_steps', 50),
                    'aspect_ratio': config.get('krea_aspect_mode', '1:1')
                })
                
                if config.get('krea_style_ref') and config.get('krea_style_url'):
                    krea_params.update({
                        'style_reference_url': config['krea_style_url'],
                        'style_strength': config.get('krea_style_strength', 0.7)
                    })
                
                success, response = generate_images_krea(
                    api_key, base_url, **krea_params
                )
                if success:
                    return True, response
                else:
                    raise Exception(response)
                    
            elif provider == "Pollinations.ai":
                # Pollinations.ai GET 請求
                prompt = params.get("prompt", "")
                width, height = params.get("size", "1024x1024").split('x')
                query_params = {
                    "model": params.get("model"),
                    "width": width,
                    "height": height,
                    "seed": random.randint(0, 1000000),
                    "nologo": "true"
                }

                query_params = {k: v for k, v in query_params.items() if v is not None}

                headers = {}
                config = st.session_state.get('api_config', {})
                auth_mode = config.get('pollinations_auth_mode', 'free')

                if auth_mode == 'token' and config.get('pollinations_token'):
                    headers['Authorization'] = f"Bearer {config['pollinations_token']}"
                elif auth_mode == 'referrer' and config.get('pollinations_referrer'):
                    headers['Referer'] = config['pollinations_referrer']

                encoded_prompt = quote(prompt)
                request_url = f"{base_url}/prompt/{encoded_prompt}?{urlencode(query_params)}"

                response = requests.get(request_url, headers=headers, timeout=120)

                if response.status_code == 200:
                    class MockResponse:
                        def __init__(self, image_data):
                            self.data = [type('obj', (object,), {
                                'url': f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                            })()]

                    return True, MockResponse(response.content)
                else:
                    error_text = response.text
                    if "Access to" in error_text and "is limited" in error_text:
                        return False, f"此模型需要認證。請在側邊欄配置 Pollinations.ai 認證信息。錯誤: {error_text}"
                    raise Exception(f"HTTP {response.status_code}: {error_text}")

            elif provider == "Hugging Face":
                # Hugging Face API 調用
                headers = {"Authorization": f"Bearer {api_key}"}
                data = {"inputs": params.get("prompt", "")}
                model_name = params.get("model", "FLUX.1-schnell")

                response = requests.post(
                    f"{base_url}/models/black-forest-labs/{model_name}",
                    headers=headers,
                    json=data,
                    timeout=60
                )

                if response.status_code == 200:
                    class MockResponse:
                        def __init__(self, image_data):
                            self.data = [type('obj', (object,), {
                                'url': f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                            })()]

                    return True, MockResponse(response.content)
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
            else:
                # Navy 和 Custom API（OpenAI Compatible）
                if client:
                    response = client.images.generate(**params)
                    return True, response
                else:
                    raise Exception("API 客戶端未初始化")

        except Exception as e:
            error_msg = str(e)

            if attempt < max_retries - 1:
                should_retry = False
                retry_conditions = ["500", "502", "503", "429", "timeout"]
                if any(code in error_msg for code in retry_conditions):
                    should_retry = True

                if should_retry:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    st.warning(f"⚠️ 第 {attempt + 1} 次嘗試失敗，{delay:.1f} 秒後重試...")
                    time.sleep(delay)
                    continue
                else:
                    return False, error_msg
            else:
                return False, error_msg

    return False, "所有重試均失敗"

def init_session_state():
    """初始化會話狀態"""
    if 'api_config' not in st.session_state:
        # 嘗試從文件載入已保存的配置
        saved_config = load_api_keys_from_file()
        if saved_config:
            st.session_state.api_config = {
                'provider': saved_config['provider'],
                'api_key': saved_config['api_key'],
                'base_url': saved_config['base_url'],
                'validated': saved_config['validated'],
                'pollinations_auth_mode': 'free',
                'pollinations_token': '',
                'pollinations_referrer': '',
                'krea_style_ref': False,
                'krea_style_strength': 0.7,
                'krea_style_url': '',
                'krea_aspect_mode': '標準尺寸',
                'krea_guidance': 7.5,
                'krea_steps': 50
            }
        else:
            st.session_state.api_config = {
                'provider': 'Navy',
                'api_key': '',
                'base_url': 'https://api.navy/v1',
                'validated': False,
                'pollinations_auth_mode': 'free',
                'pollinations_token': '',
                'pollinations_referrer': '',
                'krea_style_ref': False,
                'krea_style_strength': 0.7,
                'krea_style_url': '',
                'krea_aspect_mode': '標準尺寸',
                'krea_guidance': 7.5,
                'krea_steps': 50
            }

    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []

    if 'favorite_images' not in st.session_state:
        st.session_state.favorite_images = []

    if 'discovered_models' not in st.session_state:
        st.session_state.discovered_models = {}

    if 'models_updated' not in st.session_state:
        st.session_state.models_updated = False

def add_to_history(prompt: str, model: str, images: List[str], metadata: Dict):
    """添加生成記錄到歷史"""
    history_item = {
        "timestamp": datetime.datetime.now(),
        "prompt": prompt,
        "model": model,
        "images": images,
        "metadata": metadata,
        "id": str(uuid.uuid4())
    }
    st.session_state.generation_history.insert(0, history_item)

    # 限制歷史記錄數量
    if len(st.session_state.generation_history) > 50:
        st.session_state.generation_history = st.session_state.generation_history[:50]

def display_image_with_actions(image_url: str, image_id: str, history_item: Dict = None):
    """顯示圖像和相關操作"""
    try:
        # 處理 base64 圖像
        if image_url.startswith('data:image'):
            base64_data = image_url.split(',')[1]
            img_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(img_data))
        else:
            img_response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(img_response.content))
            img_data = img_response.content

        st.image(img, use_column_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            st.download_button(
                label="📥 下載",
                data=img_buffer.getvalue(),
                file_name=f"flux_generated_{image_id}.png",
                mime="image/png",
                key=f"download_{image_id}",
                use_container_width=True
            )

        with col2:
            is_favorite = any(fav['id'] == image_id for fav in st.session_state.favorite_images)
            if st.button(
                "⭐ 已收藏" if is_favorite else "☆ 收藏",
                key=f"favorite_{image_id}",
                use_container_width=True
            ):
                if is_favorite:
                    st.session_state.favorite_images = [
                        fav for fav in st.session_state.favorite_images if fav['id'] != image_id
                    ]
                    st.success("已取消收藏")
                else:
                    favorite_item = {
                        "id": image_id,
                        "image_url": image_url,
                        "timestamp": datetime.datetime.now(),
                        "history_item": history_item
                    }
                    st.session_state.favorite_images.append(favorite_item)
                    st.success("已加入收藏")
                rerun_app()

        with col3:
            if history_item and st.button(
                "🔄 重新生成",
                key=f"regenerate_{image_id}",
                use_container_width=True
            ):
                st.session_state.regenerate_prompt = history_item['prompt']
                st.session_state.regenerate_model = history_item['model']
                rerun_app()

    except Exception as e:
        st.error(f"圖像顯示錯誤: {str(e)}")

def init_api_client():
    """初始化 API 客戶端"""
    config = st.session_state.api_config
    
    if config.get('provider') in ["Hugging Face", "Pollinations.ai", "FLUX Krea AI Studio"]:
        return None
        
    if config.get('api_key'):
        try:
            return OpenAI(
                api_key=config['api_key'],
                base_url=config['base_url']
            )
        except Exception:
            return None
    return None

def show_api_settings():
    """顯示 API 設置界面"""
    st.subheader("🔑 API 設置 & 密鑰管理")

    provider_options = list(API_PROVIDERS.keys())
    current_provider = st.session_state.api_config.get('provider', 'Navy')

    provider_index = provider_options.index(current_provider) if current_provider in provider_options else 0

    selected_provider = st.selectbox(
        "選擇 API 提供商",
        options=provider_options,
        index=provider_index,
        format_func=lambda x: f"{API_PROVIDERS[x]['icon']} {API_PROVIDERS[x]['name']}"
    )

    provider_info = API_PROVIDERS[selected_provider]
    st.info(f"📋 {provider_info['description']}")

    # FLUX Krea AI Studio 特殊設置
    krea_settings = {}
    if selected_provider == "FLUX Krea AI Studio":
        krea_settings = show_krea_settings(selected_provider, st)

    # Pollinations.ai 特殊認證設置
    elif selected_provider == "Pollinations.ai":
        st.markdown("### 🌸 Pollinations.ai 認證設置")
        
        auth_mode = st.radio(
            "選擇認證模式",
            options=["free", "referrer", "token"],
            format_func=lambda x: {
                "free": "🆓 免費模式（基礎模型）",
                "referrer": "🌐 域名認證（推薦）",
                "token": "🔑 Token 認證（高級）"
            }[x],
            index=["free", "referrer", "token"].index(
                st.session_state.api_config.get('pollinations_auth_mode', 'free')
            )
        )

        if auth_mode == "referrer":
            st.info("輸入您的應用域名以存取更多模型（如 kontext）")
            referrer_input = st.text_input(
                "應用域名",
                value=st.session_state.api_config.get('pollinations_referrer', ''),
                placeholder="例如：myapp.vercel.app 或 username.github.io",
                help="輸入您部署應用的域名"
            )
        elif auth_mode == "token":
            st.info("使用 Token 進行後端認證，適合服務端整合")
            token_input = st.text_input(
                "Pollinations Token",
                value="",
                type="password",
                placeholder="在 https://auth.pollinations.ai 獲取您的 token",
                help="獲取 token：https://auth.pollinations.ai"
            )
            
            current_token = st.session_state.api_config.get('pollinations_token', '')
            if current_token and not token_input:
                st.caption(f"🔐 當前 Token: {current_token[:10]}...{current_token[-8:] if len(current_token) > 18 else ''}")
        else:
            st.info("免費模式：無需認證，但只能使用基礎模型")

    # API 密鑰設置
    is_key_required = selected_provider not in ["Pollinations.ai"]
    api_key_input = ""
    current_key = st.session_state.api_config.get('api_key', '')

    if is_key_required:
        masked_key = '*' * 20 + current_key[-8:] if len(current_key) > 8 else ''
        api_key_input = st.text_input(
            "API 密鑰",
            value="",
            type="password",
            placeholder=f"請輸入 {provider_info['name']} 的 API 密鑰...",
            help=f"API 密鑰通常以 '{provider_info['key_prefix']}' 開頭"
        )

        if current_key and not api_key_input:
            st.caption(f"🔐 當前密鑰: {masked_key}")
    else:
        if selected_provider != "Pollinations.ai":
            st.success("✅ 此提供商無需 API 密鑰。")
        current_key = "N/A"

    # 處理 Base URL 變化
    if selected_provider != current_provider:
        base_url_value = provider_info['base_url_default']
    else:
        base_url_value = st.session_state.api_config.get('base_url', provider_info['base_url_default'])

    base_url_input = st.text_input(
        "API 端點 URL",
        value=base_url_value,
        placeholder=provider_info['base_url_default'],
        help="API 服務的基礎 URL"
    )

    # 密鑰管理功能
    st.markdown("### 🔐 密鑰管理")
    col_save, col_load = st.columns(2)
    
    with col_save:
        if st.button("💾 保存密鑰到本地", use_container_width=True, help="將API密鑰加密保存到本地文件"):
            if save_api_keys_to_file():
                st.success("✅ API密鑰已安全保存到本地！")
    
    with col_load:
        saved_config = load_api_keys_from_file()
        if saved_config and saved_config.get('saved_at'):
            if st.button("📂 載入本地密鑰", use_container_width=True):
                st.session_state.api_config.update(saved_config)
                st.success("✅ 已載入本地保存的API密鑰！")
                time.sleep(1)
                rerun_app()
            st.caption(f"💽 上次保存: {saved_config['saved_at'][:19]}")

    # 保存設置按鈕
    col1, col2, col3 = st.columns(3)

    with col1:
        save_btn = st.button("💾 保存設置", type="primary")

    with col2:
        test_btn = st.button("🧪 測試連接")

    with col3:
        clear_btn = st.button("🗑️ 清除設置", type="secondary")

    # 處理按鈕事件
    if save_btn:
        final_api_key = api_key_input if api_key_input else current_key

        if is_key_required and not final_api_key:
            st.error("❌ 請輸入 API 密鑰")
        elif not base_url_input:
            st.error("❌ 請輸入 API 端點 URL")
        else:
            config_update = {
                'provider': selected_provider,
                'api_key': final_api_key,
                'base_url': base_url_input,
                'validated': False
            }

            # 更新 Krea 特殊設置
            if selected_provider == "FLUX Krea AI Studio":
                config_update.update({
                    'krea_style_ref': krea_settings.get('enable_style_ref', False),
                    'krea_style_strength': krea_settings.get('style_strength', 0.7),
                    'krea_style_url': krea_settings.get('style_url', ''),
                    'krea_aspect_mode': krea_settings.get('aspect_ratio_mode', '標準尺寸'),
                    'krea_guidance': krea_settings.get('guidance_scale', 7.5),
                    'krea_steps': krea_settings.get('steps', 50)
                })

            # Pollinations.ai 特殊設置
            elif selected_provider == "Pollinations.ai":
                config_update['pollinations_auth_mode'] = auth_mode
                if auth_mode == "referrer":
                    config_update['pollinations_referrer'] = referrer_input
                elif auth_mode == "token":
                    config_update['pollinations_token'] = token_input if token_input else st.session_state.api_config.get('pollinations_token', '')

            st.session_state.api_config.update(config_update)

            # 清除舊的發現模型和選擇的模型
            st.session_state.discovered_models = {}
            if 'selected_model' in st.session_state:
                del st.session_state.selected_model

            st.session_state.models_updated = True
            st.success("✅ API 設置已保存，模型列表已重置。")
            time.sleep(0.5)
            rerun_app()

    if test_btn:
        test_api_key = api_key_input if api_key_input else current_key

        if is_key_required and not test_api_key:
            st.error("❌ 請先輸入 API 密鑰")
        elif not base_url_input:
            st.error("❌ 請輸入 API 端點 URL")
        else:
            with st.spinner("正在測試 API 連接..."):
                is_valid, message = validate_api_key(test_api_key, base_url_input, selected_provider)

                if is_valid:
                    st.success(f"✅ {message}")
                    st.session_state.api_config['validated'] = True
                    
                    # 如果是 Krea，顯示額外信息
                    if selected_provider == "FLUX Krea AI Studio":
                        st.info("🎨 FLUX Krea AI Studio 連接成功！現在您可以使用專業美學模型進行圖像生成。")
                else:
                    st.error(f"❌ {message}")
                    st.session_state.api_config['validated'] = False

    if clear_btn:
        st.session_state.api_config = {
            'provider': 'Navy',
            'api_key': '',
            'base_url': 'https://api.navy/v1',
            'validated': False,
            'pollinations_auth_mode': 'free',
            'pollinations_token': '',
            'pollinations_referrer': '',
            'krea_style_ref': False,
            'krea_style_strength': 0.7,
            'krea_style_url': '',
            'krea_aspect_mode': '標準尺寸',
            'krea_guidance': 7.5,
            'krea_steps': 50
        }

        st.session_state.discovered_models = {}
        if 'selected_model' in st.session_state:
            del st.session_state.selected_model

        st.session_state.models_updated = True
        st.success("🗑️ API 設置已清除，模型列表已重置。")
        time.sleep(0.5)
        rerun_app()

def auto_discover_models():
    """執行自動模型發現"""
    config = st.session_state.api_config
    provider = config.get('provider')
    is_key_required = provider not in ["Pollinations.ai"]

    if is_key_required and not config.get('api_key'):
        st.error("❌ 請先配置 API 密鑰")
        return

    # 顯示發現進度
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        with st.spinner("🔍 正在自動發現模型..."):
            client = None
            if provider not in ["Hugging Face", "Pollinations.ai", "FLUX Krea AI Studio"]:
                client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])

            discovered = auto_discover_flux_models(
                client,
                config['provider'],
                config['api_key'],
                config['base_url']
            )

            if 'discovered_models' not in st.session_state:
                st.session_state.discovered_models = {}

            new_count = 0
            for model_id, model_info in discovered.items():
                if model_id not in BASE_FLUX_MODELS and model_id not in st.session_state.discovered_models:
                    new_count += 1
                st.session_state.discovered_models[model_id] = model_info

            if 'selected_model' in st.session_state:
                current_model = st.session_state.selected_model
                all_models = merge_models()
                if current_model not in all_models:
                    del st.session_state.selected_model

            st.session_state.models_updated = True

            if new_count > 0:
                progress_placeholder.success(f"✅ 發現 {new_count} 個新的模型！")
            elif discovered:
                progress_placeholder.info("ℹ️ 已刷新模型列表，未發現新模型。")
            else:
                progress_placeholder.warning("⚠️ 未發現任何兼容模型。")

            time.sleep(2)
            progress_placeholder.empty()
            rerun_app()

# 初始化
init_session_state()
client = init_api_client()
config = st.session_state.api_config
provider = config.get('provider')
is_key_required = provider not in ["Pollinations.ai"]
api_configured = (not is_key_required) or (config.get('api_key') and config.get('api_key') != 'N/A')

# 側邊欄
with st.sidebar:
    show_api_settings()
    st.markdown("---")
    
    if api_configured:
        st.success(f"🟢 {provider} API 已配置")
        
        # 顯示 Pollinations.ai 認證狀態
        if provider == "Pollinations.ai":
            auth_mode = config.get('pollinations_auth_mode', 'free')
            auth_status = {
                'free': '🆓 免費模式',
                'referrer': f'🌐 域名認證: {config.get("pollinations_referrer", "未設置")}',
                'token': '🔑 Token 認證'
            }
            st.caption(f"認證狀態: {auth_status[auth_mode]}")
        
        # 顯示 FLUX Krea AI Studio 狀態
        elif provider == "FLUX Krea AI Studio":
            st.caption("🎨 專業美學模式已啟用")
        
        if st.button("🔍 發現模型", use_container_width=True):
            auto_discover_models()
    else:
        st.error("🔴 API 未配置")

# 主標題和項目介紹
st.title("🎨 Flux AI 圖像生成器 Pro")
st.markdown("### 專業美學 | 藝術風格 | 密鑰管理")

# 項目介紹展示
with st.expander("📖 項目介紹與功能特色", expanded=False):
    st.markdown("""
    ## 🌟 項目概述
    
    **Flux AI 圖像生成器 Pro** 是一個功能強大的專業級AI圖像生成平台，整合了多種頂級API服務，
    特別強調FLUX Krea AI Studio的專業美學功能，為用戶提供無與倫比的圖像創作體驗。
    
    ## ✨ 核心功能特色
    
    ### 🎨 專業美學生成
    - **FLUX Krea AI Studio整合**: 專門解決AI圖像的「AI感」問題
    - **專業攝影模式**: 模擬真實攝影效果和專業光影
    - **風格參考控制**: 支援上傳參考圖像進行精確風格轉換
    - **智能提示詞增強**: 自動優化提示詞獲得更佳效果
    
    ### 🖌️ 豐富藝術風格庫
    - **70+ 藝術風格**: 涵蓋經典藝術、現代數位、電影風格等
    - **分類管理**: 按藝術運動、攝影風格、文化藝術等分類
    - **風格強度調整**: 精確控制風格在最終圖像中的影響程度
    - **實時預覽**: 風格描述與效果說明
    
    ### 🔐 安全密鑰管理
    - **本地加密保存**: 使用Fernet加密算法保護API密鑰
    - **多服務商支援**: Navy、Krea AI、Pollinations.ai、Hugging Face
    - **自動載入**: 啟動時自動載入已保存的配置
    - **安全清除**: 一鍵清除所有敏感資料
    
    ### 🔄 智能重試機制
    - **自動錯誤處理**: 智能識別暫時性錯誤並自動重試
    - **指數退避**: 使用智能延遲避免API限流
    - **詳細錯誤診斷**: 提供具體的錯誤解決建議
    
    ### 📚 完整歷史管理
    - **無限歷史記錄**: 保存所有生成記錄
    - **智能搜索**: 支援關鍵詞搜索歷史提示詞
    - **一鍵重新生成**: 快速重新生成喜愛的圖像
    - **收藏系統**: 收藏和管理優秀作品
    
    ## 🏆 技術優勢
    
    ### 🚀 性能優化
    - **並行處理**: 支援多圖像同時生成
    - **智能快取**: 減少重複API調用
    - **資源管理**: 智能內存和帶寬管理
    
    ### 🛡️ 安全性
    - **端到端加密**: API密鑰本地加密存儲
    - **無資料外洩**: 所有敏感資料僅本地處理
    - **安全連接**: 全程HTTPS加密傳輸
    
    ### 🌐 相容性
    - **跨平台**: 支援Windows、macOS、Linux
    - **多瀏覽器**: 相容所有主流瀏覽器
    - **響應式設計**: 支援桌面和移動設備
    
    ## 📈 使用統計
    
    本項目目前支援：
    - **5個** 主要API提供商
    - **70+** 藝術風格選項  
    - **8個** FLUX專業模型
    - **無限** 歷史記錄存儲
    
    ## 🔮 未來規劃
    
    - **批量生成**: 支援批量處理大量圖像
    - **API擴展**: 整合更多AI圖像生成服務
    - **協作功能**: 支援團隊協作和分享
    - **模型微調**: 支援自定義模型訓練
    """) # [web:22][web:23][web:26][web:29]

# 頁面導航
tab1, tab2, tab3 = st.tabs(["🚀 圖像生成", "📚 歷史記錄", "⭐ 收藏夾"])

# 圖像生成頁面
with tab1:
    if not api_configured:
        st.warning("⚠️ 請先在側邊欄配置 API")
        st.info("配置完成後即可開始生成圖像")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🎨 AI圖像生成")

            # 使用合併後的模型列表
            all_models = merge_models()

            # 檢查是否需要提示用戶發現模型
            if not all_models:
                st.warning("⚠️ 尚未發現任何模型，請點擊側邊欄的「發現模型」按鈕")
            else:
                # 如果模型列表被更新，顯示提示
                if st.session_state.get('models_updated', False):
                    st.info(f"🔄 模型列表已更新，共發現 {len(all_models)} 個可用模型")
                    st.session_state.models_updated = False

                # 模型選擇
                model_options = list(all_models.keys())

                # 處理重新生成時的模型選擇
                regenerate_model = getattr(st.session_state, 'regenerate_model', None)
                if regenerate_model and regenerate_model in model_options:
                    selected_model_key = regenerate_model
                elif 'selected_model' in st.session_state and st.session_state.selected_model in model_options:
                    selected_model_key = st.session_state.selected_model
                else:
                    selected_model_key = model_options[0]
                    st.session_state.selected_model = selected_model_key

                selected_model = st.selectbox(
                    "選擇模型:",
                    options=model_options,
                    index=model_options.index(selected_model_key),
                    format_func=lambda x: f"{all_models[x].get('icon', '🤖')} {all_models[x].get('name', x)}" + 
                                         (" 🔐" if all_models[x].get('auth_required', False) else "") +
                                         (" 🎨" if all_models[x].get('provider') == 'FLUX Krea AI Studio' else ""),
                    key="model_selector"
                )
                st.session_state.selected_model = selected_model

                # 顯示模型信息和認證警告
                model_info = all_models[selected_model]
                description = model_info.get('description', 'N/A')
                st.info(f"**{model_info.get('name')}**: {description}")

                # 檢查重新生成狀態
                default_prompt = ""
                if hasattr(st.session_state, 'regenerate_prompt'):
                    default_prompt = st.session_state.regenerate_prompt
                    delattr(st.session_state, 'regenerate_prompt')
                if hasattr(st.session_state, 'regenerate_model'):
                    delattr(st.session_state, 'regenerate_model')

                # 藝術風格選擇
                selected_style_desc = show_artistic_styles()

                # 提示詞輸入
                prompt_value = st.text_area(
                    "輸入提示詞:",
                    value=default_prompt,
                    height=120,
                    placeholder="描述您想要生成的圖像，例如: A majestic dragon flying over ancient mountains during sunset"
                )

                # 如果選擇了藝術風格，顯示最終提示詞預覽
                if selected_style_desc and prompt_value:
                    final_prompt = apply_artistic_style(prompt_value, selected_style_desc)
                    with st.expander("📝 最終提示詞預覽", expanded=False):
                        st.code(final_prompt, language=None)

                # 高級設置
                with st.expander("🔧 高級設置"):
                    col_size, col_num = st.columns(2)

                    with col_size:
                        size_options = {
                            "1024x1024": "正方形 (1:1)",
                            "1152x896": "橫向 (4:3.5)",
                            "896x1152": "直向 (3.5:4)",
                            "1344x768": "寬屏 (16:9)",
                            "768x1344": "超高 (9:16)"
                        }
                        selected_size = st.selectbox(
                            "圖像尺寸",
                            options=list(size_options.keys()),
                            format_func=lambda x: f"{x} - {size_options[x]}",
                            index=0
                        )

                    with col_num:
                        # Pollinations.ai 和 FLUX Krea AI Studio 僅支持單張生成
                        if provider in ["Pollinations.ai", "FLUX Krea AI Studio"]:
                            num_images = 1
                            st.caption(f"{provider} 僅支持單張圖像生成。")
                        else:
                            num_images = st.slider("生成數量", 1, 4, 1)

                # 生成按鈕
                generate_ready = prompt_value.strip() and api_configured
                generate_btn = st.button(
                    "🚀 生成圖像",
                    type="primary",
                    use_container_width=True,
                    disabled=not generate_ready
                )

                if not generate_ready:
                    if not prompt_value.strip():
                        st.warning("⚠️ 請輸入提示詞")
                    elif not api_configured:
                        st.error("❌ 請配置 API")

                # 圖像生成邏輯
                if generate_btn and generate_ready:
                    config = st.session_state.api_config

                    # 應用藝術風格到提示詞
                    final_prompt = apply_artistic_style(prompt_value, selected_style_desc) if selected_style_desc else prompt_value

                    with st.spinner(f"🎨 使用 {model_info.get('name', selected_model)} 正在生成圖像..."):
                        # 顯示進度信息
                        progress_info = st.empty()
                        style_info = f" | 風格: {st.session_state.get('style_name', '無')}" if selected_style_desc else ""
                        progress_info.info(f"⏳ 模型: {model_info.get('name')} | 尺寸: {selected_size} | 數量: {num_images}{style_info}")

                        generation_params = {
                            "model": selected_model,
                            "prompt": final_prompt,
                            "n": num_images,
                            "size": selected_size
                        }

                        success, result = generate_images_with_retry(
                            client,
                            config['provider'],
                            config['api_key'],
                            config['base_url'],
                            **generation_params
                        )

                        progress_info.empty()

                        if success:
                            response = result
                            image_urls = [img.url for img in response.data]

                            metadata = {
                                "size": selected_size,
                                "num_images": len(image_urls),
                                "model_name": model_info.get('name', selected_model),
                                "api_provider": config['provider'],
                                "generation_time": time.time(),
                                "original_prompt": prompt_value,
                                "final_prompt": final_prompt,
                                "artistic_style": st.session_state.get('style_name', '無風格'),
                                "style_category": st.session_state.get('style_category', '無風格')
                            }

                            add_to_history(final_prompt, selected_model, image_urls, metadata)
                            style_msg = f"，應用了 **{st.session_state.get('style_name', '')}** 風格" if selected_style_desc else ""
                            st.success(f"✨ 成功生成 {len(response.data)} 張圖像{style_msg}！")

                            # 顯示生
