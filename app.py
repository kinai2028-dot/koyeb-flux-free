"""
Flux AI 圖像生成器 Pro - 完整終極版
集成自動模型發現、FLUX Krea AI Studio、智能限制系統
專為 Koyeb 平台優化
"""

import streamlit as st
import requests
from openai import OpenAI
from PIL import Image
from io import BytesIO
import datetime
import base64
import time
import random
import json
import uuid
import psutil
import os
import re
from urllib.parse import urlencode, quote
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

# 設定頁面配置
st.set_page_config(
    page_title="Flux AI 圖像生成器 Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# Koyeb 平台適配
# =====================================

PORT = int(os.getenv("PORT", 8501))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

KOYEB_DOMAIN = os.getenv("KOYEB_PUBLIC_DOMAIN")
if KOYEB_DOMAIN:
    logger.info(f"🚀 運行在 Koyeb 平台: {KOYEB_DOMAIN}")

# =====================================
# 核心數據類定義
# =====================================

@dataclass
class APIProvider:
    """API提供商配置類"""
    name: str
    base_url: str
    key_prefix: str = ""
    description: str = ""
    icon: str = "🤖"
    auth_modes: List[str] = field(default_factory=list)
    supported_models: List[str] = field(default_factory=list)
    max_retries: int = 3
    timeout: int = 60

@dataclass
class ModelInfo:
    """模型信息類"""
    id: str
    name: str
    provider: str
    max_images: int = 4
    max_resolution: str = "1024x1024"
    recommended_sizes: List[str] = field(default_factory=list)
    supports_quality: bool = False
    supports_steps: bool = False
    description: str = ""
    estimated_time: int = 30
    cost_level: str = "medium"

@dataclass
class KreaModelInfo(ModelInfo):
    """Krea 模型專用信息類"""
    aesthetic_score: float = 9.0
    photography_focus: bool = True
    ai_look_avoidance: bool = True
    style_strength: float = 0.8
    supported_megapixels: List[str] = field(default_factory=lambda: ["0.25", "1"])
    go_fast_support: bool = True

# =====================================
# 配置管理類
# =====================================

class Config:
    """統一配置管理類"""
    
    def __init__(self):
        self.api_providers = self._load_providers()
        self.app_settings = self._load_app_settings()
        logger.info("配置加載完成")
    
    def _load_providers(self) -> Dict[str, APIProvider]:
        """載入API提供商配置"""
        return {
            "OpenAI Compatible": APIProvider(
                name="OpenAI Compatible API",
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                key_prefix="sk-",
                description="OpenAI 官方或兼容的 API 服務",
                icon="🤖",
                auth_modes=["api_key"],
                supported_models=["flux-pro", "flux-dev", "flux-schnell", "dall-e-3"]
            ),
            "Pollinations.ai": APIProvider(
                name="Pollinations.ai",
                base_url="https://image.pollinations.ai",
                description="免費圖像生成 API - 無需密鑰",
                icon="🌸",
                auth_modes=["free", "referrer", "token"],
                supported_models=["flux", "flux-anime", "flux-3d", "flux-realism"]
            ),
            "Hugging Face": APIProvider(
                name="Hugging Face Inference API",
                base_url="https://api-inference.huggingface.co/models",
                key_prefix="hf_",
                description="Hugging Face 推理 API",
                icon="🤗",
                auth_modes=["api_key"],
                supported_models=["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell"]
            ),
            "Krea AI (Segmind)": APIProvider(
                name="Krea AI via Segmind",
                base_url="https://api.segmind.com/v1",
                description="專業美學圖像生成 - FLUX Krea 模型 🎨",
                icon="🎨",
                auth_modes=["api_key"],
                supported_models=["flux-krea-dev"],
                timeout=60
            ),
            "Krea AI (FAL.ai)": APIProvider(
                name="Krea AI via FAL.ai",
                base_url="https://fal.run/fal-ai",
                description="快速 Krea AI 圖像生成服務 ⚡",
                icon="⚡",
                auth_modes=["api_key"],
                supported_models=["flux-krea-dev"],
                timeout=45
            )
        }
    
    def _load_app_settings(self) -> Dict[str, Any]:
        """載入應用程序設置"""
        return {
            "max_history": int(os.getenv("MAX_HISTORY", "50")),
            "timeout": int(os.getenv("REQUEST_TIMEOUT", "60")),
            "koyeb_domain": os.getenv("KOYEB_PUBLIC_DOMAIN", "localhost")
        }
    
    def get_provider(self, provider_name: str) -> APIProvider:
        """獲取特定提供商配置"""
        return self.api_providers.get(provider_name)

# =====================================
# 模型限制管理類
# =====================================

class ModelLimits:
    """模型限制管理類"""
    
    def __init__(self):
        self.model_configs = self._init_model_configs()
    
    def _init_model_configs(self) -> Dict[str, ModelInfo]:
        """初始化模型配置"""
        return {
            "flux-pro": ModelInfo(
                id="flux-pro", name="FLUX Pro", provider="OpenAI Compatible",
                max_images=4, max_resolution="2048x2048",
                recommended_sizes=["1024x1024", "1024x768", "768x1024"],
                supports_quality=True, estimated_time=45, cost_level="high",
                description="最高質量的 FLUX 模型，適合專業用途"
            ),
            "flux-dev": ModelInfo(
                id="flux-dev", name="FLUX Dev", provider="OpenAI Compatible",
                max_images=4, max_resolution="1536x1536",
                recommended_sizes=["1024x1024", "768x1024", "1024x768"],
                supports_quality=True, estimated_time=30, cost_level="medium",
                description="開發版 FLUX 模型，平衡質量和速度"
            ),
            "flux-schnell": ModelInfo(
                id="flux-schnell", name="FLUX Schnell", provider="OpenAI Compatible",
                max_images=6, max_resolution="1024x1024",
                recommended_sizes=["1024x1024", "512x512"],
                estimated_time=15, cost_level="low",
                description="快速 FLUX 模型，生成速度最快"
            ),
            "dall-e-3": ModelInfo(
                id="dall-e-3", name="DALL-E 3", provider="OpenAI Compatible",
                max_images=1, max_resolution="1024x1024",
                recommended_sizes=["1024x1024", "1024x1792", "1792x1024"],
                supports_quality=True, estimated_time=20, cost_level="high",
                description="OpenAI 的 DALL-E 3 模型"
            ),
            "flux": ModelInfo(
                id="flux", name="FLUX", provider="Pollinations.ai",
                max_images=8, max_resolution="1920x1080",
                recommended_sizes=["1024x1024", "1920x1080", "1080x1920"],
                estimated_time=25, cost_level="free",
                description="免費 FLUX 模型"
            ),
            "flux-anime": ModelInfo(
                id="flux-anime", name="FLUX Anime", provider="Pollinations.ai",
                max_images=6, max_resolution="1536x1536",
                recommended_sizes=["1024x1024", "768x1024"],
                estimated_time=30, cost_level="free",
                description="專門用於動漫風格的 FLUX 模型"
            ),
            "black-forest-labs/FLUX.1-dev": ModelInfo(
                id="black-forest-labs/FLUX.1-dev", name="FLUX.1 Dev", provider="Hugging Face",
                max_images=2, max_resolution="1024x1024",
                recommended_sizes=["1024x1024", "768x768", "512x512"],
                supports_steps=True, estimated_time=40, cost_level="medium",
                description="Hugging Face 上的 FLUX.1 開發版"
            ),
            "flux-krea-dev": KreaModelInfo(
                id="flux-krea-dev", name="FLUX.1 Krea [dev]", provider="Krea AI",
                max_images=4, max_resolution="1024x1024",
                recommended_sizes=["1024x1024", "768x1024", "1024x768"],
                supports_steps=True, estimated_time=15, cost_level="medium",
                description="專業美學 AI 模型，避免常見 AI 感，專注攝影真實感",
                aesthetic_score=9.2, photography_focus=True, ai_look_avoidance=True
            )
        }
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        return self.model_configs.get(model_id)
    
    def get_max_images(self, model_id: str) -> int:
        model_info = self.get_model_info(model_id)
        return model_info.max_images if model_info else 1
    
    def get_recommended_sizes(self, model_id: str) -> List[str]:
        model_info = self.get_model_info(model_id)
        return model_info.recommended_sizes if model_info else ["1024x1024"]

# =====================================
# Krea AI 風格預設系統
# =====================================

class KreaStylePresets:
    """Krea 風格預設庫"""
    
    def __init__(self):
        self.presets = {
            "professional_photography": {
                "name": "專業攝影", "icon": "📸",
                "description": "高端商業攝影風格，專業燈光和構圖",
                "prompt_template": "{subject}, professional photography, studio lighting, high-end commercial style, sharp details, aesthetic composition",
                "params": {"guidance": 3.5, "prompt_strength": 0.8, "go_fast": False}
            },
            "fashion_editorial": {
                "name": "時尚大片", "icon": "👗",
                "description": "時尚雜誌風格，戲劇性燈光和姿態",
                "prompt_template": "{subject}, high-fashion editorial photography, dramatic lighting, avant-garde styling, magazine quality",
                "params": {"guidance": 4.0, "prompt_strength": 0.85, "go_fast": False}
            },
            "cinematic_portrait": {
                "name": "電影肖像", "icon": "🎬",
                "description": "電影級人物肖像，情感表達豐富",
                "prompt_template": "{subject}, cinematic portrait photography, emotional depth, film-like quality, professional color grading",
                "params": {"guidance": 3.8, "prompt_strength": 0.75, "go_fast": False}
            },
            "natural_lifestyle": {
                "name": "自然生活", "icon": "🌅",
                "description": "自然光線下的生活場景，真實感強",
                "prompt_template": "{subject}, natural lifestyle photography, golden hour lighting, candid moments, authentic emotions",
                "params": {"guidance": 3.0, "prompt_strength": 0.7, "go_fast": True}
            }
        }
    
    def get_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        return self.presets.get(preset_name)
    
    def get_all_presets(self) -> Dict[str, Dict[str, Any]]:
        return self.presets
    
    def apply_preset_to_prompt(self, preset_name: str, subject: str) -> str:
        preset = self.get_preset(preset_name)
        if preset and "prompt_template" in preset:
            return preset["prompt_template"].format(subject=subject)
        return subject

# =====================================
# Krea 美學分析器
# =====================================

class KreaAestheticAnalyzer:
    """Krea 美學分析器"""
    
    def __init__(self):
        self.aesthetic_keywords = {
            "high_aesthetic": ["professional", "cinematic", "artistic", "aesthetic", "beautiful", "stunning"],
            "lighting_quality": ["golden hour", "soft lighting", "dramatic lighting", "natural light", "studio lighting"],
            "composition": ["perfect composition", "rule of thirds", "symmetrical", "balanced", "dynamic composition"],
            "quality_indicators": ["high resolution", "sharp details", "crisp", "clear", "detailed", "high quality"],
            "avoid_ai_look": ["natural", "authentic", "realistic", "genuine", "organic", "candid"]
        }
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        aesthetic_score = 0
        recommendations = []
        
        for category, keywords in self.aesthetic_keywords.items():
            found_keywords = [kw for kw in keywords if kw in prompt_lower]
            category_score = len(found_keywords) / len(keywords)
            aesthetic_score += category_score
            
            if category_score < 0.3:
                recommendations.append(self._get_category_recommendation(category))
        
        aesthetic_score = min(10, (aesthetic_score / len(self.aesthetic_keywords)) * 10)
        
        return {
            "aesthetic_score": round(aesthetic_score, 1),
            "recommendations": recommendations[:3],
            "quality_level": self._get_quality_level(aesthetic_score)
        }
    
    def _get_category_recommendation(self, category: str) -> str:
        recommendations = {
            "high_aesthetic": "添加美學描述詞，如 'beautiful', 'stunning', 'elegant'",
            "lighting_quality": "指定燈光類型，如 'golden hour lighting', 'soft natural light'",
            "composition": "添加構圖描述，如 'perfect composition', 'balanced framing'",
            "quality_indicators": "添加質量描述，如 'high resolution', 'sharp details'",
            "avoid_ai_look": "添加自然感描述，如 'natural', 'authentic', 'realistic'"
        }
        return recommendations.get(category, "優化提示詞結構")
    
    def _get_quality_level(self, score: float) -> str:
        if score >= 8: return "專業級"
        elif score >= 6: return "高質量"
        elif score >= 4: return "標準"
        else: return "需優化"
    
    def optimize_prompt_for_krea(self, prompt: str) -> str:
        analysis = self.analyze_prompt(prompt)
        if analysis["aesthetic_score"] < 6:
            optimizations = []
            if "beautiful" not in prompt.lower():
                optimizations.append("aesthetic composition")
            if not any(q in prompt.lower() for q in ["high quality", "detailed"]):
                optimizations.append("high quality details")
            if "photography" not in prompt.lower():
                optimizations.append("professional photography")
            if optimizations:
                return f"{prompt}, {', '.join(optimizations)}"
        return prompt

# =====================================
# API 客戶端基類
# =====================================

class BaseAPIClient:
    """API客戶端基類"""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 60):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Flux-AI-Generator-Koyeb/1.0',
            'Accept': 'application/json'
        })
    
    def _handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        error_msg = f"{context}: {str(error)}" if context else str(error)
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault('timeout', self.timeout)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self.session.request(method, url, **kwargs)
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)

# =====================================
# OpenAI 兼容客戶端
# =====================================

class OpenAIClient(BaseAPIClient):
    """OpenAI兼容客戶端"""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 60):
        super().__init__(api_key, base_url, timeout)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=3)
    
    def generate_image(self, **kwargs) -> Dict[str, Any]:
        try:
            model = kwargs.get("model", "flux-dev")
            
            if "dall-e" in model.lower():
                params = {
                    "model": model, "prompt": kwargs.get("prompt", ""),
                    "n": min(kwargs.get("n", 1), 4), "size": kwargs.get("size", "1024x1024"),
                    "quality": kwargs.get("quality", "standard"), "response_format": "url"
                }
            else:
                params = {
                    "model": model, "prompt": kwargs.get("prompt", ""),
                    "n": kwargs.get("n", 1), "size": kwargs.get("size", "1024x1024"),
                    "response_format": "url"
                }
            
            response = self.client.images.generate(**params)
            
            images = []
            for img in response.data:
                images.append({
                    "url": img.url,
                    "revised_prompt": getattr(img, 'revised_prompt', params["prompt"]),
                    "id": str(uuid.uuid4())
                })
            
            return {
                "success": True, "data": images, "model": params["model"],
                "prompt": params["prompt"], "provider": "OpenAI Compatible"
            }
        except Exception as e:
            return self._handle_error(e, "OpenAI 圖像生成")

# =====================================
# Pollinations.ai 客戶端
# =====================================

class PollinationsClient(BaseAPIClient):
    """Pollinations.ai 客戶端"""
    
    def generate_image(self, **kwargs) -> Dict[str, Any]:
        try:
            prompt = kwargs.get("prompt", "")
            model = kwargs.get("model", "flux")
            width = kwargs.get("width", 1024)
            height = kwargs.get("height", 1024)
            
            params = {
                "model": model, "width": width, "height": height,
                "seed": kwargs.get("seed", random.randint(1, 1000000)),
                "enhance": "true", "safe": "true"
            }
            
            auth_mode = kwargs.get("auth_mode", "free")
            if auth_mode == "referrer" and kwargs.get("referrer"):
                params["referrer"] = kwargs["referrer"]
            elif auth_mode == "token" and self.api_key:
                params["token"] = self.api_key
            
            url = f"{self.base_url}/prompt/{quote(prompt)}"
            full_url = f"{url}?{urlencode(params)}"
            
            response = self._make_request("GET", full_url)
            response.raise_for_status()
            
            return {
                "success": True,
                "data": [{"url": full_url, "revised_prompt": prompt, "id": str(uuid.uuid4())}],
                "model": model, "prompt": prompt, "provider": "Pollinations.ai"
            }
        except Exception as e:
            return self._handle_error(e, "Pollinations 圖像生成")

# =====================================
# Hugging Face 客戶端
# =====================================

class HuggingFaceClient(BaseAPIClient):
    """Hugging Face 客戶端"""
    
    def generate_image(self, **kwargs) -> Dict[str, Any]:
        try:
            model = kwargs.get("model", "black-forest-labs/FLUX.1-dev")
            prompt = kwargs.get("prompt", "")
            
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "num_inference_steps": kwargs.get("steps", 20),
                    "guidance_scale": kwargs.get("guidance_scale", 7.5)
                }
            }
            
            response = self._make_request("POST", f"{self.base_url}/{model}", headers=headers, json=payload)
            response.raise_for_status()
            
            image_data = response.content
            image_b64 = base64.b64encode(image_data).decode()
            
            return {
                "success": True,
                "data": [{
                    "image_data": image_data, "image_b64": image_b64,
                    "revised_prompt": prompt, "id": str(uuid.uuid4())
                }],
                "model": model, "prompt": prompt, "provider": "Hugging Face"
            }
        except Exception as e:
            return self._handle_error(e, "HuggingFace 圖像生成")

# =====================================
# Krea AI Segmind 客戶端
# =====================================

class KreaSegmindClient(BaseAPIClient):
    """Krea Segmind API 客戶端"""
    
    def __init__(self, api_key: str, timeout: int = 60):
        super().__init__(api_key, "https://api.segmind.com/v1", timeout)
        self.aesthetic_analyzer = KreaAestheticAnalyzer()
    
    def generate_image(self, **kwargs) -> Dict[str, Any]:
        try:
            prompt = kwargs.get("prompt", "")
            optimized_prompt = self.aesthetic_analyzer.optimize_prompt_for_krea(prompt)
            
            payload = {
                "prompt": optimized_prompt,
                "seed": kwargs.get("seed", random.randint(1, 1000000)),
                "guidance": kwargs.get("guidance", 3.5),
                "megapixels": str(kwargs.get("megapixels", 1)),
                "num_outputs": kwargs.get("n", 1),
                "aspect_ratio": self._convert_size_to_aspect_ratio(kwargs.get("size", "1024x1024")),
                "output_format": kwargs.get("output_format", "jpg"),
                "output_quality": kwargs.get("output_quality", 90),
                "prompt_strength": kwargs.get("prompt_strength", 0.8),
                "num_inference_steps": kwargs.get("steps", 40),
                "go_fast": kwargs.get("go_fast", True),
                "disable_safety_checker": kwargs.get("disable_safety_checker", False),
                "image": "null"
            }
            
            response = self._make_request(
                "POST", f"{self.base_url}/flux-krea-dev",
                json=payload, headers={"x-api-key": self.api_key}
            )
            response.raise_for_status()
            
            image_data = response.content
            image_b64 = base64.b64encode(image_data).decode()
            aesthetic_analysis = self.aesthetic_analyzer.analyze_prompt(optimized_prompt)
            
            return {
                "success": True,
                "data": [{
                    "image_data": image_data, "image_b64": image_b64,
                    "revised_prompt": optimized_prompt, "id": str(uuid.uuid4()),
                    "aesthetic_score": aesthetic_analysis["aesthetic_score"],
                    "quality_level": aesthetic_analysis["quality_level"],
                    "krea_optimized": optimized_prompt != prompt
                }],
                "model": "flux-krea-dev", "prompt": optimized_prompt,
                "original_prompt": prompt, "provider": "Krea AI (Segmind)"
            }
        except Exception as e:
            return self._handle_error(e, "Krea Segmind 圖像生成")
    
    def _convert_size_to_aspect_ratio(self, size: str) -> str:
        try:
            width, height = map(int, size.split('x'))
            ratio_map = {
                (1, 1): "1:1", (4, 3): "4:3", (3, 4): "3:4",
                (16, 9): "16:9", (9, 16): "9:16", (3, 2): "3:2", (2, 3): "2:3"
            }
            target_ratio = width / height
            best_match = "1:1"
            min_diff = float('inf')
            for (w, h), ratio_str in ratio_map.items():
                diff = abs(w/h - target_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_match = ratio_str
            return best_match
        except:
            return "1:1"

# =====================================
# Krea FAL.ai 客戶端
# =====================================

class KreaFALClient(BaseAPIClient):
    """Krea FAL.ai API 客戶端"""
    
    def __init__(self, api_key: str, timeout: int = 60):
        super().__init__(api_key, "https://fal.run/fal-ai", timeout)
        self.aesthetic_analyzer = KreaAestheticAnalyzer()
    
    def generate_image(self, **kwargs) -> Dict[str, Any]:
        try:
            prompt = kwargs.get("prompt", "")
            optimized_prompt = self.aesthetic_analyzer.optimize_prompt_for_krea(prompt)
            
            headers = {"Authorization": f"Key {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "prompt": optimized_prompt,
                "image_size": kwargs.get("size", "landscape_4_3"),
                "num_inference_steps": kwargs.get("steps", 28),
                "guidance_scale": kwargs.get("guidance", 3.5),
                "num_images": kwargs.get("n", 1),
                "enable_safety_checker": not kwargs.get("disable_safety_checker", False),
                "seed": kwargs.get("seed")
            }
            
            payload = {k: v for k, v in payload.items() if v is not None}
            
            response = self._make_request(
                "POST", f"{self.base_url}/flux-krea-dev",
                headers=headers, json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            images = []
            
            for img in result.get("images", []):
                aesthetic_analysis = self.aesthetic_analyzer.analyze_prompt(optimized_prompt)
                images.append({
                    "url": img.get("url"), "revised_prompt": optimized_prompt,
                    "id": str(uuid.uuid4()),
                    "aesthetic_score": aesthetic_analysis["aesthetic_score"],
                    "quality_level": aesthetic_analysis["quality_level"],
                    "krea_optimized": optimized_prompt != prompt
                })
            
            return {
                "success": True, "data": images, "model": "flux-krea-dev",
                "prompt": optimized_prompt, "original_prompt": prompt,
                "provider": "Krea AI (FAL.ai)"
            }
        except Exception as e:
            return self._handle_error(e, "Krea FAL.ai 圖像生成")

# =====================================
# 客戶端工廠類
# =====================================

class ClientFactory:
    """客戶端工廠類"""
    
    @staticmethod
    def create_client(provider_name: str, api_key: str, base_url: str, **kwargs) -> BaseAPIClient:
        timeout = kwargs.get("timeout", 60)
        
        if provider_name in ["OpenAI Compatible"]:
            return OpenAIClient(api_key, base_url, timeout)
        elif provider_name == "Pollinations.ai":
            return PollinationsClient(api_key or "", base_url, timeout)
        elif provider_name == "Hugging Face":
            return HuggingFaceClient(api_key, base_url, timeout)
        elif provider_name == "Krea AI (Segmind)":
            return KreaSegmindClient(api_key, timeout)
        elif provider_name == "Krea AI (FAL.ai)":
            return KreaFALClient(api_key, timeout)
        else:
            raise ValueError(f"不支持的API提供商: {provider_name}")

# =====================================
# 圖像生成器核心類
# =====================================

class ImageGenerator:
    """圖像生成核心類"""
    
    def __init__(self, config: Config):
        self.config = config
        self.clients = {}
        self.generation_stats = {"total": 0, "success": 0, "failed": 0}
    
    def get_client(self, provider_name: str, api_key: str = "") -> BaseAPIClient:
        cache_key = f"{provider_name}_{api_key[:8] if api_key else 'none'}"
        
        if cache_key not in self.clients:
            provider = self.config.get_provider(provider_name)
            if not provider:
                raise ValueError(f"未知的API提供商: {provider_name}")
            
            self.clients[cache_key] = ClientFactory.create_client(
                provider_name, api_key, provider.base_url, timeout=provider.timeout
            )
        return self.clients[cache_key]
    
    def generate_images_with_retry(self, provider_name: str, api_key: str = "", **kwargs) -> Dict[str, Any]:
        self.generation_stats["total"] += 1
        provider = self.config.get_provider(provider_name)
        max_retries = provider.max_retries if provider else 3
        
        for attempt in range(max_retries):
            try:
                client = self.get_client(provider_name, api_key)
                result = client.generate_image(**kwargs)
                
                if result.get("success"):
                    self.generation_stats["success"] += 1
                    logger.info(f"圖像生成成功: {provider_name}")
                    return result
                else:
                    logger.warning(f"生成失敗 (嘗試 {attempt + 1}/{max_retries}): {result.get('error')}")
            except Exception as e:
                logger.error(f"生成異常 (嘗試 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        self.generation_stats["failed"] += 1
        return {"success": False, "error": f"經過 {max_retries} 次重試後仍然失敗"}
    
    def get_stats(self) -> Dict[str, int]:
        return self.generation_stats.copy()

# =====================================
# 工具函數
# =====================================

def rerun_app():
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    else:
        st.stop()

def get_koyeb_info():
    return {
        "domain": os.getenv("KOYEB_PUBLIC_DOMAIN", "localhost"),
        "port": PORT
    }

def download_image(url: str) -> Optional[bytes]:
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"下載圖像失敗: {e}")
        return None

# =====================================
# UI 組件
# =====================================

def render_sidebar(config: Config, model_limits: ModelLimits):
    """渲染側邊欄"""
    with st.sidebar:
        st.title("🎨 Flux AI Pro")
        st.caption("智能模型發現版")
        
        koyeb_info = get_koyeb_info()
        if koyeb_info["domain"] != "localhost":
            st.success(f"🌐 部署於: {koyeb_info['domain']}")
        
        st.divider()
        
        # API 提供商選擇
        provider_names = list(config.api_providers.keys())
        selected_provider = st.selectbox("🔧 API 提供商", provider_names, index=0)
        provider = config.get_provider(selected_provider)
        st.info(f"{provider.icon} {provider.description}")
        
        # API 密鑰輸入
        api_key = ""
        if "api_key" in provider.auth_modes:
            api_key_env_var = f"{selected_provider.upper().replace(' ', '_').replace('.', '_')}_API_KEY"
            api_key_from_env = os.getenv(api_key_env_var, "")
            
            if api_key_from_env:
                st.success(f"✅ 使用環境變量中的 API 密鑰")
                api_key = api_key_from_env
            else:
                api_key = st.text_input("🔑 API 密鑰", type="password")
        
        # 認證模式（針對 Pollinations.ai）
        auth_mode = "free"
        referrer = ""
        if selected_provider == "Pollinations.ai":
            auth_mode = st.selectbox("🔐 認證模式", provider.auth_modes, index=0)
            if auth_mode == "referrer":
                referrer = st.text_input("🌐 Referrer URL", value=f"https://{koyeb_info['domain']}")
        
        # 模型選擇
        available_models = provider.supported_models
        if available_models:
            selected_model = st.selectbox("🤖 模型", available_models, index=0)
        else:
            selected_model = st.text_input("🤖 自定義模型", value="flux-dev")
        
        # 獲取模型信息
        model_info = model_limits.get_model_info(selected_model)
        if model_info:
            with st.expander("📊 模型詳情", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("最大圖片數", model_info.max_images)
                    st.metric("預估時間", f"{model_info.estimated_time}秒")
                with col2:
                    cost_colors = {"free": "green", "low": "green", "medium": "orange", "high": "red"}
                    st.markdown(f"**成本等級:** :{cost_colors.get(model_info.cost_level, 'gray')}[{model_info.cost_level.upper()}]")
                if model_info.description:
                    st.caption(model_info.description)
        
        # Krea AI 專用控制面板
        krea_params = {}
        if "Krea AI" in selected_provider:
            krea_params = render_krea_controls()
        
        # 智能參數設置
        st.subheader("⚙️ 智能參數")
        
        max_images = model_info.max_images if model_info else 4
        num_images = st.slider("圖片數量", 1, max_images, min(2, max_images))
        
        recommended_sizes = model_limits.get_recommended_sizes(selected_model)
        all_sizes = list(set(recommended_sizes + ["512x512", "768x768", "1024x1024", "768x1024", "1024x768"]))
        
        sorted_sizes = []
        for size in recommended_sizes:
            if size in all_sizes:
                sorted_sizes.append(f"🌟 {size} (推薦)")
                all_sizes.remove(size)
        for size in sorted(all_sizes):
            sorted_sizes.append(size)
        
        selected_size_display = st.selectbox("圖像尺寸", sorted_sizes)
        selected_size = selected_size_display.replace("🌟 ", "").replace(" (推薦)", "")
        
        # 高級設置
        with st.expander("🔧 高級設置"):
            quality = "standard"
            if model_info and model_info.supports_quality:
                quality = st.selectbox("圖像質量", ["standard", "hd"], index=0)
            
            steps = 20
            guidance_scale = 7.5
            if model_info and model_info.supports_steps:
                steps = st.slider("推理步數", 10, 50, 20)
                guidance_scale = st.slider("引導比例", 1.0, 20.0, 7.5)
            
            use_seed = st.checkbox("使用固定種子")
            seed = None
            if use_seed:
                seed = st.number_input("種子值", 0, 999999, random.randint(0, 999999))
        
        # 存儲設置
        st.session_state.update({
            "selected_provider": selected_provider, "api_key": api_key,
            "auth_mode": auth_mode, "referrer": referrer,
            "selected_model": selected_model, "num_images": num_images,
            "image_size": selected_size, "quality": quality,
            "steps": steps, "guidance_scale": guidance_scale, "seed": seed,
            **krea_params
        })
        
        # 使用統計
        with st.expander("📈 使用統計"):
            if 'image_generator' in st.session_state:
                stats = st.session_state.image_generator.get_stats()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("總生成", stats["total"])
                    st.metric("成功", stats["success"])
                with col2:
                    st.metric("失敗", stats["failed"])
                    if stats["total"] > 0:
                        success_rate = (stats["success"] / stats["total"]) * 100
                        st.metric("成功率", f"{success_rate:.1f}%")

def render_krea_controls():
    """渲染 Krea AI 專用控制面板"""
    st.subheader("🎨 Krea AI 專業控制")
    
    if 'krea_style_presets' not in st.session_state:
        st.session_state.krea_style_presets = KreaStylePresets()
    if 'krea_aesthetic_analyzer' not in st.session_state:
        st.session_state.krea_aesthetic_analyzer = KreaAestheticAnalyzer()
    
    krea_params = {}
    
    # 風格預設
    with st.expander("🎭 專業風格預設", expanded=True):
        style_presets = st.session_state.krea_style_presets.get_all_presets()
        preset_options = ["自定義風格"] + [f"{preset['icon']} {preset['name']}" for preset in style_presets.values()]
        
        selected_preset_display = st.selectbox("選擇風格預設", preset_options)
        
        if selected_preset_display != "自定義風格":
            preset_name = None
            for name, preset in style_presets.items():
                if f"{preset['icon']} {preset['name']}" == selected_preset_display:
                    preset_name = name
                    break
            
            if preset_name:
                preset_info = style_presets[preset_name]
                st.info(f"📝 {preset_info['description']}")
                st.session_state.selected_krea_preset = preset_name
                krea_params.update(preset_info['params'])
    
    # Krea 專業參數
    with st.expander("⚙️ Krea 專業參數"):
        col1, col2 = st.columns(2)
        
        with col1:
            guidance = st.slider("🎯 引導值", 0.0, 10.0, krea_params.get('guidance', 3.5), step=0.1)
            prompt_strength = st.slider("💪 提示詞強度", 0.0, 1.0, krea_params.get('prompt_strength', 0.8), step=0.05)
        
        with col2:
            megapixels = st.selectbox("📐 分辨率等級", ["0.25", "1"], index=1)
            go_fast = st.checkbox("⚡ 快速模式", value=krea_params.get('go_fast', True))
        
        if not go_fast:
            steps = st.slider("🔄 推理步數", 20, 60, 40)
        else:
            steps = 25
        
        output_format = st.selectbox("📁 輸出格式", ["jpg", "png", "webp"], index=0)
        if output_format == "jpg":
            output_quality = st.slider("🎨 JPG 質量", 50, 100, 90)
        else:
            output_quality = 100
    
    # 美學分析工具
    with st.expander("🔍 美學分析工具"):
        st.write("**提示詞美學評估**")
        if hasattr(st.session_state, 'current_prompt') and st.session_state.current_prompt:
            analyzer = st.session_state.krea_aesthetic_analyzer
            analysis = analyzer.analyze_prompt(st.session_state.current_prompt)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("美學分數", f"{analysis['aesthetic_score']}/10")
            with col2:
                quality_colors = {"專業級": "green", "高質量": "blue", "標準": "orange", "需優化": "red"}
                quality_level = analysis['quality_level']
                st.markdown(f"**質量等級:** :{quality_colors.get(quality_level, 'gray')}[{quality_level}]")
            
            if analysis['recommendations']:
                st.write("**優化建議:**")
                for i, rec in enumerate(analysis['recommendations'], 1):
                    st.caption(f"{i}. {rec}")
        else:
            st.info("輸入提示詞後顯示美學分析")
    
    disable_safety = st.checkbox("🔓 禁用安全檢查", value=False)
    
    krea_params.update({
        "guidance": guidance, "prompt_strength": prompt_strength,
        "megapixels": float(megapixels), "go_fast": go_fast, "steps": steps,
        "output_format": output_format, "output_quality": output_quality,
        "disable_safety_checker": disable_safety
    })
    
    return krea_params

def render_main_page(config: Config, image_generator: ImageGenerator, model_limits: ModelLimits):
    """渲染主頁面"""
    st.title("🎨 Flux AI 圖像生成器 Pro")
    st.caption("專為 Koyeb 平台優化 - 集成自動模型發現和 Krea AI Studio")
    
    if not hasattr(st.session_state, 'selected_provider'):
        st.info("👈 請先在側邊欄選擇 API 提供商開始使用")
        return
    
    # 標籤頁
    tab1, tab2, tab3 = st.tabs(["🎨 圖像生成", "📚 歷史記錄", "⭐ 收藏夾"])
    
    with tab1:
        render_generation_tab(image_generator, model_limits)
    with tab2:
        render_history_tab()
    with tab3:
        render_favorites_tab()

def render_generation_tab(image_generator: ImageGenerator, model_limits: ModelLimits):
    """渲染圖像生成標籤頁"""
    if not hasattr(st.session_state, 'selected_model'):
        st.info("👈 請先在側邊欄選擇模型")
        return
    
    selected_provider = st.session_state.get('selected_provider', '')
    
    # 智能提示詞輸入
    if "Krea AI" in selected_provider:
        prompt = render_krea_prompt_studio()
    else:
        prompt = render_standard_prompt_input()
    
    # 操作按鈕
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        generate_button = st.button("🎨 生成圖像", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ 清空結果", use_container_width=True):
            st.session_state.current_images = []
            rerun_app()
    
    with col3:
        if st.button("🎲 隨機提示", use_container_width=True):
            random_prompts = [
                "A beautiful sunset over mountains, digital art style, highly detailed",
                "Cyberpunk city at night with neon lights, 4k quality",
                "Professional portrait photography, studio lighting, elegant",
                "Abstract art with vibrant colors, modern composition"
            ]
            st.session_state.random_prompt = random.choice(random_prompts)
            rerun_app()
    
    # 使用隨機提示
    if hasattr(st.session_state, 'random_prompt'):
        prompt = st.session_state.random_prompt
        del st.session_state.random_prompt
    
    # 執行圖像生成
    if generate_button and prompt.strip():
        provider_name = st.session_state.get("selected_provider")
        api_key = st.session_state.get("api_key", "")
        
        config = st.session_state.config
        provider = config.get_provider(provider_name)
        
        if "api_key" in provider.auth_modes and not api_key:
            st.error("❌ 請在側邊欄輸入 API 密鑰")
            return
        
        execute_generation(image_generator, prompt.strip(), model_limits)
    
    elif generate_button and not prompt.strip():
        st.warning("⚠️ 請輸入提示詞")
    
    # 顯示當前生成的圖像
    if hasattr(st.session_state, 'current_images') and st.session_state.current_images:
        st.subheader("🖼️ 生成結果")
        display_images(st.session_state.current_images)

def render_standard_prompt_input():
    """渲染標準提示詞輸入"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area(
            "描述你想要生成的圖像",
            height=120,
            placeholder="例如：A beautiful sunset over mountains, digital art style, highly detailed"
        )
    
    with col2:
        st.write("💡 **提示詞建議**")
        tips = [
            "• 使用具體描述詞",
            "• 添加藝術風格",
            "• 指定圖像質量",
            "• 描述構圖元素"
        ]
        for tip in tips:
            st.caption(tip)
    
    return prompt

def render_krea_prompt_studio():
    """渲染 Krea 提示詞工作室"""
    st.subheader("✍️ Krea 提示詞工作室")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt_input_method = st.radio("輸入方式", ["直接輸入", "風格模板"], horizontal=True)
        
        if prompt_input_method == "直接輸入":
            prompt = st.text_area(
                "描述你想要的圖像",
                height=120,
                placeholder="例如: A professional model in elegant evening wear, studio photography, dramatic lighting"
            )
        else:  # 風格模板
            if hasattr(st.session_state, 'krea_style_presets'):
                presets = st.session_state.krea_style_presets.get_all_presets()
                
                template_name = st.selectbox(
                    "選擇模板",
                    list(presets.keys()),
                    format_func=lambda x: f"{presets[x]['icon']} {presets[x]['name']}"
                )
                
                subject = st.text_input("主題描述", placeholder="例如: a young woman in a red dress")
                
                if subject:
                    prompt = st.session_state.krea_style_presets.apply_preset_to_prompt(template_name, subject)
                    st.text_area("生成的提示詞", prompt, height=100, disabled=True)
                else:
                    prompt = ""
            else:
                prompt = ""
    
    with col2:
        st.write("💡 **Krea 提示詞技巧**")
        tips = [
            "🎯 使用專業攝影詞匯",
            "🎨 添加自然感描述",
            "📸 指定燈光風格",
            "🔍 強調細節質量"
        ]
        for tip in tips:
            st.caption(tip)
    
    # 保存當前提示詞用於分析
    st.session_state.current_prompt = prompt
    return prompt

def execute_generation(image_generator: ImageGenerator, prompt: str, model_limits: ModelLimits):
    """執行圖像生成"""
    provider_name = st.session_state.get("selected_provider")
    api_key = st.session_state.get("api_key", "")
    model_id = st.session_state.get("selected_model")
    
    size_str = st.session_state.get("image_size", "1024x1024")
    width, height = map(int, size_str.split("x"))
    
    generation_params = {
        "prompt": prompt, "model": model_id,
        "n": st.session_state.get("num_images", 1),
        "size": size_str, "width": width, "height": height,
        "quality": st.session_state.get("quality", "standard"),
        "steps": st.session_state.get("steps", 20),
        "guidance_scale": st.session_state.get("guidance_scale", 7.5)
    }
    
    if st.session_state.get("seed"):
        generation_params["seed"] = st.session_state.get("seed")
    
    # Pollinations.ai 特定參數
    if provider_name == "Pollinations.ai":
        generation_params.update({
            "auth_mode": st.session_state.get("auth_mode", "free"),
            "referrer": st.session_state.get("referrer", "")
        })
    
    # Krea AI 特定參數
    if "Krea AI" in provider_name:
        generation_params.update({
            "guidance": st.session_state.get("guidance", 3.5),
            "prompt_strength": st.session_state.get("prompt_strength", 0.8),
            "megapixels": st.session_state.get("megapixels", 1),
            "go_fast": st.session_state.get("go_fast", True),
            "output_format": st.session_state.get("output_format", "jpg"),
            "output_quality": st.session_state.get("output_quality", 90),
            "disable_safety_checker": st.session_state.get("disable_safety_checker", False)
        })
    
    # 顯示進度
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        with st.spinner(f"🎨 使用 {provider_name} 生成圖像..."):
            status_text.text("🔗 正在連接 API...")
            progress_bar.progress(25)
            
            result = image_generator.generate_images_with_retry(provider_name, api_key, **generation_params)
            
            progress_bar.progress(75)
            status_text.text("🎨 正在處理圖像...")
            progress_bar.progress(100)
        
        progress_bar.empty()
        status_text.empty()
        
        if result.get("success"):
            st.session_state.current_images = result["data"]
            add_to_history(prompt, provider_name, result, generation_params)
            st.success(f"✅ 成功生成 {len(result['data'])} 張圖像！")
        else:
            st.error(f"❌ 生成失敗: {result.get('error')}")
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ 生成過程中出現錯誤: {e}")

def add_to_history(prompt: str, provider: str, result: Dict, params: Dict):
    """添加到歷史記錄"""
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    history_record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt": prompt, "provider": provider,
        "model": result.get("model"), "images": result["data"], "params": params
    }
    
    st.session_state.generation_history.insert(0, history_record)
    
    if len(st.session_state.generation_history) > 50:
        st.session_state.generation_history = st.session_state.generation_history[:50]

def display_images(images: List[Dict[str, Any]]):
    """顯示圖像網格"""
    if not images:
        return
    
    num_cols = min(len(images), 3)
    cols = st.columns(num_cols)
    
    for i, img in enumerate(images):
        with cols[i % num_cols]:
            try:
                # 顯示圖像
                if "url" in img:
                    st.image(img["url"], use_column_width=True)
                    image_url = img["url"]
                    image_data = None
                elif "image_data" in img:
                    if "image_b64" in img:
                        st.image(f"data:image/png;base64,{img['image_b64']}", use_column_width=True)
                    else:
                        image = Image.open(BytesIO(img["image_data"]))
                        st.image(image, use_column_width=True)
                    image_url = None
                    image_data = img["image_data"]
                else:
                    st.error("無效的圖像數據")
                    continue
                
                # 顯示提示詞
                if img.get("revised_prompt"):
                    st.caption(f"**提示詞:** {img['revised_prompt'][:100]}...")
                
                # Krea AI 特殊信息
                if img.get('aesthetic_score'):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("美學分數", f"{img['aesthetic_score']}/10")
                    with col2:
                        if img.get('krea_optimized'):
                            st.success("✨ Krea 優化")
                
                # 操作按鈕
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    # 下載按鈕
                    if image_url:
                        if st.button(f"💾 下載", key=f"download_{i}_{img.get('id', i)}"):
                            downloaded_data = download_image(image_url)
                            if downloaded_data:
                                st.download_button(
                                    "📥 點擊下載", downloaded_data,
                                    file_name=f"flux_image_{int(time.time())}.png",
                                    mime="image/png", key=f"dl_btn_{i}_{img.get('id', i)}"
                                )
                    elif image_data:
                        st.download_button(
                            "💾 下載", image_data,
                            file_name=f"flux_image_{int(time.time())}.png",
                            mime="image/png", key=f"download_{i}_{img.get('id', i)}"
                        )
                
                with btn_col2:
                    # 收藏按鈕
                    img_id = img.get('id', str(uuid.uuid4()))
                    
                    if 'favorites' not in st.session_state:
                        st.session_state.favorites = []
                    
                    is_favorited = any(fav.get('id') == img_id for fav in st.session_state.favorites)
                    
                    if not is_favorited:
                        if st.button(f"⭐ 收藏", key=f"fav_{i}_{img_id}"):
                            img_copy = img.copy()
                            img_copy['id'] = img_id
                            img_copy['favorited_at'] = datetime.datetime.now().isoformat()
                            st.session_state.favorites.insert(0, img_copy)
                            st.success("已收藏！")
                            rerun_app()
                    else:
                        st.write("⭐ 已收藏")
            
            except Exception as e:
                st.error(f"顯示圖像失敗: {e}")

def render_history_tab():
    """渲染歷史記錄標籤頁"""
    st.subheader("📚 生成歷史")
    
    history = st.session_state.get('generation_history', [])
    
    if not history:
        st.info("還沒有生成歷史記錄")
        return
    
    # 清空歷史按鈕
    if st.button("🗑️ 清空歷史"):
        st.session_state.generation_history = []
        rerun_app()
    
    # 顯示歷史記錄
    for record in history:
        with st.expander(f"🕒 {record.get('timestamp', '未知時間')[:19]} - {record.get('prompt', '')[:50]}..."):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**提示詞:** {record.get('prompt')}")
                st.write(f"**提供商:** {record.get('provider')}")
                st.write(f"**模型:** {record.get('model')}")
            
            with col2:
                if st.button(f"🔄 重新生成", key=f"regen_{record.get('id')}"):
                    st.session_state.regen_prompt = record.get('prompt')
                    rerun_app()
            
            # 顯示圖像
            if record.get('images'):
                display_images(record['images'])

def render_favorites_tab():
    """渲染收藏夾標籤頁"""
    st.subheader("⭐ 我的收藏")
    
    favorites = st.session_state.get('favorites', [])
    
    if not favorites:
        st.info("還沒有收藏的圖像")
        return
    
    # 顯示收藏
    for fav in favorites:
        with st.expander(f"⭐ {fav.get('prompt', fav.get('revised_prompt', ''))[:50]}..."):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**提示詞:** {fav.get('prompt', fav.get('revised_prompt', ''))}")
                st.write(f"**收藏時間:** {fav.get('favorited_at', '未知')[:19]}")
            
            with col2:
                if st.button(f"❌ 取消收藏", key=f"unfav_{fav.get('id')}"):
                    st.session_state.favorites = [
                        f for f in st.session_state.favorites 
                        if f.get('id') != fav.get('id')
                    ]
                    rerun_app()
            
            # 顯示圖像
            display_images([fav])

# =====================================
# 主應用邏輯
# =====================================

def initialize_app():
    """初始化應用程序"""
    logger.info("初始化 Flux AI 圖像生成器 - 終極版")
    
    if 'config' not in st.session_state:
        st.session_state.config = Config()
        logger.info("配置初始化完成")
    
    if 'model_limits' not in st.session_state:
        st.session_state.model_limits = ModelLimits()
        logger.info("模型限制初始化完成")
    
    if 'image_generator' not in st.session_state:
        st.session_state.image_generator = ImageGenerator(st.session_state.config)
        logger.info("圖像生成器初始化完成")
    
    if 'current_images' not in st.session_state:
        st.session_state.current_images = []
    
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    
    # 處理重新生成請求
    if hasattr(st.session_state, 'regen_prompt'):
        st.session_state.current_prompt = st.session_state.regen_prompt
        del st.session_state.regen_prompt

def main():
    """主應用函數"""
    try:
        # 初始化應用
        initialize_app()
        
        config = st.session_state.config
        model_limits = st.session_state.model_limits
        image_generator = st.session_state.image_generator
        
        # 渲染側邊欄
        render_sidebar(config, model_limits)
        
        # 渲染主頁面
        render_main_page(config, image_generator, model_limits)
        
        # 頁腳信息
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.caption("🚀 Powered by Koyeb")
        
        with col2:
            koyeb_info = get_koyeb_info()
            st.caption(f"🌐 {koyeb_info['domain']}")
        
        with col3:
            stats = image_generator.get_stats()
            st.caption(f"📊 總生成: {stats['total']}")
        
    except Exception as e:
        logger.error(f"應用運行錯誤: {e}")
        st.error(f"應用運行出現錯誤: {e}")
        st.info("請刷新頁面重試")

if __name__ == "__main__":
    main()
