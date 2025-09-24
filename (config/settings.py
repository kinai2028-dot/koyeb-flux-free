"""
配置管理模塊 - 統一管理所有配置項
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
import json

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

class Config:
    """統一配置管理類"""
    
    def __init__(self):
        self.api_providers = self._load_providers()
        self.app_settings = self._load_app_settings()
        self.ui_settings = self._load_ui_settings()
    
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
                supported_models=["flux-pro", "flux-dev", "flux-schnell"]
            ),
            "Pollinations.ai": APIProvider(
                name="Pollinations.ai",
                base_url="https://image.pollinations.ai",
                description="支援免費和認證模式的圖像生成 API",
                icon="🌸",
                auth_modes=["free", "referrer", "token"],
                supported_models=["flux", "flux-anime", "flux-3d"]
            ),
            "Hugging Face": APIProvider(
                name="Hugging Face Inference API",
                base_url="https://api-inference.huggingface.co/models",
                key_prefix="hf_",
                description="Hugging Face 推理 API",
                icon="🤗",
                auth_modes=["api_key"],
                supported_models=["flux.1-dev", "flux.1-schnell"]
            )
        }
    
    def _load_app_settings(self) -> Dict[str, Any]:
        """載入應用程序設置"""
        return {
            "max_history": int(os.getenv("MAX_HISTORY", "50")),
            "cache_size": int(os.getenv("CACHE_SIZE", "100")),
            "timeout": int(os.getenv("REQUEST_TIMEOUT", "60")),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "enable_cache": os.getenv("ENABLE_CACHE", "true").lower() == "true"
        }
