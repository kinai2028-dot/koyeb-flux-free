"""
API客戶端模塊 - 統一封裝不同API提供商的接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import requests
from openai import OpenAI
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseAPIClient(ABC):
    """API客戶端基類"""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 60):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
    
    @abstractmethod
    def generate_image(self, **kwargs) -> Dict[str, Any]:
        """生成圖像"""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """列出可用模型"""
        pass

class OpenAIClient(BaseAPIClient):
    """OpenAI兼容客戶端"""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 60):
        super().__init__(api_key, base_url, timeout)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    
    def generate_image(self, **kwargs) -> Dict[str, Any]:
        """生成圖像"""
        try:
            params = {
                "model": kwargs.get("model", "flux-dev"),
                "prompt": kwargs.get("prompt", ""),
                "n": kwargs.get("n", 1),
                "size": kwargs.get("size", "1024x1024")
            }
            
            response = self.client.images.generate(**params)
            
            images = []
            for img in response.data:
                images.append({
                    "url": img.url,
                    "revised_prompt": getattr(img, 'revised_prompt', params["prompt"])
                })
            
            return {
                "success": True,
                "data": images,
                "model": params["model"],
                "prompt": params["prompt"]
            }
            
        except Exception as e:
            logger.error(f"OpenAI 圖像生成失敗: {e}")
            return {"success": False, "error": str(e)}

class ClientFactory:
    """客戶端工廠類"""
    
    @staticmethod
    def create_client(provider_name: str, api_key: str, base_url: str, **kwargs) -> BaseAPIClient:
        """創建API客戶端"""
        timeout = kwargs.get("timeout", 60)
        
        if provider_name in ["OpenAI Compatible", "Navy"]:
            return OpenAIClient(api_key, base_url, timeout)
        else:
            raise ValueError(f"不支持的API提供商: {provider_name}")
