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
# 配置類定義
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

class Config:
    """統一配置管理類"""
    
    def __init__(self):
        self.api_providers = self._load_providers()
        self.app_settings = self._load_app_settings()
    
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
            "timeout": int(os.getenv("REQUEST_TIMEOUT", "60"))
        }
    
    def get_provider(self, provider_name: str) -> APIProvider:
        """獲取特定提供商配置"""
        return self.api_providers.get(provider_name)

# =====================================
# API 客戶端類
# =====================================

class BaseAPIClient:
    """API客戶端基類"""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 60):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()

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
            return {"success": False, "error": str(e)}

class PollinationsClient(BaseAPIClient):
    """Pollinations.ai 客戶端"""
    
    def generate_image(self, **kwargs) -> Dict[str, Any]:
        """生成圖像"""
        try:
            prompt = kwargs.get("prompt", "")
            model = kwargs.get("model", "flux")
            width = kwargs.get("width", 1024)
            height = kwargs.get("height", 1024)
            
            params = {
                "model": model,
                "width": width,
                "height": height,
                "enhance": "true"
            }
            
            url = f"{self.base_url}/prompt/{quote(prompt)}"
            query_string = urlencode(params)
            full_url = f"{url}?{query_string}"
            
            response = self.session.get(full_url, timeout=self.timeout)
            response.raise_for_status()
            
            return {
                "success": True,
                "data": [{
                    "url": full_url,
                    "revised_prompt": prompt
                }],
                "model": model,
                "prompt": prompt
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# =====================================
# 主應用邏輯
# =====================================

def rerun_app():
    """兼容性重新運行函數"""
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()

def main():
    """主應用函數"""
    # 初始化配置
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    
    config = st.session_state.config
    
    # 側邊欄
    with st.sidebar:
        st.title("🎨 Flux AI Pro")
        
        # API 提供商選擇
        provider_names = list(config.api_providers.keys())
        selected_provider = st.selectbox("🔧 API 提供商", provider_names)
        
        provider = config.get_provider(selected_provider)
        st.info(f"{provider.icon} {provider.description}")
        
        # API 密鑰輸入
        api_key = ""
        if "api_key" in provider.auth_modes:
            api_key = st.text_input("🔑 API 密鑰", type="password")
        
        # 模型選擇
        selected_model = st.selectbox("🤖 模型", provider.supported_models)
    
    # 主頁面
    st.title("🎨 Flux AI 圖像生成器 Pro")
    
    # 提示詞輸入
    prompt = st.text_area(
        "✍️ 描述你想要生成的圖像",
        height=100,
        placeholder="例如：A beautiful sunset over mountains, digital art style"
    )
    
    # 生成按鈕
    if st.button("🎨 生成圖像", type="primary"):
        if not prompt:
            st.error("請輸入提示詞")
            return
        
        if "api_key" in provider.auth_modes and not api_key:
            st.error("請輸入 API 密鑰")
            return
        
        with st.spinner(f"🎨 正在使用 {selected_provider} 生成圖像..."):
            try:
                # 創建客戶端
                if selected_provider == "OpenAI Compatible":
                    client = OpenAIClient(api_key, provider.base_url)
                elif selected_provider == "Pollinations.ai":
                    client = PollinationsClient("", provider.base_url)
                else:
                    st.error("不支持的 API 提供商")
                    return
                
                # 生成圖像
                result = client.generate_image(
                    prompt=prompt,
                    model=selected_model,
                    n=1
                )
                
                if result.get("success"):
                    st.success("✅ 圖像生成成功！")
                    
                    # 顯示圖像
                    for img in result["data"]:
                        if "url" in img:
                            st.image(img["url"], caption=f"提示詞: {img.get('revised_prompt', prompt)}")
                        
                        # 下載按鈕
                        if st.button(f"💾 下載圖像"):
                            try:
                                response = requests.get(img["url"])
                                st.download_button(
                                    "📥 點擊下載",
                                    response.content,
                                    file_name=f"flux_image_{int(time.time())}.png",
                                    mime="image/png"
                                )
                            except Exception as e:
                                st.error(f"下載失敗: {e}")
                else:
                    st.error(f"❌ 生成失敗: {result.get('error')}")
                    
            except Exception as e:
                st.error(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    main()
