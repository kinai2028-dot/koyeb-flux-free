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
    page_title="Flux AI 圖像生成器 Pro - Auto Models",
    page_icon="🎨",
    layout="wide"
)

# API 提供商配置
API_PROVIDERS = {
    "OpenAI Compatible": {
        "name": "OpenAI Compatible API",
        "base_url_default": "https://api.openai.com/v1",
        "key_prefix": "sk-",
        "description": "OpenAI 官方或兼容的 API 服務",
        "icon": "🤖"
    },
    "Navy": {
        "name": "Navy API",
        "base_url_default": "https://api.navy/v1", 
        "key_prefix": "sk-",
        "description": "Navy 提供的 AI 圖像生成服務",
        "icon": "⚓"
    },
    "Hugging Face": {
        "name": "Hugging Face Inference",
        "base_url_default": "https://api-inference.huggingface.co",
        "key_prefix": "hf_",
        "description": "Hugging Face Inference API",
        "icon": "🤗"
    },
    "Together AI": {
        "name": "Together AI",
        "base_url_default": "https://api.together.xyz/v1",
        "key_prefix": "",
        "description": "Together AI 平台",
        "icon": "🤝"
    },
    "Fireworks AI": {
        "name": "Fireworks AI",
        "base_url_default": "https://api.fireworks.ai/inference/v1",
        "key_prefix": "",
        "description": "Fireworks AI 快速推理",
        "icon": "🎆"
    },
    "Custom": {
        "name": "自定義 API",
        "base_url_default": "",
        "key_prefix": "",
        "description": "自定義的 API 端點",
        "icon": "🔧"
    }
}

# 基礎 Flux 模型配置（手動維護的核心模型）
BASE_FLUX_MODELS = {
    "flux.1-schnell": {
        "name": "FLUX.1 Schnell",
        "description": "最快的生成速度，開源模型",
        "icon": "⚡",
        "type": "快速生成",
        "test_prompt": "A simple cat sitting on a table",
        "expected_size": "1024x1024",
        "priority": 1,
        "source": "base"
    },
    "flux.1-dev": {
        "name": "FLUX.1 Dev", 
        "description": "開發版本，平衡速度與質量",
        "icon": "🔧",
        "type": "開發版本",
        "test_prompt": "A beautiful landscape with mountains",
        "expected_size": "1024x1024",
        "priority": 2,
        "source": "base"
    },
    "flux.1.1-pro": {
        "name": "FLUX.1.1 Pro",
        "description": "改進的旗艦模型，最佳品質",
        "icon": "👑",
        "type": "旗艦版本",
        "test_prompt": "Professional portrait of a person in business attire",
        "expected_size": "1024x1024",
        "priority": 3,
        "source": "base"
    },
    "flux.1-kontext-pro": {
        "name": "FLUX.1 Kontext Pro",
        "description": "支持圖像編輯和上下文理解",
        "icon": "🎯",
        "type": "編輯專用",
        "test_prompt": "Abstract geometric shapes in vibrant colors",
        "expected_size": "1024x1024",
        "priority": 4,
        "source": "base"
    }
}

# 模型自動發現規則
FLUX_MODEL_PATTERNS = {
    # 基本 Flux 模型模式
    r'flux[\.\-]?1[\.\-]?schnell': {
        "name_template": "FLUX.1 Schnell",
        "icon": "⚡",
        "type": "快速生成",
        "priority_base": 100
    },
    r'flux[\.\-]?1[\.\-]?dev': {
        "name_template": "FLUX.1 Dev",
        "icon": "🔧", 
        "type": "開發版本",
        "priority_base": 200
    },
    r'flux[\.\-]?1[\.\-]?pro': {
        "name_template": "FLUX.1 Pro",
        "icon": "👑",
        "type": "專業版本",
        "priority_base": 300
    },
    r'flux[\.\-]?1[\.\-]?kontext': {
        "name_template": "FLUX.1 Kontext",
        "icon": "🎯",
        "type": "上下文理解",
        "priority_base": 400
    },
    r'flux[\.\-]?2': {
        "name_template": "FLUX.2",
        "icon": "🚀",
        "type": "下一代",
        "priority_base": 500
    },
    # 自定義和微調模型
    r'flux.*krea': {
        "name_template": "FLUX Krea",
        "icon": "🎨",
        "type": "創意增強",
        "priority_base": 600
    },
    r'flux.*anime': {
        "name_template": "FLUX Anime",
        "icon": "🌸",
        "type": "動漫風格",
        "priority_base": 700
    },
    r'flux.*realism': {
        "name_template": "FLUX Realism",
        "icon": "📷",
        "type": "寫實風格",
        "priority_base": 800
    },
    r'flux.*art': {
        "name_template": "FLUX Art",
        "icon": "🖼️",
        "type": "藝術風格",
        "priority_base": 900
    }
}

# 提供商特定的模型端點
HF_FLUX_ENDPOINTS = [
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1.1-pro",
    "XLabs-AI/flux-RealismLora",
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
    "multimodalart/FLUX.1-merged",
]

def auto_discover_flux_models(client, provider: str, api_key: str, base_url: str) -> Dict[str, Dict]:
    """自動發現 Flux 模型"""
    discovered_models = {}
    
    try:
        if provider == "Hugging Face":
            # Hugging Face 特殊處理
            for endpoint in HF_FLUX_ENDPOINTS:
                model_id = endpoint.split('/')[-1]
                model_info = analyze_model_name(model_id, endpoint)
                model_info['source'] = 'huggingface'
                model_info['endpoint'] = endpoint
                discovered_models[model_id] = model_info
        
        else:
            # OpenAI 兼容 API
            response = client.models.list()
            
            for model in response.data:
                model_id = model.id.lower()
                
                # 檢查是否是 Flux 相關模型
                if is_flux_model(model_id):
                    model_info = analyze_model_name(model.id)
                    model_info['source'] = 'api_discovery'
                    discovered_models[model.id] = model_info
        
        return discovered_models
    
    except Exception as e:
        st.warning(f"模型自動發現失敗: {str(e)}")
        return {}

def is_flux_model(model_name: str) -> bool:
    """檢查模型名稱是否為 Flux 模型"""
    model_lower = model_name.lower()
    flux_keywords = ['flux', 'black-forest-labs']
    return any(keyword in model_lower for keyword in flux_keywords)

def analyze_model_name(model_id: str, full_path: str = None) -> Dict:
    """分析模型名稱並生成模型信息"""
    model_lower = model_id.lower()
    
    # 嘗試匹配已知模式
    for pattern, info in FLUX_MODEL_PATTERNS.items():
        if re.search(pattern, model_lower):
            analyzed_info = {
                "name": info["name_template"],
                "icon": info["icon"],
                "type": info["type"],
                "description": f"自動發現的 {info['name_template']} 模型",
                "test_prompt": "A beautiful scene with detailed elements",
                "expected_size": "1024x1024",
                "priority": info["priority_base"] + hash(model_id) % 100,
                "auto_discovered": True
            }
            
            # 如果有完整路徑，提取更多信息
            if full_path:
                analyzed_info["full_path"] = full_path
                # 嘗試從路徑提取作者信息
                if '/' in full_path:
                    author = full_path.split('/')[0]
                    analyzed_info["name"] += f" ({author})"
            
            return analyzed_info
    
    # 如果沒有匹配到模式，創建通用信息
    return {
        "name": model_id.replace('-', ' ').replace('_', ' ').title(),
        "icon": "🤖",
        "type": "自動發現",
        "description": f"自動發現的模型: {model_id}",
        "test_prompt": "A detailed and beautiful image",
        "expected_size": "1024x1024", 
        "priority": 999,
        "auto_discovered": True,
        "full_path": full_path if full_path else model_id
    }

def merge_models() -> Dict[str, Dict]:
    """合併基礎模型和自動發現的模型"""
    # 從會話狀態獲取自動發現的模型
    discovered = st.session_state.get('discovered_models', {})
    
    # 合併模型
    merged_models = BASE_FLUX_MODELS.copy()
    
    for model_id, model_info in discovered.items():
        if model_id not in merged_models:
            merged_models[model_id] = model_info
    
    return merged_models

def show_model_discovery_panel():
    """顯示模型自動發現面板"""
    st.subheader("🔍 模型自動發現")
    
    col_info, col_controls = st.columns([2, 1])
    
    with col_info:
        st.markdown("""
        **自動發現功能:**
        - 🔍 掃描 API 端點可用模型
        - 🤖 智能識別 Flux 相關模型
        - 📋 自動分類和標註
        - ⚡ 實時更新模型列表
        """)
        
        # 顯示發現統計
        if 'discovered_models' in st.session_state:
            total_discovered = len(st.session_state.discovered_models)
            new_models = len([m for m in st.session_state.discovered_models.values() if m.get('auto_discovered')])
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("發現模型", total_discovered)
            with col_stat2:
                st.metric("新增模型", new_models)
    
    with col_controls:
        if st.button("🔍 開始自動發現", type="primary", use_container_width=True):
            auto_discover_models()
        
        if st.button("🔄 重新發現", use_container_width=True):
            st.session_state.discovered_models = {}
            auto_discover_models()
        
        if st.button("🗑️ 清除發現", use_container_width=True):
            st.session_state.discovered_models = {}
            st.success("已清除自動發現的模型")
            rerun_app()
        
        # 自動發現設置
        with st.expander("⚙️ 發現設置"):
            auto_test = st.checkbox("發現後自動測試", value=True)
            include_experimental = st.checkbox("包含實驗性模型", value=False)
            max_models = st.slider("最大發現數量", 5, 50, 20)

def auto_discover_models():
    """執行自動模型發現"""
    if 'api_config' not in st.session_state or not st.session_state.api_config.get('api_key'):
        st.error("❌ 請先配置 API 密鑰")
        return
    
    config = st.session_state.api_config
    
    with st.spinner("🔍 正在自動發現 Flux 模型..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 步驟 1: 連接 API
        status_text.text("📡 連接 API 服務...")
        progress_bar.progress(0.2)
        
        if config['provider'] == "Hugging Face":
            client = None
        else:
            client = OpenAI(
                api_key=config['api_key'],
                base_url=config['base_url']
            )
        
        # 步驟 2: 發現模型
        status_text.text("🔍 掃描可用模型...")
        progress_bar.progress(0.4)
        
        discovered = auto_discover_flux_models(
            client, config['provider'], config['api_key'], config['base_url']
        )
        
        # 步驟 3: 分析和分類
        status_text.text("🤖 分析模型信息...")
        progress_bar.progress(0.6)
        
        # 保存到會話狀態
        if 'discovered_models' not in st.session_state:
            st.session_state.discovered_models = {}
        
        new_count = 0
        for model_id, model_info in discovered.items():
            if model_id not in st.session_state.discovered_models:
                new_count += 1
            st.session_state.discovered_models[model_id] = model_info
        
        # 步驟 4: 完成
        status_text.text("✅ 發現完成")
        progress_bar.progress(1.0)
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        if new_count > 0:
            st.success(f"✅ 發現 {new_count} 個新的 Flux 模型！")
        else:
            st.info("ℹ️ 未發現新的 Flux 模型")
        
        # 自動測試新發現的模型（如果啟用）
        if new_count > 0:
            if st.checkbox("是否測試新發現的模型？", value=True):
                test_discovered_models(list(discovered.keys())[:5])  # 限制測試數量

def test_discovered_models(model_ids: List[str]):
    """測試自動發現的模型"""
    if not model_ids:
        return
    
    config = st.session_state.api_config
    
    if config['provider'] == "Hugging Face":
        client = None
    else:
        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['base_url']
        )
    
    with st.spinner(f"🧪 測試 {len(model_ids)} 個新發現的模型..."):
        test_results = {}
        progress_bar = st.progress(0)
        
        for i, model_id in enumerate(model_ids):
            progress = (i + 1) / len(model_ids)
            progress_bar.progress(progress)
            
            try:
                result = test_model_availability(
                    client, model_id, config['provider'],
                    config['api_key'], config['base_url']
                )
                test_results[model_id] = result
                
                # 短暫延遲避免過於頻繁的請求
                time.sleep(0.5)
            except Exception as e:
                st.warning(f"測試模型 {model_id} 時出錯: {str(e)}")
        
        progress_bar.empty()
        
        # 更新測試結果
        if 'model_test_results' not in st.session_state:
            st.session_state.model_test_results = {}
        
        st.session_state.model_test_results.update(test_results)
        
        # 顯示結果摘要
        available_count = sum(1 for r in test_results.values() if r.get('available'))
        st.success(f"✅ 測試完成：{available_count}/{len(test_results)} 個模型可用")

def show_discovered_models_list():
    """顯示已發現的模型列表"""
    if 'discovered_models' not in st.session_state or not st.session_state.discovered_models:
        st.info("🔍 尚未發現任何模型，點擊「開始自動發現」來掃描可用模型")
        return
    
    st.subheader("📋 已發現的模型")
    
    # 按來源和優先級排序
    sorted_models = sorted(
        st.session_state.discovered_models.items(),
        key=lambda x: (
            x[1].get('source', 'unknown'),
            x[1].get('priority', 999),
            x[0]
        )
    )
    
    # 按來源分組顯示
    sources = {}
    for model_id, model_info in sorted_models:
        source = model_info.get('source', 'unknown')
        if source not in sources:
            sources[source] = []
        sources[source].append((model_id, model_info))
    
    for source, models in sources.items():
        source_names = {
            'base': '🏠 基礎模型',
            'api_discovery': '🤖 API 發現',
            'huggingface': '🤗 Hugging Face',
            'unknown': '❓ 未知來源'
        }
        
        st.markdown(f"### {source_names.get(source, source)}")
        
        for model_id, model_info in models:
            with st.expander(f"{model_info.get('icon', '🤖')} {model_info.get('name', model_id)}"):
                col_info, col_actions = st.columns([2, 1])
                
                with col_info:
                    st.markdown(f"**模型 ID**: `{model_id}`")
                    st.markdown(f"**描述**: {model_info.get('description', 'N/A')}")
                    st.markdown(f"**類型**: {model_info.get('type', 'N/A')}")
                    st.markdown(f"**來源**: {source}")
                    
                    if model_info.get('full_path'):
                        st.markdown(f"**完整路徑**: `{model_info['full_path']}`")
                    
                    # 顯示測試結果
                    if model_id in st.session_state.get('model_test_results', {}):
                        result = st.session_state.model_test_results[model_id]
                        if result.get('available'):
                            st.success(f"✅ 模型可用 (響應時間: {result.get('response_time', 0):.2f}s)")
                        else:
                            st.error(f"❌ 模型不可用: {result.get('error', 'Unknown error')}")
                
                with col_actions:
                    # 測試單個模型
                    if st.button(f"🧪 測試", key=f"test_discovered_{model_id}"):
                        test_single_discovered_model(model_id)
                    
                    # 移除模型
                    if st.button(f"🗑️ 移除", key=f"remove_discovered_{model_id}"):
                        del st.session_state.discovered_models[model_id]
                        st.success(f"已移除模型: {model_id}")
                        rerun_app()
                    
                    # 加入收藏
                    if st.button(f"⭐ 收藏", key=f"favorite_discovered_{model_id}"):
                        add_model_to_favorites(model_id, model_info)

def test_single_discovered_model(model_id: str):
    """測試單個已發現的模型"""
    config = st.session_state.api_config
    
    if config['provider'] == "Hugging Face":
        client = None
    else:
        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['base_url']
        )
    
    with st.spinner(f"🧪 測試模型 {model_id}..."):
        result = test_model_availability(
            client, model_id, config['provider'],
            config['api_key'], config['base_url']
        )
        
        if 'model_test_results' not in st.session_state:
            st.session_state.model_test_results = {}
        
        st.session_state.model_test_results[model_id] = result
        
        if result.get('available'):
            st.success(f"✅ 模型 {model_id} 測試成功！")
        else:
            st.error(f"❌ 模型 {model_id} 測試失敗: {result.get('error')}")
        
        rerun_app()

def add_model_to_favorites(model_id: str, model_info: Dict):
    """將模型加入收藏"""
    if 'favorite_models' not in st.session_state:
        st.session_state.favorite_models = []
    
    # 檢查是否已經收藏
    if not any(fav['id'] == model_id for fav in st.session_state.favorite_models):
        favorite_item = {
            'id': model_id,
            'info': model_info,
            'added_at': datetime.datetime.now()
        }
        st.session_state.favorite_models.append(favorite_item)
        st.success(f"⭐ 已將 {model_info.get('name', model_id)} 加入收藏")
    else:
        st.info("該模型已在收藏列表中")

# 原有的函數保持不變...
def validate_api_key(api_key: str, base_url: str, provider: str) -> Tuple[bool, str]:
    """驗證 API 密鑰是否有效"""
    try:
        if provider == "Hugging Face":
            headers = {"Authorization": f"Bearer {api_key}"}
            test_url = f"{base_url}/models/black-forest-labs/FLUX.1-schnell"
            response = requests.get(test_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return True, "Hugging Face API 密鑰驗證成功"
            else:
                return False, f"HTTP {response.status_code}: 驗證失敗"
        else:
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

def test_model_availability(client, model_name: str, provider: str, api_key: str, base_url: str, test_prompt: str = None) -> Dict:
    """測試特定模型的可用性"""
    if test_prompt is None:
        # 從合併的模型配置中獲取測試提示詞
        all_models = merge_models()
        test_prompt = all_models.get(model_name, {}).get('test_prompt', 'A simple test image')
    
    test_result = {
        'model': model_name,
        'available': False,
        'response_time': 0,
        'error': None,
        'details': {}
    }
    
    try:
        start_time = time.time()
        
        if provider == "Hugging Face":
            # Hugging Face API 調用
            headers = {"Authorization": f"Bearer {api_key}"}
            data = {"inputs": test_prompt}
            
            # 處理完整路徑
            all_models = merge_models()
            model_info = all_models.get(model_name, {})
            endpoint_path = model_info.get('full_path', f"black-forest-labs/{model_name}")
            
            response = requests.post(
                f"{base_url}/models/{endpoint_path}",
                headers=headers,
                json=data,
                timeout=30
            )
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                test_result.update({
                    'available': True,
                    'response_time': response_time,
                    'details': {
                        'status_code': response.status_code,
                        'test_prompt': test_prompt,
                        'endpoint': endpoint_path
                    }
                })
            else:
                test_result.update({
                    'available': False,
                    'error': f"HTTP {response.status_code}",
                    'details': {'status_code': response.status_code}
                })
        else:
            # OpenAI Compatible API 調用
            response = client.images.generate(
                model=model_name,
                prompt=test_prompt,
                n=1,
                size="1024x1024"
            )
            end_time = time.time()
            response_time = end_time - start_time
            
            test_result.update({
                'available': True,
                'response_time': response_time,
                'details': {
                    'image_count': len(response.data),
                    'test_prompt': test_prompt,
                    'image_url': response.data[0].url if response.data else None
                }
            })
            
    except Exception as e:
        error_msg = str(e)
        test_result.update({
            'available': False,
            'error': error_msg,
            'details': {
                'error_type': 'generation_failed',
                'test_prompt': test_prompt
            }
        })
    
    return test_result

# 其餘原有函數保持不變，但需要更新模型引用...
def init_session_state():
    """初始化會話狀態"""
    if 'api_config' not in st.session_state:
        st.session_state.api_config = {
            'provider': 'Navy',
            'api_key': '',
            'base_url': 'https://api.navy/v1',
            'validated': False
        }
    
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'favorite_images' not in st.session_state:
        st.session_state.favorite_images = []
    
    if 'model_test_results' not in st.session_state:
        st.session_state.model_test_results = {}
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "生成器"
    
    if 'discovered_models' not in st.session_state:
        st.session_state.discovered_models = {}
    
    if 'favorite_models' not in st.session_state:
        st.session_state.favorite_models = []

def init_api_client():
    """初始化 API 客戶端"""
    if 'api_config' in st.session_state and st.session_state.api_config.get('api_key'):
        config = st.session_state.api_config
        if config['provider'] == "Hugging Face":
            return None  # Hugging Face 使用直接請求
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
    st.subheader("🔑 API 設置")
    
    provider_options = list(API_PROVIDERS.keys())
    current_provider = st.session_state.api_config.get('provider', 'Navy')
    selected_provider = st.selectbox(
        "選擇 API 提供商",
        options=provider_options,
        index=provider_options.index(current_provider) if current_provider in provider_options else 1,
        format_func=lambda x: f"{API_PROVIDERS[x]['icon']} {API_PROVIDERS[x]['name']}"
    )
    
    provider_info = API_PROVIDERS[selected_provider]
    st.info(f"📋 {provider_info['description']}")
    
    current_key = st.session_state.api_config.get('api_key', '')
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
    
    base_url_input = st.text_input(
        "API 端點 URL",
        value=st.session_state.api_config.get('base_url', provider_info['base_url_default']),
        placeholder=provider_info['base_url_default'],
        help="API 服務的基礎 URL"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        save_btn = st.button("💾 保存設置", type="primary")
    
    with col2:
        test_btn = st.button("🧪 測試連接")
    
    with col3:
        clear_btn = st.button("🗑️ 清除設置", type="secondary")
    
    if save_btn:
        if not api_key_input and not current_key:
            st.error("❌ 請輸入 API 密鑰")
        elif not base_url_input:
            st.error("❌ 請輸入 API 端點 URL")
        else:
            final_api_key = api_key_input if api_key_input else current_key
            st.session_state.api_config = {
                'provider': selected_provider,
                'api_key': final_api_key,
                'base_url': base_url_input,
                'validated': False
            }
            st.success("✅ API 設置已保存")
            # 清除舊的模型測試結果和發現結果
            st.session_state.model_test_results = {}
            st.session_state.discovered_models = {}
            rerun_app()
    
    if test_btn:
        test_api_key = api_key_input if api_key_input else current_key
        if not test_api_key:
            st.error("❌ 請先輸入 API 密鑰")
        elif not base_url_input:
            st.error("❌ 請輸入 API 端點 URL")
        else:
            with st.spinner("正在測試 API 連接..."):
                is_valid, message = validate_api_key(test_api_key, base_url_input, selected_provider)
                if is_valid:
                    st.success(f"✅ {message}")
                    st.session_state.api_config['validated'] = True
                    
                    # API 驗證成功後，提供自動發現選項
                    if st.button("🔍 立即發現可用模型", key="auto_discover_after_test"):
                        st.session_state.api_config = {
                            'provider': selected_provider,
                            'api_key': test_api_key,
                            'base_url': base_url_input,
                            'validated': True
                        }
                        auto_discover_models()
                else:
                    st.error(f"❌ {message}")
                    st.session_state.api_config['validated'] = False
    
    if clear_btn:
        st.session_state.api_config = {
            'provider': 'Navy',
            'api_key': '',
            'base_url': 'https://api.navy/v1',
            'validated': False
        }
        st.session_state.model_test_results = {}
        st.session_state.discovered_models = {}
        st.success("🗑️ API 設置已清除")
        rerun_app()
    
    # 顯示當前狀態
    if st.session_state.api_config['api_key']:
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            if st.session_state.api_config.get('validated', False):
                st.success("🟢 API 已驗證")
            else:
                st.warning("🟡 API 未驗證")
        
        with status_col2:
            st.info(f"🔧 使用: {provider_info['name']}")

# 初始化
init_session_state()

# 初始化 API 客戶端
client = init_api_client()
api_configured = client is not None or (st.session_state.api_config.get('provider') == "Hugging Face" and st.session_state.api_config.get('api_key'))

# 側邊欄
with st.sidebar:
    show_api_settings()
    st.markdown("---")
    
    # API 狀態顯示
    if api_configured:
        st.success("🟢 API 已配置")
        provider = st.session_state.api_config.get('provider', 'Unknown')
        st.caption(f"使用: {API_PROVIDERS.get(provider, {}).get('name', provider)}")
    else:
        st.error("🔴 API 未配置")
    
    # 快速發現按鈕
    st.markdown("### 🔍 快速操作")
    if api_configured:
        if st.button("🔍 發現模型", use_container_width=True):
            auto_discover_models()
    else:
        st.info("配置 API 後可發現模型")
    
    # 模型統計
    st.markdown("### 📊 模型統計")
    all_models = merge_models()
    base_count = len([m for m in all_models.values() if m.get('source') == 'base'])
    discovered_count = len([m for m in all_models.values() if m.get('auto_discovered')])
    
    st.metric("基礎模型", base_count)
    st.metric("發現模型", discovered_count)
    st.metric("總模型數", len(all_models))

# 主標題
st.title("🎨 Flux AI 圖像生成器 Pro - 自動模型發現")

# 頁面導航
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 圖像生成", 
    "🔍 模型發現", 
    "📋 模型列表",
    "🧪 模型測試", 
    "💡 幫助"
])

# 圖像生成頁面
with tab1:
    st.subheader("🚀 圖像生成")
    if not api_configured:
        st.warning("⚠️ 請先在側邊欄配置 API 密鑰")
        st.info("配置完成後即可開始生成圖像")
    else:
        # 使用合併後的模型列表
        all_models = merge_models()
        
        st.success(f"📋 當前可用 {len(all_models)} 個 Flux 模型")
        
        # 模型選擇
        model_options = list(all_models.keys())
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = model_options[0] if model_options else None
        
        if st.session_state.selected_model not in model_options:
            st.session_state.selected_model = model_options[0] if model_options else None
        
        selected_model = st.selectbox(
            "選擇模型:",
            options=model_options,
            index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
            format_func=lambda x: f"{all_models[x].get('icon', '🤖')} {all_models[x].get('name', x)}"
        )
        
        st.session_state.selected_model = selected_model
        
        # 顯示模型信息
        model_info = all_models[selected_model]
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.info(f"**描述**: {model_info.get('description', 'N/A')}")
        with col_info2:
            source_labels = {
                'base': '🏠 基礎',
                'api_discovery': '🤖 API發現',
                'huggingface': '🤗 HuggingFace',
                'auto_discovered': '🔍 自動發現'
            }
            source = model_info.get('source', 'unknown')
            st.info(f"**來源**: {source_labels.get(source, source)}")
        
        # 提示詞輸入
        prompt = st.text_area(
            "輸入提示詞:",
            height=100,
            placeholder="描述您想要生成的圖像..."
        )
        
        # 生成按鈕
        if st.button("🚀 生成圖像", type="primary", use_container_width=True):
            if not prompt.strip():
                st.error("請輸入提示詞")
            else:
                st.info("🚧 圖像生成功能正在開發中，當前主要展示自動模型發現功能")

# 模型發現頁面
with tab2:
    show_model_discovery_panel()

# 模型列表頁面  
with tab3:
    show_discovered_models_list()

# 模型測試頁面
with tab4:
    st.subheader("🧪 模型測試")
    if not api_configured:
        st.warning("⚠️ 請先配置 API 密鑰")
    else:
        all_models = merge_models()
        if all_models:
            # 批量測試
            if st.button("🧪 測試所有模型", type="primary"):
                model_ids = list(all_models.keys())[:10]  # 限制測試數量
                test_discovered_models(model_ids)
            
            # 顯示測試結果
            if st.session_state.get('model_test_results'):
                st.subheader("📊 測試結果")
                for model_id, result in st.session_state.model_test_results.items():
                    model_info = all_models.get(model_id, {})
                    
                    col_model, col_status, col_time = st.columns([2, 1, 1])
                    
                    with col_model:
                        st.write(f"{model_info.get('icon', '🤖')} {model_info.get('name', model_id)}")
                    
                    with col_status:
                        if result.get('available'):
                            st.success("✅ 可用")
                        else:
                            st.error("❌ 不可用")
                    
                    with col_time:
                        if result.get('response_time'):
                            st.metric("響應時間", f"{result['response_time']:.2f}s")
        else:
            st.info("請先發現一些模型")

# 幫助頁面
with tab5:
    st.subheader("💡 使用幫助")
    
    st.markdown("### 🔍 自動模型發現")
    st.markdown("""
    **功能特色:**
    - **智能掃描**: 自動掃描 API 端點的所有可用模型
    - **模式識別**: 智能識別 Flux 相關模型並分類
    - **實時更新**: 動態更新模型列表，無需手動維護
    - **多平台支持**: 支持 OpenAI、Hugging Face、Together AI 等多個平台
    
    **使用步驟:**
    1. 配置 API 密鑰
    2. 點擊「開始自動發現」
    3. 系統自動掃描和分析模型
    4. 查看發現的模型列表
    5. 測試模型可用性
    6. 開始生成圖像
    """)
    
    st.markdown("### 🎯 支持的模型模式")
    st.markdown("""
    系統能自動識別以下類型的 Flux 模型：
    - **FLUX.1 Schnell**: 快速生成模型
    - **FLUX.1 Dev**: 開發版本模型  
    - **FLUX.1 Pro**: 專業版本模型
    - **FLUX.1 Kontext**: 上下文理解模型
    - **FLUX.2**: 下一代模型
    - **自定義微調**: Anime、Realism、Art 等風格化模型
    """)
    
    st.markdown("### 🚀 Koyeb 部署優勢")
    st.markdown("""
    - **Scale-to-Zero**: 閒置時自動縮減成本
    - **全球部署**: 50+ 個地區可選
    - **自動縮放**: 根據需求自動調整資源
    - **安全可靠**: 自動 HTTPS 和環境變量加密
    """)

# 頁腳
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🚀 Koyeb 部署 | 🔍 自動模型發現 | 🎨 Flux AI Pro
</div>
""", unsafe_allow_html=True)
