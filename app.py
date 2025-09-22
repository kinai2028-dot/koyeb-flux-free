import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import time
import os
import psutil
import sys

# 頁面配置
st.set_page_config(
    page_title="Flux AI - Koyeb CPU",
    page_icon="🚀",
    layout="wide"
)

# Koyeb 專用 CSS
st.markdown("""
<style>
.koyeb-header {
    background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.resource-monitor {
    background: #f8fafc;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2563eb;
    margin: 1rem 0;
}

.koyeb-stats {
    position: fixed;
    top: 70px;
    right: 20px;
    background: rgba(255,255,255,0.95);
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    font-size: 0.8rem;
    z-index: 1000;
}

.api-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# API 服務配置
API_SERVICES = {
    "Hugging Face Inference": {
        "endpoint": "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
        "free_quota": "1000 請求/月",
        "avg_time": "10-20秒",
        "quality": "高品質"
    },
    "FAL.AI": {
        "endpoint": "https://fal.run/fal-ai/flux/schnell", 
        "free_quota": "50 張/月",
        "avg_time": "5-10秒",
        "quality": "高品質"
    },
    "Replicate": {
        "model": "black-forest-labs/flux-schnell",
        "free_quota": "有限試用",
        "avg_time": "15-30秒", 
        "quality": "最高品質"
    }
}

def get_system_info():
    """獲取 Koyeb 系統資源信息"""
    try:
        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 內存使用
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024**2)
        memory_total_mb = memory.total / (1024**2)
        memory_percent = memory.percent
        
        # 磁盤使用
        disk = psutil.disk_usage('/')
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        
        return {
            "cpu_percent": cpu_percent,
            "memory_used": memory_used_mb,
            "memory_total": memory_total_mb,
            "memory_percent": memory_percent,
            "disk_used": disk_used_gb,
            "disk_total": disk_total_gb,
            "python_version": sys.version.split()[0]
        }
    except Exception as e:
        return {"error": str(e)}

def call_huggingface_api(prompt, hf_token):
    """調用 Hugging Face Inference API"""
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "guidance_scale": 0.0,
            "num_inference_steps": 4
        }
    }
    
    try:
        response = requests.post(
            API_SERVICES["Hugging Face Inference"]["endpoint"],
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return {"status": "success", "image": image, "service": "Hugging Face"}
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def call_replicate_api(prompt, replicate_token):
    """調用 Replicate API"""
    try:
        import replicate
        
        os.environ["REPLICATE_API_TOKEN"] = replicate_token
        
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt,
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 80
            }
        )
        
        # 下載圖像
        image_url = output[0] if isinstance(output, list) else output
        response = requests.get(image_url, timeout=30)
        image = Image.open(BytesIO(response.content))
        
        return {"status": "success", "image": image, "service": "Replicate"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def simulate_demo_generation(prompt):
    """演示模式 - 生成佔位符圖像"""
    try:
        # 創建帶有提示詞的佔位符圖像
        placeholder_text = prompt[:30] + "..." if len(prompt) > 30 else prompt
        placeholder_url = f"https://via.placeholder.com/512x512/2563eb/ffffff?text={placeholder_text.replace(' ', '+')}"
        
        response = requests.get(placeholder_url, timeout=10)
        image = Image.open(BytesIO(response.content))
        
        return {"status": "success", "image": image, "service": "Demo"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    # 主標題
    st.markdown("""
    <div class="koyeb-header">
        <h1>🚀 Flux AI on Koyeb CPU</h1>
        <p>高性能 CPU 實例 | 自動縮放 | 全球部署</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 系統資源監控
    system_info = get_system_info()
    
    if "error" not in system_info:
        st.markdown(f"""
        <div class="koyeb-stats">
            <strong>🖥️ Koyeb 實例狀態</strong><br>
            CPU: {system_info['cpu_percent']:.1f}%<br>
            RAM: {system_info['memory_used']:.0f}MB / {system_info['memory_total']:.0f}MB<br>
            磁盤: {system_info['disk_used']:.1f}GB / {system_info['disk_total']:.1f}GB<br>
            Python: {system_info['python_version']}
        </div>
        """, unsafe_allow_html=True)
    
    # 側邊欄配置
    with st.sidebar:
        st.header("⚙️ Koyeb 配置")
        
        # 實例信息
        st.markdown("""
        <div class="resource-monitor">
            <h4>📊 當前實例</h4>
            <p><strong>類型:</strong> Free / Nano</p>
            <p><strong>vCPU:</strong> 0.1 - 0.25</p>
            <p><strong>RAM:</strong> 256-512MB</p>
            <p><strong>磁盤:</strong> 2-5GB SSD</p>
        </div>
        """, unsafe_allow_html=True)
        
        # API 服務選擇
        st.subheader("🔌 API 服務")
        
        selected_service = st.selectbox(
            "選擇生成服務:",
            ["演示模式"] + list(API_SERVICES.keys())
        )
        
        if selected_service != "演示模式":
            # API Token 輸入
            api_token = st.text_input(
                f"{selected_service} API Token:",
                type="password",
                help="請在官網獲取免費 API Token"
            )
            
            # 服務信息
            if selected_service in API_SERVICES:
                service_info = API_SERVICES[selected_service]
                st.info(f"""
                **{selected_service}**
                - 免費額度: {service_info['free_quota']}
                - 平均耗時: {service_info['avg_time']}
                -
