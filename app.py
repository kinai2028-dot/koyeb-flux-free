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
                - 圖像品質: {service_info['quality']}
                """)
        else:
            api_token = None
            st.info("""
            **演示模式**
            - 無 API 成本
            - 即時響應
            - 佔位符圖像
            - 適合測試部署
            """)
        
        st.divider()
        
        # 優化設置
        st.subheader("🎛️ 性能優化")
        
        enable_cache = st.checkbox("啟用結果緩存", value=True, help="減少重複 API 調用")
        compress_images = st.checkbox("壓縮圖像", value=True, help="減少內存使用")
        batch_processing = st.checkbox("批次處理", value=False, help="適合多個請求")
        
        # 成本追蹤
        st.subheader("💰 成本追蹤")
        st.metric("Koyeb 費用", "$0.00", "免費額度")
        st.metric("API 成本", "變動", "依使用量")
        st.metric("總運行時間", "24/7", "不休眠")
        
        # 部署信息
        st.subheader("📍 部署信息")
        st.write("**區域**: 自動選擇")
        st.write("**縮放**: 自動")
        st.write("**SSL**: 自動")
        st.write("**域名**: .koyeb.app")
    
    # 主界面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 AI 圖像生成")
        
        # 提示詞輸入
        prompt = st.text_area(
            "輸入提示詞:",
            placeholder="A beautiful mountain landscape with a serene lake",
            height=100,
            help="描述您想要生成的圖像內容"
        )
        
        # 快速提示詞模板
        quick_prompts = {
            "自然風景": "A serene mountain landscape with a crystal clear lake reflecting the sky",
            "現代建築": "Modern glass skyscraper with sleek geometric design against blue sky", 
            "抽象藝術": "Abstract geometric patterns with vibrant colors and flowing lines",
            "科技風格": "Futuristic digital interface with holographic elements and neon lights",
            "簡約設計": "Minimalist design with clean lines and neutral color palette"
        }
        
        selected_template = st.selectbox("或選擇快速模板:", ["自訂"] + list(quick_prompts.keys()))
        
        if selected_template != "自訂":
            prompt = quick_prompts[selected_template]
        
        # 生成控制
        col_gen1, col_gen2, col_gen3 = st.columns([2, 1, 1])
        
        with col_gen1:
            generate_btn = st.button(
                "🎨 生成圖像",
                type="primary",
                use_container_width=True,
                disabled=not prompt.strip()
            )
        
        with col_gen2:
            if st.button("🎲 隨機", use_container_width=True):
                import random
                prompt = random.choice(list(quick_prompts.values()))
                st.rerun()
        
        with col_gen3:
            if selected_service == "演示模式":
                est_time = "即時"
            elif selected_service in API_SERVICES:
                est_time = API_SERVICES[selected_service]["avg_time"]
            else:
                est_time = "10-30秒"
            
            st.metric("預估時間", est_time)
        
        # 圖像生成邏輯
        if generate_btn and prompt.strip():
            # 檢查 API Token (演示模式除外)
            if selected_service != "演示模式" and not api_token:
                st.error(f"請輸入 {selected_service} 的 API Token")
            else:
                with st.spinner(f"使用 {selected_service} 生成中..."):
                    start_time = time.time()
                    
                    # 調用相應的生成服務
                    if selected_service == "演示模式":
                        result = simulate_demo_generation(prompt)
                    elif selected_service == "Hugging Face Inference":
                        result = call_huggingface_api(prompt, api_token)
                    elif selected_service == "Replicate":
                        result = call_replicate_api(prompt, api_token)
                    else:
                        result = {"status": "error", "message": "服務暫未實現"}
                    
                    generation_time = time.time() - start_time
                    
                    if result["status"] == "success":
                        st.success(f"✅ 生成成功！耗時: {generation_time:.1f}秒")
                        
                        # 顯示圖像
                        image = result["image"]
                        
                        # 圖像壓縮 (如果啟用)
                        if compress_images and selected_service != "演示模式":
                            # 壓縮圖像以節省內存
                            image = image.resize((512, 512), Image.Resampling.LANCZOS)
                        
                        st.image(
                            image,
                            caption=f"提示詞: {prompt} | 服務: {result.get('service', selected_service)}",
                            use_column_width=True
                        )
                        
                        # 下載功能
                        img_buffer = BytesIO()
                        image.save(img_buffer, format="PNG", optimize=True)
                        img_buffer.seek(0)
                        
                        st.download_button(
                            "📥 下載圖像",
                            data=img_buffer,
                            file_name=f"flux_koyeb_{int(time.time())}.png",
                            mime="image/png"
                        )
                        
                        # 緩存結果 (如果啟用)
                        if enable_cache:
                            if 'generated_cache' not in st.session_state:
                                st.session_state.generated_cache = []
                            
                            st.session_state.generated_cache.append({
                                'prompt': prompt,
                                'service': selected_service,
                                'time': time.strftime('%H:%M:%S'),
                                'generation_time': f"{generation_time:.1f}s"
                            })
                            
                            # 限制緩存大小
                            if len(st.session_state.generated_cache) > 5:
                                st.session_state.generated_cache.pop(0)
                    
                    else:
                        st.error(f"❌ 生成失敗: {result['message']}")
                        
                        # 提供解決方案
                        st.info("""
                        **可能的解決方案:**
                        - 檢查 API Token 是否正確
                        - 嘗試切換演示模式測試
                        - 確認網絡連接正常
                        - 聯繫 API 服務提供商
                        """)
    
    with col2:
        st.subheader("💡 Koyeb 優勢")
        
        st.markdown("""
        **🚀 Koyeb 特色:**
        - 全球 50+ 地區部署
        - 自動縮放 & Scale-to-Zero  
        - 內建負載均衡
        - 自動 HTTPS & SSL
        - Git 驅動部署
        
        **💰 成本優化:**
        - 免費實例: $0.00/月
        - 按需付費: $0.0036/小時起
        - 無閒置費用 (Scale-to-Zero)
        - 無基礎設施管理
        
        **📈 性能監控:**
        """)
        
        # 顯示當前系統狀態
        if "error" not in system_info:
            col_cpu, col_mem = st.columns(2)
            with col_cpu:
                st.metric("CPU 使用", f"{system_info['cpu_percent']:.1f}%")
            with col_mem:
                st.metric("內存使用", f"{system_info['memory_percent']:.1f}%")
        
        # 生成歷史 (如果有緩存)
        if 'generated_cache' in st.session_state and st.session_state.generated_cache:
            st.subheader("📚 生成歷史")
            
            for i, item in enumerate(reversed(st.session_state.generated_cache)):
                with st.expander(f"記錄 {i+1} - {item['time']}"):
                    st.write(f"**提示詞**: {item['prompt'][:50]}...")
                    st.write(f"**服務**: {item['service']}")
                    st.write(f"**耗時**: {item['generation_time']}")
        
        # 部署指南
        st.subheader("🛠️ 部署指南")
        
        with st.expander("📖 快速部署"):
            st.code("""
# 1. 推送代碼到 GitHub
git init
git add .
git commit -m "Flux AI Koyeb"
git push origin main

# 2. 在 Koyeb 控制台
# - 點擊 "Create Service"
# - 選擇 GitHub 倉庫
# - 選擇 CPU 實例類型
# - 設置環境變量
# - 點擊 Deploy
            """, language="bash")

if __name__ == "__main__":
    main()
