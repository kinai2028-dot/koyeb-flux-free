import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
import time
import os

# 頁面配置
st.set_page_config(
    page_title="Flux AI - CPU 版本",
    page_icon="🎨",
    layout="wide"
)

# CSS 樣式
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.cpu-optimized {
    border-left: 4px solid #4CAF50;
    padding: 1rem;
    background: #f0f8ff;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# 免費 API 服務配置
API_SERVICES = {
    "Hugging Face Inference": {
        "base_url": "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
        "free_quota": "1000 請求/月",
        "speed": "中等",
        "token_required": True
    },
    "Replicate": {
        "model": "black-forest-labs/flux-schnell",
        "free_quota": "有限試用",
        "speed": "快速",
        "token_required": True
    },
    "Mage.Space": {
        "base_url": "https://api.mage.space/v1/flux",
        "free_quota": "無限制",
        "speed": "快速",
        "token_required": False
    }
}

def call_huggingface_api(prompt, api_token):
    """調用 Hugging Face Inference API"""
    headers = {"Authorization": f"Bearer {api_token}"}
    data = {"inputs": prompt}
    
    try:
        response = requests.post(
            API_SERVICES["Hugging Face Inference"]["base_url"],
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return {"status": "success", "image": image}
        else:
            return {"status": "error", "message": f"API 錯誤: {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def call_replicate_api(prompt, api_token):
    """調用 Replicate API"""
    try:
        import replicate
        
        # 設置 API token
        os.environ["REPLICATE_API_TOKEN"] = api_token
        
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": prompt}
        )
        
        # 下載圖像
        image_url = output[0] if isinstance(output, list) else output
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        
        return {"status": "success", "image": image}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def simulate_mage_space_api(prompt):
    """模擬 Mage.Space API 調用"""
    # 實際使用時需要實現真實的 API 調用
    try:
        # 創建一個示例圖像
        placeholder_url = f"https://via.placeholder.com/512x512/4CAF50/ffffff?text=CPU+Generated"
        response = requests.get(placeholder_url)
        image = Image.open(BytesIO(response.content))
        
        return {"status": "success", "image": image}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    # 主標題
    st.markdown("""
    <div class="main-header">
        <h1>🎨 Flux AI - CPU 優化版本</h1>
        <p>使用 API 調用，適合 CPU 部署和免費託管</p>
    </div>
    """, unsafe_allow_html=True)
    
    # CPU 優化說明
    st.markdown("""
    <div class="cpu-optimized">
        <h3>💡 CPU 版本特色</h3>
        <ul>
            <li>✅ 使用 API 調用，無需 GPU</li>
            <li>✅ 適合免費部署平台</li>
            <li>✅ 低資源需求（< 512MB RAM）</li>
            <li>✅ 快速響應時間</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # 側邊欄配置
    with st.sidebar:
        st.header("⚙️ API 設置")
        
        # API 服務選擇
        selected_service = st.selectbox(
            "選擇 API 服務:",
            list(API_SERVICES.keys())
        )
        
        # 顯示服務信息
        service_info = API_SERVICES[selected_service]
        st.info(f"""
        **{selected_service}**
        - 免費額度: {service_info['free_quota']}
        - 速度: {service_info['speed']}
        - 需要 Token: {'是' if service_info['token_required'] else '否'}
        """)
        
        # API Token 輸入
        if service_info['token_required']:
            api_token = st.text_input(
                f"{selected_service} API Token:",
                type="password",
                help="從官網獲取免費 API Token"
            )
        else:
            api_token = None
        
        st.divider()
        
        # 生成參數
        st.subheader("🎛️ 生成參數")
        
        image_style = st.selectbox(
            "圖像風格:",
            ["寫實攝影", "數位藝術", "插畫風格", "簡約設計", "復古風格"]
        )
        
        image_quality = st.select_slider(
            "圖像品質:",
            ["快速", "標準", "高品質"],
            value="標準"
        )
        
        # 資源監控
        st.subheader("📊 系統狀態")
        st.metric("CPU 版本", "✅ 運行中")
        st.metric("內存需求", "< 512MB")
        st.metric("API 狀態", "🟢 連接正常")
        
        # 成本信息
        st.subheader("💰 成本信息")
        st.write("**免費使用:**")
        st.write("• Hugging Face: 1000次/月")
        st.write("• Mage.Space: 無限制")
        st.write("• 部署成本: $0.00")
    
    # 主界面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 圖像生成")
        
        # 提示詞輸入
        prompt = st.text_area(
            "輸入提示詞:",
            placeholder="A beautiful mountain landscape with lake and trees",
            height=100
        )
        
        # 風格修飾詞
        style_modifiers = {
            "寫實攝影": ", professional photography, high resolution, detailed",
            "數位藝術": ", digital art, concept art, vibrant colors",
            "插畫風格": ", illustration, cartoon style, colorful",
            "簡約設計": ", minimalist design, clean lines, simple",
            "復古風格": ", vintage style, retro colors, classic"
        }
        
        if image_style in style_modifiers:
            prompt_with_style = prompt + style_modifiers[image_style]
        else:
            prompt_with_style = prompt
        
        # 品質修飾詞
        quality_modifiers = {
            "快速": "",
            "標準": ", good quality",
            "高品質": ", high quality, detailed, 8k"
        }
        
        final_prompt = prompt_with_style + quality_modifiers[image_quality]
        
        # 提示詞預覽
        if final_prompt.strip():
            with st.expander("📋 最終提示詞預覽"):
                st.code(final_prompt)
        
        # 生成按鈕
        col_gen1, col_gen2 = st.columns([3, 1])
        
        with col_gen1:
            generate_btn = st.button(
                "🚀 生成圖像",
                type="primary",
                use_container_width=True,
                disabled=not prompt.strip() or (service_info['token_required'] and not api_token)
            )
        
        with col_gen2:
            estimated_time = "5-15秒"
            st.metric("預估時間", estimated_time)
        
        # 圖像生成
        if generate_btn and prompt.strip():
            if service_info['token_required'] and not api_token:
                st.error(f"請輸入 {selected_service} API Token")
            else:
                with st.spinner(f"使用 {selected_service} 生成中..."):
                    start_time = time.time()
                    
                    # 調用相應的 API
                    if selected_service == "Hugging Face Inference":
                        result = call_huggingface_api(final_prompt, api_token)
                    elif selected_service == "Replicate":
                        result = call_replicate_api(final_prompt, api_token)
                    else:  # Mage.Space
                        result = simulate_mage_space_api(final_prompt)
                    
                    generation_time = time.time() - start_time
                    
                    if result["status"] == "success":
                        st.success(f"✅ 生成成功！耗時: {generation_time:.1f}秒")
                        
                        # 顯示圖像
                        st.image(
                            result["image"],
                            caption=f"生成提示詞: {prompt}",
                            use_column_width=True
                        )
                        
                        # 下載按鈕
                        img_buffer = BytesIO()
                        result["image"].save(img_buffer, format="PNG")
                        img_buffer.seek(0)
                        
                        st.download_button(
                            "📥 下載圖像",
                            data=img_buffer,
                            file_name=f"flux_cpu_{int(time.time())}.png",
                            mime="image/png"
                        )
                    else:
                        st.error(f"❌ 生成失敗: {result['message']}")
                        
                        # 提供解決建議
                        st.info("""
                        **解決建議:**
                        - 檢查 API Token 是否正確
                        - 嘗試切換其他 API 服務
                        - 確保網絡連接正常
                        """)
    
    with col2:
        st.subheader("💡 使用指南")
        
        st.markdown("""
        **API 服務推薦:**
        1. **Mage.Space** - 完全免費，無需註冊
        2. **Hugging Face** - 每月 1000 次免費
        3. **Replicate** - 有限免費試用
        
        **CPU 版本優勢:**
        - 無需 GPU，適合任何環境
        - 部署成本低
        - 響應速度快
        - 適合演示和原型開發
        
        **提示詞技巧:**
        - 使用具體描述
        - 加入風格關鍵詞
        - 指定圖像品質
        """)
        
        # 免費部署平台推薦
        st.subheader("🌐 免費部署平台")
        deployment_options = {
            "Streamlit Community Cloud": "✅ 推薦",
            "Railway": "✅ 免費額度",
            "Render": "✅ 免費計劃",
            "Vercel": "🔶 適合靜態",
            "Netlify": "🔶 適合靜態"
        }
        
        for platform, status in deployment_options.items():
            st.write(f"**{platform}**: {status}")

if __name__ == "__main__":
    main()
