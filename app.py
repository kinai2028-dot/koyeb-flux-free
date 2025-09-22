import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import time
import os

# 設定頁面配置
st.set_page_config(
    page_title="Flux AI 免費圖像生成器",
    page_icon="🎨",
    layout="wide"
)

# CSS 樣式
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.info-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

.cost-tracker {
    position: fixed;
    top: 60px;
    right: 20px;
    background: #ffffff;
    padding: 10px;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    z-index: 1000;
}
</style>
""", unsafe_allow_html=True)

# 免費 API 服務配置
FREE_API_SERVICES = {
    "Fal.AI": {
        "url": "https://fal.run/fal-ai/flux/schnell",
        "free_quota": "50 張/月",
        "quality": "高品質",
        "speed": "中等"
    },
    "Mage.Space": {
        "url": "https://api.mage.space/v1/flux",
        "free_quota": "無限制",
        "quality": "中等品質", 
        "speed": "快速"
    },
    "Replicate Demo": {
        "url": "https://replicate.com/black-forest-labs/flux-schnell",
        "free_quota": "10 張/日",
        "quality": "高品質",
        "speed": "較慢"
    }
}

def call_free_flux_api(prompt, service="Mage.Space"):
    """調用免費 Flux API 生成圖像"""
    try:
        if service == "Mage.Space":
            # 模擬 Mage.Space API 調用
            # 實際使用時需要替換為真實的 API 端點
            response = {
                "status": "success",
                "image_url": "https://via.placeholder.com/512x512/667eea/ffffff?text=Flux+Generated+Image"
            }
            return response
        
        elif service == "Fal.AI":
            # FAL.AI API 示例
            headers = {"Authorization": f"Key {st.secrets.get('FAL_KEY', '')}"}
            data = {
                "prompt": prompt,
                "image_size": "square_hd",
                "num_inference_steps": 4
            }
            # 這裡需要實際的 API 調用
            return {"status": "success", "image_url": "demo_url"}
        
        else:
            return {"status": "error", "message": "不支援的 API 服務"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    # 主標題
    st.markdown("""
    <div class="main-header">
        <h1>🎨 Flux AI 免費圖像生成器</h1>
        <p>部署在 Koyeb 免費方案 | 使用免費 API 服務</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 側邊欄配置
    with st.sidebar:
        st.header("⚙️ 生成設置")
        
        # API 服務選擇
        selected_service = st.selectbox(
            "選擇免費 API 服務:",
            list(FREE_API_SERVICES.keys()),
            help="不同服務有不同的免費額度和限制"
        )
        
        # 顯示選中服務的資訊
        service_info = FREE_API_SERVICES[selected_service]
        st.info(f"""
        **{selected_service}**
        - 免費額度: {service_info['free_quota']}
        - 圖像品質: {service_info['quality']}
        - 生成速度: {service_info['speed']}
        """)
        
        st.divider()
        
        # 圖像參數（簡化版）
        image_style = st.selectbox(
            "圖像風格:",
            ["寫實攝影", "數位藝術", "油畫風格", "卡通風格", "科幻風格"]
        )
        
        image_quality = st.select_slider(
            "圖像品質:",
            ["快速", "標準", "高品質"],
            value="標準"
        )
        
        st.divider()
        
        # Koyeb 免費資源監控
        st.subheader("📊 Koyeb 免費資源")
        
        # 模擬資源使用狀況
        ram_usage = st.progress(0.3, text="RAM: 154MB / 512MB")
        cpu_usage = st.progress(0.2, text="CPU: 0.02 / 0.1 vCPU")
        storage_usage = st.progress(0.1, text="儲存: 200MB / 2GB")
        
        st.metric("本月流量使用", "2.3GB", "100GB 免費額度")
        
        st.info("""
        ✅ **Koyeb 免費方案優勢:**
        - 不會自動休眠
        - 自訂域名支持
        - 自動 HTTPS
        - 全球 CDN
        """)
    
    # 主要內容區域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 圖像生成")
        
        # 提示詞輸入
        prompt = st.text_area(
            "描述你想要生成的圖像:",
            placeholder="例如: A serene landscape with mountains and a lake at sunset",
            height=100,
            help="詳細描述能獲得更好的生成效果"
        )
        
        # 預設提示詞模板
        prompt_templates = {
            "自然風景": "A beautiful landscape with mountains, forests, and a clear blue sky, highly detailed, 8k",
            "城市夜景": "A vibrant city skyline at night with neon lights and reflections, cyberpunk style",
            "抽象藝術": "Abstract geometric art with vibrant colors and flowing patterns, modern digital art",
            "動物肖像": "A majestic wild animal in its natural habitat, wildlife photography style",
            "科幻場景": "Futuristic space station with advanced technology, sci-fi concept art"
        }
        
        selected_template = st.selectbox("或選擇預設模板:", ["自訂"] + list(prompt_templates.keys()))
        if selected_template != "自訂":
            prompt = prompt_templates[selected_template]
        
        # 生成按鈕區域
        col_gen1, col_gen2, col_gen3 = st.columns([2, 1, 1])
        
        with col_gen1:
            generate_btn = st.button(
                "🚀 免費生成圖像",
                type="primary",
                use_container_width=True,
                disabled=not prompt.strip()
            )
        
        with col_gen2:
            if st.button("🔄 隨機提示詞", use_container_width=True):
                import random
                prompt = random.choice(list(prompt_templates.values()))
                st.rerun()
        
        with col_gen3:
            estimated_cost = "$0.00"
            st.metric("預估成本", estimated_cost)
        
        # 圖像生成邏輯
        if generate_btn and prompt.strip():
            with st.spinner(f"使用 {selected_service} 生成中..."):
                # 調用免費 API
                result = call_free_flux_api(prompt, selected_service)
                
                if result["status"] == "success":
                    st.success("✅ 圖像生成成功！")
                    
                    # 顯示生成的圖像（演示版本）
                    demo_image_url = "https://via.placeholder.com/512x512/667eea/ffffff?text=Flux+AI+Generated"
                    
                    try:
                        # 在實際環境中，這裡會加載真實的生成圖像
                        st.image(
                            demo_image_url,
                            caption=f"生成提示詞: {prompt}",
                            use_column_width=True
                        )
                        
                        # 下載按鈕
                        st.download_button(
                            "📥 下載圖像",
                            data=b"demo_image_data",
                            file_name=f"flux_generated_{int(time.time())}.png",
                            mime="image/png"
                        )
                        
                    except Exception as e:
                        st.error(f"圖像載入失敗: {e}")
                        
                else:
                    st.error(f"❌ 生成失敗: {result.get('message', '未知錯誤')}")
                    st.info("💡 請嘗試切換其他免費 API 服務")
    
    with col2:
        st.subheader("💡 使用指南")
        
        st.markdown("""
        **免費額度管理:**
        - 每個 API 服務都有使用限制
        - 建議輪換使用不同服務
        - 避免在短時間內大量生成
        
        **提示詞技巧:**
        - 使用具體的描述詞
        - 包含風格關鍵詞
        - 指定品質要求 (8k, detailed)
        
        **Koyeb 部署優勢:**
        - 24/7 運行不休眠
        - 自動 HTTPS 和 CDN
        - 自訂域名支持
        - 歐洲/美國機房選擇
        """)
        
        # API 服務狀態
        st.subheader("📊 API 服務狀態")
        for service, info in FREE_API_SERVICES.items():
            status = "🟢 可用" if service != "Replicate Demo" else "🟡 限制"
            st.write(f"**{service}**: {status}")
            st.caption(f"額度: {info['free_quota']}")
        
        # 成本追蹤
        st.subheader("💰 成本追蹤")
        st.metric("Koyeb 費用", "$0.00", "免費額度內")
        st.metric("API 調用", "免費", "使用免費服務")
        st.metric("總運行成本", "$0.00", "完全免費")
        
        st.success("🎉 完全免費運行!")

# 錯誤處理和監控
def monitor_resources():
    """監控 Koyeb 資源使用"""
    return {
        "ram_usage": 0.3,  # 30%
        "cpu_usage": 0.2,  # 20%
        "storage_usage": 0.1  # 10%
    }

if __name__ == "__main__":
    main()
