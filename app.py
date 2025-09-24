import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
import datetime

# ==============================================================================
# 1. 應用程式全域設定
# ==============================================================================

st.set_page_config(
    page_title="Flux AI 圖像生成器 (Koyeb 免費版)",
    page_icon="☁️",
    layout="wide"
)

# API 提供商配置 (用戶可自行選擇)
API_PROVIDERS = {
    "OpenAI Compatible": {"name": "OpenAI Compatible API", "base_url_default": "https://api.openai.com/v1"},
    "Navy": {"name": "Navy API", "base_url_default": "https://api.navy/v1"},
    "Custom": {"name": "自定義 API", "base_url_default": ""}
}

# 模型列表 (僅作為選項，實際是否可用取決於後端 API)
FLUX_MODELS = {
    "flux.1-schnell": {"name": "FLUX.1 Schnell"},
    "flux.1-krea-dev": {"name": "FLUX.1 Krea Dev"},
    "flux.1.1-pro": {"name": "FLUX.1.1 Pro"}
}

# ==============================================================================
# 2. Session State 與 UI 函式
# ==============================================================================

def init_session_state():
    """初始化會話狀態"""
    if 'api_key' not in st.session_state:
        # 優先從 Koyeb 的 Secrets 中讀取
        st.session_state.api_key = st.secrets.get("API_KEY", "")
    if 'base_url' not in st.session_state:
        st.session_state.base_url = st.secrets.get("BASE_URL", API_PROVIDERS["Navy"]["base_url_default"])
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []

def show_api_settings():
    """顯示 API 設置界面"""
    st.subheader("🔑 後端 API 設置")
    st.info("此應用僅為前端介面，實際圖像生成由後端 API 完成。")

    # 允許用戶覆蓋預設的 API Key 和 URL
    api_key_input = st.text_input(
        "API 密鑰 (可留空以使用環境變數)", 
        type="password", 
        value=st.session_state.api_key
    )
    base_url_input = st.text_input(
        "API 端點 URL", 
        value=st.session_state.base_url
    )

    if st.button("💾 保存設置"):
        st.session_state.api_key = api_key_input
        st.session_state.base_url = base_url_input
        st.success("設置已更新！")
        st.rerun()

# ==============================================================================
# 3. 主應用程式執行流程
# ==============================================================================

def main():
    init_session_state()

    with st.sidebar:
        show_api_settings()

    st.title("☁️ Flux AI 生成器 (UI on Koyeb Free Tier)")

    if not st.session_state.api_key or not st.session_state.base_url:
        st.error("❌ 請在側邊欄或 Koyeb 環境變數中設置您的後端 API 密鑰和 URL。")
        st.stop()
    
    # 初始化 OpenAI client
    try:
        client = OpenAI(api_key=st.session_state.api_key, base_url=st.session_state.base_url)
    except Exception as e:
        st.error(f"無法初始化 API 客戶端: {e}")
        st.stop()

    # --- 圖像生成介面 ---
    st.subheader("✏️ 輸入您的創意")
    model = st.selectbox("選擇模型 (請確保後端 API 支持)", options=list(FLUX_MODELS.keys()), format_func=lambda x: FLUX_MODELS[x]['name'])
    prompt = st.text_area("描述您想要的圖像...", height=120)
    
    if st.button("🚀 生成圖像", type="primary", disabled=not prompt):
        with st.spinner(f"正在向後端 API 發送請求..."):
            try:
                response = client.images.generate(
                    model=model,
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )
                image_url = response.data[0].url
                st.success("🎉 圖像生成成功！")
                st.image(image_url, caption=f"模型: {model}")
                # 將結果添加到歷史記錄
                st.session_state.generation_history.insert(0, {'prompt': prompt, 'model': model, 'url': image_url})
            except Exception as e:
                st.error(f"❌ 生成失敗: {e}")

    # --- 歷史記錄 ---
    if st.session_state.generation_history:
        st.markdown("---")
        st.subheader("📜 最近生成")
        for item in st.session_state.generation_history[:5]:
            with st.expander(f"{item['prompt'][:50]...}"):
                st.image(item['url'], width=256)

if __name__ == "__main__":
    main()
