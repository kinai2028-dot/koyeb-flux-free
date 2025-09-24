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
    page_title="Flux AI (優化佈局)",
    page_icon="🎨",
    layout="wide"
)

# API 提供商配置
API_PROVIDERS = {
    "Pollinations.ai": {"name": "Pollinations.ai", "base_url_default": "https://pollinations.ai/v1"},
    "Navy": {"name": "Navy API", "base_url_default": "https://api.navy/v1"},
    "OpenAI Compatible": {"name": "OpenAI Compatible API", "base_url_default": "https://api.openai.com/v1"},
    "Custom": {"name": "自定義 API", "base_url_default": ""}
}

# 模型列表
FLUX_MODELS = {
    "flux.1.1-pro": {"name": "Flux 1.1 Pro"},
    "flux.1-schnell": {"name": "Flux 1 Schnell"},
    "flux-por": {"name": "Flux POR (藝術風格)"},
    "pollinations-custom": {"name": "Pollinations 自定義模型"}
}

# ==============================================================================
# 2. Session State 與 UI 函式
# ==============================================================================

def init_session_state():
    """初始化會話狀態"""
    if 'api_provider' not in st.session_state:
        st.session_state.api_provider = st.secrets.get("API_PROVIDER", "Pollinations.ai")
    if 'api_key' not in st.session_state:
        st.session_state.api_key = st.secrets.get("API_KEY", "")
    if 'base_url' not in st.session_state:
        default_url = API_PROVIDERS.get(st.session_state.api_provider, {}).get("base_url_default", "")
        st.session_state.base_url = st.secrets.get("BASE_URL", default_url)
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []

def show_api_settings():
    """顯示 API 設置界面"""
    st.subheader("🔌 後端 API 設置")
    # ... (此處程式碼與上一版相同，省略以保持簡潔)

# ==============================================================================
# 3. 主應用程式執行流程 (佈局更新)
# ==============================================================================

def main():
    init_session_state()

    with st.sidebar:
        show_api_settings()

    st.title("🎨 Flux AI 生成器 (優化佈局)")

    if not st.session_state.api_key or not st.session_state.base_url:
        st.error("❌ 請在側邊欄或 Koyeb 環境變數中設置您的後端 API 密鑰和 URL。")
        st.stop()
    
    try:
        client = OpenAI(api_key=st.session_state.api_key, base_url=st.session_state.base_url)
    except Exception as e:
        st.error(f"無法初始化 API 客戶端: {e}")
        st.stop()

    # --- 圖像生成介面 ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. 選擇模型")
        default_model_index = list(FLUX_MODELS.keys()).index("flux.1.1-pro")
        model = st.selectbox(
            "選擇模型 (請確保後端 API 支持)", 
            options=list(FLUX_MODELS.keys()), 
            index=default_model_index, 
            format_func=lambda x: FLUX_MODELS[x]['name'],
            label_visibility="collapsed"
        )
        
        st.subheader("2. 描述您的創意")
        prompt = st.text_area("在這裡輸入提示詞...", height=120, label_visibility="collapsed")

        # ==========================================================
        #  位置已調整：智能參數設置區塊，預設展開
        # ==========================================================
        st.subheader("3. 調整智能參數")
        with st.expander("⚙️ 參數設置", expanded=True):
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                num_images = st.slider("生成數量", 1, 4, 1)
            with param_col2:
                selected_size = st.selectbox("圖像尺寸", ["1024x1024", "1152x896", "896x1152"])
        
        # --- 生成按鈕 ---
        st.subheader("4. 開始生成")
        if st.button("🚀 生成圖像", type="primary", disabled=not prompt, use_container_width=True):
            with st.spinner(f"正在向 {st.session_state.api_provider} 發送請求..."):
                try:
                    response = client.images.generate(
                        model=model,
                        prompt=prompt,
                        n=num_images,
                        size=selected_size
                    )
                    
                    st.success(f"🎉 成功生成 {len(response.data)} 張圖像！")
                    
                    # 顯示生成的圖像
                    for i, img_data in enumerate(response.data):
                        st.image(img_data.url, caption=f"圖像 {i+1}")
                    
                    # 將第一張圖添加到歷史記錄
                    if response.data:
                        st.session_state.generation_history.insert(0, {'prompt': prompt, 'model': model, 'url': response.data[0].url})
                        
                except Exception as e:
                    st.error(f"❌ 生成失敗: {e}")

    with col2:
        st.subheader("📜 最近生成歷史")
        if not st.session_state.generation_history:
            st.info("暫無歷史記錄。")
        else:
            for item in st.session_state.generation_history[:5]:
                st.image(item['url'], caption=item['prompt'][:50] + "...")
                st.markdown("---")


if __name__ == "__main__":
    main()
