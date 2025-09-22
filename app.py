import streamlit as st
import torch
from diffusers import FluxPipeline
from huggingface_hub import login
import os
import time
from PIL import Image
import io
import base64

# 頁面配置
st.set_page_config(
    page_title="Flux AI Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 樣式
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.generation-card {
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    background: white;
}

.gpu-status {
    position: fixed;
    top: 70px;
    right: 20px;
    background: rgba(255,255,255,0.95);
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    font-size: 0.8rem;
}

.parameter-section {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# 初始化 Hugging Face
@st.cache_resource
def init_model():
    """初始化並緩存 Flux 模型"""
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        login(token=hf_token)
    
    try:
        # 根據可用 GPU 內存選擇模型
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            if gpu_memory >= 40:  # A100 40GB+
                model_id = "black-forest-labs/FLUX.1-dev"
                torch_dtype = torch.bfloat16
            else:  # 較小的 GPU
                model_id = "black-forest-labs/FLUX.1-schnell"
                torch_dtype = torch.float16
        else:
            st.error("需要 GPU 支持")
            return None
        
        pipeline = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            cache_dir="/tmp/flux_models"
        )
        pipeline.to("cuda")
        
        return pipeline, model_id
    except Exception as e:
        st.error(f"模型加載失敗: {e}")
        return None, None

def generate_image(pipeline, prompt, negative_prompt="", **kwargs):
    """生成圖像"""
    try:
        with torch.no_grad():
            if "schnell" in pipeline.config._name_or_path.lower():
                # Schnell 版本不使用 guidance_scale
                kwargs.pop('guidance_scale', None)
                image = pipeline(
                    prompt=prompt,
                    num_inference_steps=kwargs.get('num_inference_steps', 4),
                    height=kwargs.get('height', 1024),
                    width=kwargs.get('width', 1024),
                    generator=kwargs.get('generator', None)
                ).images[0]
            else:
                # Dev 版本使用完整參數
                image = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=kwargs.get('num_inference_steps', 28),
                    guidance_scale=kwargs.get('guidance_scale', 7.5),
                    height=kwargs.get('height', 1024),
                    width=kwargs.get('width', 1024),
                    generator=kwargs.get('generator', None)
                ).images[0]
        
        return image
    except Exception as e:
        st.error(f"圖像生成失敗: {e}")
        return None

def image_to_base64(image):
    """將 PIL 圖像轉換為 base64"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def main():
    # 主標題
    st.markdown("""
    <div class="main-header">
        <h1>🎨 Flux AI Studio</h1>
        <p>Professional AI Image Generation on Koyeb GPU Infrastructure</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 初始化模型
    model_result = init_model()
    if model_result[0] is None:
        st.stop()
    
    pipeline, model_name = model_result
    
    # GPU 狀態顯示
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        st.markdown(f"""
        <div class="gpu-status">
            <strong>🚀 GPU 狀態</strong><br>
            {gpu_name}<br>
            {gpu_memory}GB VRAM<br>
            模型: {model_name.split('/')[-1]}
        </div>
        """, unsafe_allow_html=True)
    
    # 側邊欄配置
    with st.sidebar:
        st.header("🎛️ 生成參數")
        
        # 基本參數
        st.subheader("基本設置")
        
        # 模型信息
        st.info(f"**當前模型**: {model_name.split('/')[-1]}")
        
        # 圖像尺寸
        col1, col2 = st.columns(2)
        with col1:
            width = st.selectbox("寬度", [512, 768, 1024, 1280], index=2)
        with col2:
            height = st.selectbox("高度", [512, 768, 1024, 1280], index=2)
        
        # 生成參數
        if "dev" in model_name.lower():
            num_steps = st.slider("推理步數", 1, 50, 28)
            guidance_scale = st.slider("引導比例", 0.1, 20.0, 7.5)
        else:
            num_steps = st.slider("推理步數", 1, 8, 4)
            guidance_scale = None
        
        # 高級設置
        with st.expander("🔧 高級設置"):
            seed = st.number_input("隨機種子", -1, 2147483647, -1)
            batch_size = st.slider("批量生成", 1, 4, 1)
            
        # 預設風格
        st.subheader("🎨 風格預設")
        style_presets = {
            "無": "",
            "攝影風格": ", professional photography, high resolution, detailed",
            "數位藝術": ", digital art, concept art, trending on artstation",
            "油畫風格": ", oil painting, classical art style, fine art",
            "科幻風格": ", sci-fi, futuristic, cyberpunk, neon lights",
            "動漫風格": ", anime style, manga, japanese art style"
        }
        
        selected_style = st.selectbox("選擇風格", list(style_presets.keys()))
        style_suffix = style_presets[selected_style]
        
        # 資源監控
        st.subheader("📊 資源監控")
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / (1024**3)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_percent = (memory_used / memory_total) * 100
            
            st.metric("GPU 內存", f"{memory_used:.1f}GB", f"{memory_percent:.1f}%")
            
            if st.button("🧹 清理 GPU 內存"):
                torch.cuda.empty_cache()
                st.success("GPU 內存已清理")
    
    # 主界面
    col_main, col_history = st.columns([2, 1])
    
    with col_main:
        st.subheader("📝 提示詞輸入")
        
        # 主提示詞
        prompt = st.text_area(
            "主提示詞:",
            placeholder="例如: A majestic dragon flying over ancient mountains during sunset",
            height=120,
            help="詳細描述您想要生成的圖像"
        )
        
        # 負面提示詞 (僅適用於 dev 模型)
        if "dev" in model_name.lower():
            negative_prompt = st.text_area(
                "負面提示詞 (可選):",
                placeholder="例如: blurry, low quality, distorted",
                height=60,
                help="描述您不希望出現在圖像中的元素"
            )
        else:
            negative_prompt = ""
        
        # 提示詞模板
        st.subheader("💡 提示詞模板")
        
        template_categories = {
            "風景攝影": [
                "A breathtaking mountain landscape at sunrise with mist rolling through valleys",
                "Serene lake reflecting autumn colors with a wooden pier extending into calm water",
                "Dense forest path with sunlight filtering through ancient trees"
            ],
            "人物肖像": [
                "Professional headshot of a confident business person in modern office setting",
                "Artistic portrait with dramatic lighting and atmospheric background",
                "Candid street photography style portrait with urban bokeh background"
            ],
            "科幻/奇幻": [
                "Futuristic cityscape with flying vehicles and neon-lit skyscrapers at night",
                "Magical forest with glowing creatures and ethereal light beams",
                "Space station orbiting a distant planet with nebula in background"
            ],
            "藝術風格": [
                "Abstract geometric composition with vibrant colors and flowing lines",
                "Minimalist design with clean lines and balanced negative space",
                "Vintage poster art with retro colors and typography elements"
            ]
        }
        
        selected_category = st.selectbox("選擇模板類別:", list(template_categories.keys()))
        selected_template = st.selectbox(
            "選擇具體模板:", 
            ["自定義"] + template_categories[selected_category]
        )
        
        if selected_template != "自定義":
            prompt = selected_template + style_suffix
        elif style_suffix:
            prompt = prompt + style_suffix
        
        # 生成按鈕區域
        col_gen1, col_gen2, col_gen3 = st.columns([2, 1, 1])
        
        with col_gen1:
            generate_btn = st.button(
                "🚀 生成圖像",
                type="primary",
                use_container_width=True,
                disabled=not prompt.strip()
            )
        
        with col_gen2:
            if st.button("🎲 隨機提示詞", use_container_width=True):
                import random
                all_templates = [t for templates in template_categories.values() for t in templates]
                prompt = random.choice(all_templates) + style_suffix
                st.rerun()
        
        with col_gen3:
            estimated_time = "30-60秒" if "dev" in model_name.lower() else "5-15秒"
            st.metric("預估時間", estimated_time)
        
        # 圖像生成
        if generate_btn and prompt.strip():
            with st.spinner(f"🎨 正在使用 {model_name.split('/')[-1]} 生成圖像..."):
                start_time = time.time()
                
                # 準備生成參數
                generator = torch.Generator().manual_seed(seed) if seed >= 0 else None
                
                params = {
                    'height': height,
                    'width': width,
                    'num_inference_steps': num_steps,
                    'generator': generator
                }
                
                if guidance_scale is not None:
                    params['guidance_scale'] = guidance_scale
                
                # 批量生成
                images = []
                for i in range(batch_size):
                    if batch_size > 1:
                        st.write(f"生成第 {i+1}/{batch_size} 張圖像...")
                    
                    image = generate_image(pipeline, prompt, negative_prompt, **params)
                    if image:
                        images.append(image)
                
                generation_time = time.time() - start_time
                
                if images:
                    st.success(f"✅ 成功生成 {len(images)} 張圖像！耗時: {generation_time:.1f}秒")
                    
                    # 顯示圖像
                    if len(images) == 1:
                        st.image(images[0], caption=prompt, use_column_width=True)
                    else:
                        # 網格顯示多張圖像
                        cols = st.columns(min(len(images), 2))
                        for i, image in enumerate(images):
                            with cols[i % 2]:
                                st.image(image, caption=f"圖像 {i+1}", use_column_width=True)
                    
                    # 保存到會話狀態
                    if 'generated_images' not in st.session_state:
                        st.session_state.generated_images = []
                    
                    for image in images:
                        st.session_state.generated_images.append({
                            'image': image,
                            'prompt': prompt,
                            'timestamp': time.strftime('%H:%M:%S'),
                            'params': {
                                'model': model_name.split('/')[-1],
                                'size': f"{width}x{height}",
                                'steps': num_steps,
                                'guidance': guidance_scale
                            }
                        })
                    
                    # 下載選項
                    if len(images) == 1:
                        img_buffer = io.BytesIO()
                        images[0].save(img_buffer, format="PNG")
                        img_buffer.seek(0)
                        
                        st.download_button(
                            "📥 下載圖像",
                            data=img_buffer,
                            file_name=f"flux_generated_{int(time.time())}.png",
                            mime="image/png"
                        )
                
                # 清理內存
                torch.cuda.empty_cache()
    
    with col_history:
        st.subheader("📚 生成歷史")
        
        if 'generated_images' in st.session_state and st.session_state.generated_images:
            # 顯示最近生成的圖像
            for i, item in enumerate(reversed(st.session_state.generated_images[-5:])):
                with st.expander(f"圖像 {len(st.session_state.generated_images)-i} - {item['timestamp']}"):
                    st.image(item['image'], use_column_width=True)
                    st.caption(item['prompt'][:100] + "..." if len(item['prompt']) > 100 else item['prompt'])
                    
                    # 參數信息
                    params = item['params']
                    st.write(f"**模型**: {params['model']}")
                    st.write(f"**尺寸**: {params['size']}")
                    st.write(f"**步數**: {params['steps']}")
                    if params['guidance']:
                        st.write(f"**引導**: {params['guidance']}")
            
            # 清空歷史按鈕
            if st.button("🗑️ 清空歷史"):
                st.session_state.generated_images = []
                st.rerun()
        else:
            st.info("還沒有生成圖像。開始創作您的第一張圖像吧！")
        
        # 使用統計
        st.subheader("📊 使用統計")
        total_generated = len(st.session_state.get('generated_images', []))
        st.metric("總生成數量", total_generated)
        
        if total_generated > 0:
            avg_time = "45秒" if "dev" in model_name.lower() else "10秒"
            st.metric("平均生成時間", avg_time)

if __name__ == "__main__":
    main()
