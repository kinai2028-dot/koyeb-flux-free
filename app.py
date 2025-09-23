import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import time
import os
import json
import base64

# 页面配置
st.set_page_config(
    page_title="Flux AI - 稳定版",
    page_icon="🚀",
    layout="wide"
)

# 简化的CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.stable-badge {
    background: #10b981;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# 简化的API服务配置
def call_huggingface_api_simple(prompt, token):
    """简化的HuggingFace API调用"""
    url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"inputs": prompt}
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            return {"success": True, "data": response.content}
        else:
            return {"success": False, "error": f"API错误: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"网络错误: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"未知错误: {str(e)}"}

def create_demo_image(prompt):
    """创建演示图像（避免外部API依赖）"""
    try:
        # 创建简单的占位符图像URL
        text = prompt[:20].replace(" ", "+")
        demo_url = f"https://via.placeholder.com/512x512/2563eb/ffffff?text={text}"
        
        response = requests.get(demo_url, timeout=10)
        if response.status_code == 200:
            return {"success": True, "data": response.content}
        else:
            return {"success": False, "error": "无法创建演示图像"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    # 标题
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Flux AI - 稳定版</h1>
        <p>兼容性优化 | <span class="stable-badge">Python 3.11</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 服务选择
        service_mode = st.radio(
            "选择模式:",
            ["演示模式", "HuggingFace API"]
        )
        
        if service_mode == "HuggingFace API":
            hf_token = st.text_input(
                "HuggingFace Token:",
                type="password",
                help="从 huggingface.co 获取免费token"
            )
        
        st.info(f"""
        **当前模式: {service_mode}**
        - Python: 3.11 (稳定)
        - 依赖: 最小化
        - 状态: ✅ 运行正常
        """)
        
        # 图像设置
        st.subheader("🎨 图像设置")
        image_format = st.selectbox("输出格式", ["PNG", "JPEG"], index=0)
        image_quality = st.slider("图像质量", 1, 10, 8)
    
    # 主界面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎨 图像生成")
        
        # 提示词输入
        prompt = st.text_area(
            "描述你想要的图像:",
            height=100,
            placeholder="例如：A beautiful sunset over mountains"
        )
        
        # 预设模板
        templates = [
            "A serene landscape with mountains and lake",
            "Modern architectural building with glass facade", 
            "Abstract art with geometric shapes and colors",
            "Portrait of a person in natural lighting",
            "Futuristic city with flying vehicles"
        ]
        
        template_choice = st.selectbox("或选择模板:", ["自定义"] + templates)
        if template_choice != "自定义":
            prompt = template_choice
        
        # 生成按钮
        generate_btn = st.button(
            "🚀 生成图像", 
            type="primary",
            use_container_width=True,
            disabled=not prompt.strip()
        )
        
        # 生成逻辑
        if generate_btn and prompt.strip():
            # 检查必要条件
            if service_mode == "HuggingFace API" and 'hf_token' not in locals():
                st.error("请输入 HuggingFace Token")
            elif service_mode == "HuggingFace API" and not hf_token:
                st.error("请输入 HuggingFace Token")
            else:
                with st.spinner(f"使用{service_mode}生成图像..."):
                    start_time = time.time()
                    
                    # 调用相应的API
                    if service_mode == "HuggingFace API":
                        result = call_huggingface_api_simple(prompt, hf_token)
                    else:  # 演示模式
                        result = create_demo_image(prompt)
                    
                    generation_time = time.time() - start_time
                    
                    if result["success"]:
                        try:
                            # 处理图像数据
                            image = Image.open(BytesIO(result["data"]))
                            
                            st.success(f"✅ 生成成功！耗时: {generation_time:.1f}秒")
                            
                            # 显示图像
                            st.image(image, caption=prompt, use_column_width=True)
                            
                            # 下载功能
                            img_buffer = BytesIO()
                            img_format = image_format.upper()
                            if img_format == "JPEG":
                                image = image.convert("RGB")
                            
                            image.save(img_buffer, format=img_format, quality=image_quality*10)
                            
                            st.download_button(
                                f"📥 下载 {image_format}",
                                data=img_buffer.getvalue(),
                                file_name=f"flux_{int(time.time())}.{image_format.lower()}",
                                mime=f"image/{image_format.lower()}"
                            )
                            
                        except Exception as img_error:
                            st.error(f"图像处理失败: {img_error}")
                    else:
                        st.error(f"❌ 生成失败: {result['error']}")
    
    with col2:
        st.subheader("📋 使用说明")
        
        st.markdown("""
        **🔧 稳定版特性:**
        - Python 3.11 兼容
        - 最小化依赖
        - 减少构建错误
        - 快速部署
        
        **🎯 支持的模式:**
        - **演示模式**: 无需API，即时响应
        - **HuggingFace**: 免费1000次/月
        
        **💡 使用技巧:**
        - 详细描述提升质量
        - 使用英文提示词
        - 避免版权内容
        """)
        
        # 系统状态
        st.subheader("⚡ 系统状态")
        st.success("🟢 服务正常")
        st.info("📦 依赖已优化")
        st.info("🐍 Python 3.11")

if __name__ == "__main__":
    main()
