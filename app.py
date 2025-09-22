import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import time
import os
import json
import base64
import psutil
from typing import Dict, Any, Optional

# 页面配置
st.set_page_config(
    page_title="Flux AI on Koyeb CPU",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Koyeb 优化的 CSS
st.markdown("""
<style>
.koyeb-header {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #1e40af 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(37, 99, 235, 0.3);
}

.resource-card {
    background: #f8fafc;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #2563eb;
    margin: 1rem 0;
}

.api-status {
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: bold;
    margin: 0.25rem;
    display: inline-block;
}

.status-success { background: #dcfce7; color: #166534; }
.status-warning { background: #fef3c7; color: #92400e; }
.status-error { background: #fee2e2; color: #991b1b; }

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# API 服务配置
API_SERVICES = {
    "Black Forest Labs": {
        "endpoint": "https://api.bfl.ml/v1/flux-pro-1.1",
        "model": "flux-pro-1.1",
        "free_quota": "$1 免费额度",
        "avg_time": "15-30秒",
        "quality": "最高品质",
        "cost_per_image": "$0.05"
    },
    "Hugging Face": {
        "endpoint": "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
        "model": "FLUX.1-schnell",
        "free_quota": "1000次/月",
        "avg_time": "10-20秒", 
        "quality": "高品质",
        "cost_per_image": "免费"
    },
    "Replicate": {
        "model": "black-forest-labs/flux-schnell",
        "free_quota": "$1 免费额度",
        "avg_time": "20-40秒",
        "quality": "高品质",
        "cost_per_image": "$0.003"
    },
    "Demo Mode": {
        "endpoint": "placeholder",
        "free_quota": "无限制",
        "avg_time": "即时",
        "quality": "演示品质",
        "cost_per_image": "$0.00"
    }
}

def get_system_metrics() -> Dict[str, Any]:
    """获取系统资源使用情况"""
    try:
        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024**2)
        memory_total_mb = memory.total / (1024**2)
        memory_percent = memory.percent
        
        # 磁盘信息
        disk = psutil.disk_usage('/')
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        disk_percent = (disk.used / disk.total) * 100
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count
            },
            "memory": {
                "used_mb": memory_used_mb,
                "total_mb": memory_total_mb,
                "percent": memory_percent
            },
            "disk": {
                "used_gb": disk_used_gb,
                "total_gb": disk_total_gb,
                "percent": disk_percent
            }
        }
    except Exception as e:
        return {"error": str(e)}

def call_huggingface_api(prompt: str, token: str) -> Dict[str, Any]:
    """调用 Hugging Face Inference API"""
    url = API_SERVICES["Hugging Face"]["endpoint"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "guidance_scale": 0.0,
            "num_inference_steps": 4,
            "max_sequence_length": 512
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.content,
                "service": "Hugging Face",
                "model": "FLUX.1-schnell"
            }
        elif response.status_code == 503:
            return {
                "success": False, 
                "error": "模型正在加载中，请稍后重试",
                "retry_after": 30
            }
        else:
            return {
                "success": False,
                "error": f"API 错误 {response.status_code}: {response.text}"
            }
    except requests.exceptions.Timeout:
        return {"success": False, "error": "请求超时，请重试"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def call_replicate_api(prompt: str, token: str) -> Dict[str, Any]:
    """调用 Replicate API"""
    try:
        import replicate
        
        # 设置 API token
        os.environ["REPLICATE_API_TOKEN"] = token
        
        output = replicate.run(
            API_SERVICES["Replicate"]["model"],
            input={
                "prompt": prompt,
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 90
            }
        )
        
        # 下载图像
        if isinstance(output, list) and output:
            image_url = output[0]
        else:
            image_url = output
            
        response = requests.get(image_url, timeout=60)
        
        return {
            "success": True,
            "data": response.content,
            "service": "Replicate",
            "model": "flux-schnell"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def call_bfl_api(prompt: str, token: str) -> Dict[str, Any]:
    """调用 Black Forest Labs 官方 API"""
    url = "https://api.bfl.ml/v1/flux-pro-1.1"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
        "prompt_upsampling": False,
        "seed": None,
        "safety_tolerance": 2
    }
    
    try:
        # 提交任务
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            task_data = response.json()
            task_id = task_data["id"]
            
            # 轮询结果
            result_url = f"https://api.bfl.ml/v1/get_result?id={task_id}"
            
            max_attempts = 60  # 最多等待5分钟
            for attempt in range(max_attempts):
                time.sleep(5)  # 每5秒检查一次
                
                result_response = requests.get(result_url, headers=headers, timeout=30)
                
                if result_response.status_code == 200:
                    result_data = result_response.json()
                    
                    if result_data["status"] == "Ready":
                        # 下载图像
                        image_url = result_data["result"]["sample"]
                        image_response = requests.get(image_url, timeout=60)
                        
                        return {
                            "success": True,
                            "data": image_response.content,
                            "service": "Black Forest Labs",
                            "model": "flux-pro-1.1"
                        }
                    elif result_data["status"] == "Error":
                        return {
                            "success": False,
                            "error": f"生成失败: {result_data.get('error', '未知错误')}"
                        }
            
            return {"success": False, "error": "生成超时"}
        else:
            return {"success": False, "error": f"API 错误: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def create_demo_image(prompt: str) -> Dict[str, Any]:
    """创建演示图像"""
    try:
        # 创建带文字的占位符图像
        text = prompt[:30].replace(" ", "+")
        demo_url = f"https://via.placeholder.com/512x512/2563eb/ffffff?text=Demo:+{text}"
        
        response = requests.get(demo_url, timeout=15)
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.content,
                "service": "Demo Mode",
                "model": "placeholder"
            }
        else:
            return {"success": False, "error": "无法创建演示图像"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    # 主标题
    st.markdown("""
    <div class="koyeb-header">
        <h1>🚀 Flux AI on Koyeb CPU</h1>
        <p>高性能 CPU 实例 | 自动缩放 | Scale-to-Zero</p>
        <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.9;">
            支持多种 API 服务 | 免费额度优化 | 全球部署
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 获取系统资源信息
    metrics = get_system_metrics()
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ Koyeb 配置")
        
        # 显示系统资源
        if "error" not in metrics:
            st.markdown(f"""
            <div class="resource-card">
                <h4>📊 实例资源</h4>
                <div class="metrics-grid">
                    <div><strong>CPU:</strong> {metrics['cpu']['percent']:.1f}%</div>
                    <div><strong>内存:</strong> {metrics['memory']['percent']:.1f}%</div>
                    <div><strong>磁盘:</strong> {metrics['disk']['percent']:.1f}%</div>
                </div>
                <div style="font-size: 0.8rem; color: #6b7280; margin-top: 0.5rem;">
                    RAM: {metrics['memory']['used_mb']:.0f}MB / {metrics['memory']['total_mb']:.0f}MB<br>
                    存储: {metrics['disk']['used_gb']:.1f}GB / {metrics['disk']['total_gb']:.1f}GB
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # API 服务选择
        st.subheader("🔌 API 服务")
        
        selected_service = st.selectbox(
            "选择生成服务:",
            list(API_SERVICES.keys()),
            help="不同服务有不同的成本和质量特点"
        )
        
        # 显示服务信息
        service_info = API_SERVICES[selected_service]
        
        # 状态指示器
        status_class = "status-success" if selected_service == "Demo Mode" else "status-warning"
        
        st.markdown(f"""
        <div class="resource-card">
            <h4>{selected_service}</h4>
            <div class="api-status {status_class}">
                {service_info['free_quota']}
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                <strong>响应时间:</strong> {service_info['avg_time']}<br>
                <strong>图像质量:</strong> {service_info['quality']}<br>
                <strong>单张成本:</strong> {service_info['cost_per_image']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # API Token 输入
        if selected_service != "Demo Mode":
            api_token = st.text_input(
                f"{selected_service} API Token:",
                type="password",
                help="从官方网站获取免费 API Token",
                placeholder="输入您的 API Token..."
            )
        else:
            api_token = None
        
        st.divider()
        
        # 生成设置
        st.subheader("🎨 生成设置")
        
        # 图像参数
        col1, col2 = st.columns(2)
        with col1:
            image_width = st.selectbox("宽度", [512, 768, 1024], index=2)
        with col2:
            image_height = st.selectbox("高度", [512, 768, 1024], index=2)
        
        image_quality = st.select_slider(
            "质量级别:",
            ["快速", "标准", "高质量"],
            value="标准"
        )
        
        # 高级选项
        with st.expander("🔧 高级选项"):
            enable_upscaling = st.checkbox("启用提示词优化", value=True)
            safety_tolerance = st.slider("安全级别", 1, 5, 2)
            retry_failed = st.checkbox("失败自动重试", value=True)
        
        st.divider()
        
        # Koyeb 优势说明
        st.subheader("🚀 Koyeb 优势")
        st.markdown("""
        **Scale-to-Zero:**
        - 闲置时自动缩减到零
        - 请求时快速启动 (200ms)
        - 大幅降低运行成本
        
        **全球部署:**
        - 50+ 个地区可选
        - 自动 CDN 加速
        - 就近用户访问
        
        **开发友好:**
        - Git 驱动部署
        - 自动 HTTPS/SSL
        - 内建负载均衡
        """)
    
    # 主界面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎨 AI 图像生成")
        
        # 提示词输入区域
        prompt = st.text_area(
            "输入提示词 (支持中英文):",
            height=120,
            placeholder="例如：A majestic dragon flying over ancient mountains during sunset, highly detailed, fantasy art style",
            help="详细的描述能获得更好的生成效果"
        )
        
        # 快速提示词模板
        st.subheader("💡 快速模板")
        
        template_categories = {
            "自然风景": [
                "A serene mountain landscape with crystal clear lake reflecting the sky",
                "Dense ancient forest with sunlight filtering through tall trees", 
                "Spectacular sunset over rolling hills with wildflowers"
            ],
            "艺术创作": [
                "Abstract geometric composition with vibrant colors and flowing lines",
                "Minimalist design with clean shapes and negative space",
                "Surreal digital art with impossible architecture and floating elements"
            ],
            "科幻未来": [
                "Futuristic cityscape with flying vehicles and neon-lit skyscrapers",
                "Advanced space station orbiting a distant planet with nebula background",
                "Cyberpunk street scene with holographic advertisements and rain"
            ],
            "人物肖像": [
                "Professional headshot with soft natural lighting and neutral background",
                "Artistic portrait with dramatic lighting and creative composition",
                "Candid street photography style with urban background bokeh"
            ]
        }
        
        selected_category = st.selectbox("选择类别:", list(template_categories.keys()))
        selected_template = st.selectbox(
            "选择具体模板:",
            ["自定义"] + template_categories[selected_category]
        )
        
        if selected_template != "自定义":
            prompt = selected_template
        
        # 提示词优化建议
        if prompt and enable_upscaling:
            quality_keywords = ", highly detailed, professional quality, 8k resolution"
            if quality_keywords not in prompt:
                optimized_prompt = prompt + quality_keywords
                with st.expander("📈 优化后的提示词"):
                    st.code(optimized_prompt)
                    if st.button("使用优化版本"):
                        prompt = optimized_prompt
                        st.rerun()
        
        # 生成控制面板
        col_gen1, col_gen2, col_gen3 = st.columns([2, 1, 1])
        
        with col_gen1:
            generate_btn = st.button(
                f"🚀 使用 {selected_service} 生成",
                type="primary",
                use_container_width=True,
                disabled=not prompt.strip()
            )
        
        with col_gen2:
            if st.button("🎲 随机", use_container_width=True):
                import random
                all_templates = [t for templates in template_categories.values() for t in templates]
                prompt = random.choice(all_templates)
                st.rerun()
        
        with col_gen3:
            est_cost = API_SERVICES[selected_service]['cost_per_image']
            st.metric("预估成本", est_cost)
        
        # 图像生成主逻辑
        if generate_btn and prompt.strip():
            # 验证 API Token
            if selected_service != "Demo Mode" and not api_token:
                st.error(f"请输入 {selected_service} 的 API Token")
                st.info("💡 您可以先使用演示模式测试应用功能")
            else:
                with st.spinner(f"🎨 使用 {selected_service} 生成图像中..."):
                    # 显示进度信息
                    progress_placeholder = st.empty()
                    progress_placeholder.info(f"⏳ 预计等待时间: {service_info['avg_time']}")
                    
                    start_time = time.time()
                    
                    # 调用相应的 API
                    try:
                        if selected_service == "Hugging Face":
                            result = call_huggingface_api(prompt, api_token)
                        elif selected_service == "Replicate":
                            result = call_replicate_api(prompt, api_token)
                        elif selected_service == "Black Forest Labs":
                            result = call_bfl_api(prompt, api_token)
                        else:  # Demo Mode
                            result = create_demo_image(prompt)
                        
                        generation_time = time.time() - start_time
                        progress_placeholder.empty()
                        
                        if result["success"]:
                            st.success(f"✅ 生成成功！耗时: {generation_time:.1f}秒")
                            
                            # 处理图像数据
                            try:
                                image = Image.open(BytesIO(result["data"]))
                                
                                # 显示图像
                                st.image(
                                    image,
                                    caption=f"🎨 {prompt} | 服务: {result['service']} | 模型: {result['model']}",
                                    use_column_width=True
                                )
                                
                                # 图像信息
                                col_info1, col_info2, col_info3 = st.columns(3)
                                with col_info1:
                                    st.metric("图像尺寸", f"{image.width}×{image.height}")
                                with col_info2:
                                    st.metric("文件格式", image.format or "PNG")
                                with col_info3:
                                    file_size = len(result["data"]) / 1024
                                    st.metric("文件大小", f"{file_size:.1f}KB")
                                
                                # 下载选项
                                col_dl1, col_dl2 = st.columns(2)
                                
                                with col_dl1:
                                    # PNG 下载
                                    png_buffer = BytesIO()
                                    image.save(png_buffer, format="PNG", optimize=True)
                                    st.download_button(
                                        "📥 下载 PNG",
                                        data=png_buffer.getvalue(),
                                        file_name=f"flux_{int(time.time())}.png",
                                        mime="image/png",
                                        use_container_width=True
                                    )
                                
                                with col_dl2:
                                    # JPEG 下载 (更小文件)
                                    jpeg_buffer = BytesIO()
                                    rgb_image = image.convert("RGB")
                                    rgb_image.save(jpeg_buffer, format="JPEG", quality=90, optimize=True)
                                    st.download_button(
                                        "📥 下载 JPEG",
                                        data=jpeg_buffer.getvalue(),
                                        file_name=f"flux_{int(time.time())}.jpg",
                                        mime="image/jpeg",
                                        use_container_width=True
                                    )
                                
                                # 保存到会话历史
                                if 'generation_history' not in st.session_state:
                                    st.session_state.generation_history = []
                                
                                st.session_state.generation_history.append({
                                    'prompt': prompt,
                                    'service': result['service'],
                                    'model': result['model'],
                                    'timestamp': time.strftime('%H:%M:%S'),
                                    'generation_time': f"{generation_time:.1f}s",
                                    'cost': API_SERVICES[selected_service]['cost_per_image']
                                })
                                
                                # 限制历史记录数量
                                if len(st.session_state.generation_history) > 10:
                                    st.session_state.generation_history.pop(0)
                            
                            except Exception as img_error:
                                st.error(f"❌ 图像处理失败: {img_error}")
                        
                        else:
                            st.error(f"❌ 生成失败: {result['error']}")
                            
                            # 自动重试逻辑
                            if retry_failed and "retry_after" in result:
                                st.info(f"🔄 将在 {result['retry_after']} 秒后自动重试...")
                                time.sleep(result['retry_after'])
                                st.rerun()
                            
                            # 错误解决建议
                            st.info("""
                            **可能的解决方案:**
                            - 检查 API Token 是否正确且有效
                            - 尝试切换到其他 API 服务
                            - 简化提示词内容
                            - 使用演示模式测试功能
                            """)
                    
                    except Exception as e:
                        progress_placeholder.empty()
                        st.error(f"❌ 请求处理异常: {str(e)}")
    
    with col2:
        st.subheader("📊 实时状态")
        
        # 当前系统状态
        if "error" not in metrics:
            st.markdown(f"""
            **🖥️ CPU 使用率**
            ```
            {metrics['cpu']['percent']:.1f}% ({metrics['cpu']['count']} 核心)
            ```
            
            **💾 内存使用**
            ```
            {metrics['memory']['used_mb']:.0f}MB / {metrics['memory']['total_mb']:.0f}MB
            ({metrics['memory']['percent']:.1f}%)
            ```
            
            **💿 磁盘使用**
            ```
            {metrics['disk']['used_gb']:.1f}GB / {metrics['disk']['total_gb']:.1f}GB
            ({metrics['disk']['percent']:.1f}%)
            ```
            """)
        
        # API 服务状态
        st.subheader("🌐 API 服务状态")
        for service_name, info in API_SERVICES.items():
            if service_name == selected_service:
                status_indicator = "🟢 当前使用"
            elif service_name == "Demo Mode":
                status_indicator = "🟢 始终可用"
            else:
                status_indicator = "🟡 需要 Token"
            
            st.write(f"**{service_name}**: {status_indicator}")
        
        # 生成历史
        if 'generation_history' in st.session_state and st.session_state.generation_history:
            st.subheader("📚 生成历史")
            
            for i, record in enumerate(reversed(st.session_state.generation_history[-5:])):
                with st.expander(f"记录 {i+1} - {record['timestamp']}"):
                    st.write(f"**提示词**: {record['prompt'][:50]}...")
                    st.write(f"**服务**: {record['service']}")
                    st.write(f"**耗时**: {record['generation_time']}")
                    st.write(f"**成本**: {record['cost']}")
            
            if st.button("🗑️ 清空历史"):
                st.session_state.generation_history = []
                st.rerun()
        
        # 使用统计
        st.subheader("📈 使用统计")
        total_generations = len(st.session_state.get('generation_history', []))
        st.metric("总生成次数", total_generations)
        
        if total_generations > 0:
            avg_time = sum(float(r['generation_time'].replace('s', '')) 
                          for r in st.session_state.generation_history) / total_generations
            st.metric("平均耗时", f"{avg_time:.1f}s")
        
        # 部署信息
        st.subheader("🚀 部署信息")
        st.info(f"""
        **平台**: Koyeb CPU 实例
        **区域**: 自动选择最优
        **缩放**: Scale-to-Zero 已启用
        **SSL**: 自动配置
        **状态**: ✅ 运行正常
        """)

if __name__ == "__main__":
    main()
