# 使用稳定的 Python 3.11 版本
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip 和构建工具
RUN pip install --upgrade pip setuptools wheel

# 复制依赖文件
COPY requirements.txt .

# 分步安装依赖，避免构建错误
RUN pip install --no-cache-dir streamlit==1.28.1
RUN pip install --no-cache-dir requests==2.31.0
RUN pip install --no-cache-dir pillow==10.0.1
RUN pip install --no-cache-dir psutil==5.9.0

# 安装可选依赖
RUN pip install --no-cache-dir replicate==0.25.0 || echo "Replicate installation failed, skipping..."

# 复制应用代码
COPY app.py .

# 设置环境变量
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8000
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/_stcore/health || exit 1

# 启动命令
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
