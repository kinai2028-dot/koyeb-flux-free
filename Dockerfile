FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# 安裝 Python 和基本工具
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 複製應用文件
COPY requirements.txt .
COPY *.py .

# 安裝 Python 依賴
RUN pip3 install --no-cache-dir -r requirements.txt

# 設置環境變量
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8000
ENV HF_HOME=/app/cache

# 暴露端口
EXPOSE 8000

# 啟動命令（根據需要修改）
CMD ["python3", "-c", "import sys; exec('streamlit run app.py --server.port=8000 --server.address=0.0.0.0' if 'streamlit' in sys.modules or any('streamlit' in line for line in open('requirements.txt')) else 'python gradio_app.py')"]
