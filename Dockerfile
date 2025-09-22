# 使用輕量級 Python 鏡像
FROM python:3.11-slim

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 複製並安裝 Python 依賴
COPY requirements.txt .

# 安裝 PyTorch CPU 版本
RUN pip install --no-cache-dir \
    torch==2.8.0+cpu \
    torchvision==0.23.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 安裝其他依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY app.py .

# 設置環境變量
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8000
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 暴露端口
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/_stcore/health || exit 1

# 啟動命令
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
