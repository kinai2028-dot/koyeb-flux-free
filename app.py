import os
import base64
import io
import torch
from flask import Flask, request, jsonify
from diffusers import FluxPipeline
from PIL import Image

app = Flask(__name__)

# 全域變數儲存模型
pipeline = None

def load_model():
    """載入 FLUX.1 Krea [dev] 模型"""
    global pipeline
    
    if pipeline is None:
        print("正在載入 FLUX.1 Krea [dev] 模型...")
        
        # 檢查 HuggingFace Token
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("需要設置 HF_TOKEN 環境變數來訪問模型")
        
        try:
            pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Krea-dev",
                torch_dtype=torch.bfloat16,
                use_auth_token=hf_token
            )
            
            # 啟用 CPU 卸載以節省 VRAM
            pipeline.enable_model_cpu_offload()
            
            print("模型載入成功！")
            
        except Exception as e:
            print(f"模型載入失敗: {str(e)}")
            raise e
    
    return pipeline

def image_to_base64(image):
    """將 PIL 圖像轉換為 base64 字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return f"data:image/png;base64,{img_str.decode()}"

@app.route('/health', methods=['GET'])
def health_check():
    """健康檢查端點"""
    return jsonify({"status": "healthy", "model": "FLUX.1 Krea [dev]"})

@app.route('/predict', methods=['POST'])
def generate_image():
    """圖像生成端點"""
    try:
        # 解析請求數據
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "缺少 prompt 參數"}), 400
        
        prompt = data['prompt']
        height = data.get('height', 1024)
        width = data.get('width', 1024)
        guidance_scale = data.get('guidance_scale', 4.5)
        num_inference_steps = data.get('num_inference_steps', 50)
        
        # 載入模型
        pipe = load_model()
        
        print(f"正在生成圖像: {prompt}")
        
        # 生成圖像
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images[0]
        
        # 轉換為 base64
        image_b64 = image_to_base64(image)
        
        return jsonify({
            "images": [image_b64],
            "prompt": prompt,
            "model": "FLUX.1 Krea [dev]",
            "parameters": {
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps
            }
        })
        
    except Exception as e:
        print(f"生成錯誤: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """根端點"""
    return jsonify({
        "message": "FLUX.1 Krea [dev] API 已就緒",
        "model": "FLUX.1 Krea [dev]",
        "endpoints": {
            "generate": "/predict",
            "health": "/health"
        }
    })

if __name__ == '__main__':
    # 預先載入模型（可選）
    try:
        load_model()
    except Exception as e:
        print(f"預載模型失敗，將在第一次請求時載入: {e}")
    
    # 啟動 Flask 應用
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
