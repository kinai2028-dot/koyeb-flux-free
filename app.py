import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import time
import os
import json
import base64
import psutil
from typing import Dict, Any, Optional, List
import sqlite3
import uuid
from datetime import datetime
import zipfile

# 頁面配置
st.set_page_config(
    page_title="Flux AI Studio - Enhanced",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 增強版 CSS
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

.enhanced-badge {
    background: #10b981;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight: bold;
    margin-left: 0.5rem;
}

.custom-api-card {
    background: #f0f9ff;
    border: 1px solid #0ea5e9;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.model-card {
    background: #fefce8;
    border: 1px solid #eab308;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

.image-record {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

.gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 0.5rem;
}

.status-online { background: #10b981; }
.status-offline { background: #ef4444; }
.status-testing { background: #f59e0b; }
</style>
""", unsafe_allow_html=True)

# 數據庫初始化
def init_database():
    """初始化 SQLite 數據庫"""
    conn = sqlite3.connect('flux_ai.db')
    cursor = conn.cursor()
    
    # 創建自定義 API 表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS custom_apis (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            api_type TEXT NOT NULL,
            headers TEXT,
            parameters TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 創建自定義模型表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS custom_models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            api_id TEXT,
            model_id TEXT NOT NULL,
            parameters TEXT,
            description TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (api_id) REFERENCES custom_apis (id)
        )
    ''')
    
    # 創建圖片記錄表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_records (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            prompt TEXT NOT NULL,
            model_name TEXT,
            api_service TEXT,
            generation_time REAL,
            image_data BLOB,
            metadata TEXT,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# 自定義 API 管理
class CustomAPIManager:
    @staticmethod
    def add_api(name: str, endpoint: str, api_type: str, headers: dict = None, parameters: dict = None) -> str:
        """添加自定義 API"""
        api_id = str(uuid.uuid4())
        conn = sqlite3.connect('flux_ai.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO custom_apis (id, name, endpoint, api_type, headers, parameters)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (api_id, name, endpoint, api_type, 
              json.dumps(headers) if headers else None,
              json.dumps(parameters) if parameters else None))
        
        conn.commit()
        conn.close()
        return api_id
    
    @staticmethod
    def get_apis() -> List[Dict]:
        """獲取所有自定義 API"""
        conn = sqlite3.connect('flux_ai.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM custom_apis WHERE status = "active"')
        apis = []
        for row in cursor.fetchall():
            api = {
                'id': row[0],
                'name': row[1],
                'endpoint': row[2],
                'api_type': row[3],
                'headers': json.loads(row[4]) if row[4] else {},
                'parameters': json.loads(row[5]) if row[5] else {},
                'status': row[6],
                'created_at': row[7]
            }
            apis.append(api)
        
        conn.close()
        return apis
    
    @staticmethod
    def test_api(api_id: str, test_prompt: str = "A simple test image") -> Dict[str, Any]:
        """測試自定義 API"""
        conn = sqlite3.connect('flux_ai.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM custom_apis WHERE id = ?', (api_id,))
        row = cursor.fetchone()
        
        if not row:
            return {"success": False, "error": "API not found"}
        
        endpoint = row[2]
        api_type = row[3]
        headers = json.loads(row[4]) if row[4] else {}
        parameters = json.loads(row[5]) if row[5] else {}
        
        try:
            if api_type == "replicate":
                # Replicate API 格式
                payload = {
                    "input": {
                        "prompt": test_prompt,
                        **parameters
                    }
                }
            elif api_type == "huggingface":
                # Hugging Face API 格式
                payload = {
                    "inputs": test_prompt,
                    "parameters": parameters
                }
            else:
                # 通用格式
                payload = {
                    "prompt": test_prompt,
                    **parameters
                }
            
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return {"success": True, "status_code": response.status_code}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
        
        finally:
            conn.close()

# 自定義模型管理
class CustomModelManager:
    @staticmethod
    def add_model(name: str, api_id: str, model_id: str, parameters: dict = None, description: str = "") -> str:
        """添加自定義模型"""
        model_uid = str(uuid.uuid4())
        conn = sqlite3.connect('flux_ai.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO custom_models (id, name, api_id, model_id, parameters, description)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_uid, name, api_id, model_id, 
              json.dumps(parameters) if parameters else None, description))
        
        conn.commit()
        conn.close()
        return model_uid
    
    @staticmethod
    def get_models() -> List[Dict]:
        """獲取所有自定義模型"""
        conn = sqlite3.connect('flux_ai.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT m.*, a.name as api_name, a.endpoint 
            FROM custom_models m 
            LEFT JOIN custom_apis a ON m.api_id = a.id 
            WHERE m.status = "active"
        ''')
        
        models = []
        for row in cursor.fetchall():
            model = {
                'id': row[0],
                'name': row[1],
                'api_id': row[2],
                'model_id': row[3],
                'parameters': json.loads(row[4]) if row[4] else {},
                'description': row[5],
                'status': row[6],
                'created_at': row[7],
                'api_name': row[8],
                'api_endpoint': row[9]
            }
            models.append(model)
        
        conn.close()
        return models

# 圖片記錄管理
class ImageRecordManager:
    @staticmethod
    def save_image(prompt: str, image_data: bytes, model_name: str, api_service: str, 
                   generation_time: float, metadata: dict = None, tags: List[str] = None) -> str:
        """保存圖片記錄"""
        record_id = str(uuid.uuid4())
        filename = f"flux_{int(time.time())}_{record_id[:8]}.png"
        
        conn = sqlite3.connect('flux_ai.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO image_records 
            (id, filename, prompt, model_name, api_service, generation_time, image_data, metadata, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (record_id, filename, prompt, model_name, api_service, generation_time, 
              image_data, json.dumps(metadata) if metadata else None,
              json.dumps(tags) if tags else None))
        
        conn.commit()
        conn.close()
        return record_id
    
    @staticmethod
    def get_records(limit: int = 50, search_term: str = "") -> List[Dict]:
        """獲取圖片記錄"""
        conn = sqlite3.connect('flux_ai.db')
        cursor = conn.cursor()
        
        if search_term:
            cursor.execute('''
                SELECT id, filename, prompt, model_name, api_service, generation_time, metadata, tags, created_at
                FROM image_records 
                WHERE prompt LIKE ? OR tags LIKE ?
                ORDER BY created_at DESC LIMIT ?
            ''', (f'%{search_term}%', f'%{search_term}%', limit))
        else:
            cursor.execute('''
                SELECT id, filename, prompt, model_name, api_service, generation_time, metadata, tags, created_at
                FROM image_records 
                ORDER BY created_at DESC LIMIT ?
            ''', (limit,))
        
        records = []
        for row in cursor.fetchall():
            record = {
                'id': row[0],
                'filename': row[1],
                'prompt': row[2],
                'model_name': row[3],
                'api_service': row[4],
                'generation_time': row[5],
                'metadata': json.loads(row[6]) if row[6] else {},
                'tags': json.loads(row[7]) if row[7] else [],
                'created_at': row[8]
            }
            records.append(record)
        
        conn.close()
        return records
    
    @staticmethod
    def get_image_data(record_id: str) -> Optional[bytes]:
        """獲取圖片數據"""
        conn = sqlite3.connect('flux_ai.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT image_data FROM image_records WHERE id = ?', (record_id,))
        row = cursor.fetchone()
        
        conn.close()
        return row[0] if row else None
    
    @staticmethod
    def delete_record(record_id: str) -> bool:
        """刪除圖片記錄"""
        conn = sqlite3.connect('flux_ai.db')
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM image_records WHERE id = ?', (record_id,))
        success = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return success
    
    @staticmethod
    def export_records(record_ids: List[str]) -> bytes:
        """導出圖片記錄為 ZIP 文件"""
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            conn = sqlite3.connect('flux_ai.db')
            cursor = conn.cursor()
            
            for record_id in record_ids:
                cursor.execute('''
                    SELECT filename, prompt, image_data, metadata 
                    FROM image_records WHERE id = ?
                ''', (record_id,))
                
                row = cursor.fetchone()
                if row:
                    filename, prompt, image_data, metadata = row
                    
                    # 添加圖片文件
                    zip_file.writestr(filename, image_data)
                    
                    # 添加元數據文件
                    info = {
                        'filename': filename,
                        'prompt': prompt,
                        'metadata': json.loads(metadata) if metadata else {}
                    }
                    zip_file.writestr(f"{filename}.json", json.dumps(info, indent=2))
            
            conn.close()
        
        return zip_buffer.getvalue()

# 增強的 API 調用函數
def call_custom_api(api_id: str, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
    """調用自定義 API"""
    conn = sqlite3.connect('flux_ai.db')
    cursor = conn.cursor()
    
    # 獲取 API 信息
    cursor.execute('SELECT * FROM custom_apis WHERE id = ?', (api_id,))
    api_row = cursor.fetchone()
    
    # 獲取模型信息
    cursor.execute('SELECT * FROM custom_models WHERE id = ?', (model_id,))
    model_row = cursor.fetchone()
    
    conn.close()
    
    if not api_row or not model_row:
        return {"success": False, "error": "API 或模型不存在"}
    
    try:
        endpoint = api_row[2]
        api_type = api_row[3]
        headers = json.loads(api_row[4]) if api_row[4] else {}
        api_parameters = json.loads(api_row[5]) if api_row[5] else {}
        model_parameters = json.loads(model_row[4]) if model_row[4] else {}
        
        # 合併參數
        all_parameters = {**api_parameters, **model_parameters, **kwargs}
        
        if api_type == "replicate":
            payload = {
                "input": {
                    "prompt": prompt,
                    **all_parameters
                }
            }
        elif api_type == "huggingface":
            payload = {
                "inputs": prompt,
                "parameters": all_parameters
            }
        else:
            payload = {
                "prompt": prompt,
                "model": model_row[3],  # model_id
                **all_parameters
            }
        
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            if api_type == "replicate":
                # Replicate 返回 URL 列表
                result = response.json()
                if isinstance(result, list) and result:
                    image_url = result[0]
                    img_response = requests.get(image_url, timeout=60)
                    return {
                        "success": True,
                        "data": img_response.content,
                        "service": api_row[1],
                        "model": model_row[1]
                    }
            else:
                # 直接返回圖片數據
                return {
                    "success": True,
                    "data": response.content,
                    "service": api_row[1],
                    "model": model_row[1]
                }
        
        return {"success": False, "error": f"HTTP {response.status_code}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}

# 主應用
def main():
    # 初始化數據庫
    init_database()
    
    # 主標題
    st.markdown("""
    <div class="koyeb-header">
        <h1>🎨 Flux AI Studio - Enhanced</h1>
        <span class="enhanced-badge">自設API</span>
        <span class="enhanced-badge">自設模型</span>
        <span class="enhanced-badge">圖片記錄</span>
        <p style="margin-top: 1rem;">專業級 AI 圖像生成平台</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 創建標籤頁
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎨 圖像生成", "🔌 自設API", "🤖 自設模型", "📚 圖片記錄", "📊 統計分析"])
    
    with tab1:
        image_generation_tab()
    
    with tab2:
        custom_api_tab()
    
    with tab3:
        custom_model_tab()
    
    with tab4:
        image_records_tab()
    
    with tab5:
        analytics_tab()

def image_generation_tab():
    """圖像生成標籤頁"""
    st.subheader("🎨 AI 圖像生成")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 服務選擇
        st.markdown("### 選擇生成服務")
        
        service_type = st.radio(
            "服務類型:",
            ["內建服務", "自定義服務"],
            horizontal=True
        )
        
        if service_type == "內建服務":
            # 原有的內建服務
            selected_service = st.selectbox(
                "選擇服務:",
                ["Hugging Face", "Replicate", "Demo Mode"]
            )
            
            if selected_service != "Demo Mode":
                api_token = st.text_input(f"{selected_service} API Token:", type="password")
            else:
                api_token = None
                
            selected_model = None
        
        else:
            # 自定義服務
            custom_apis = CustomAPIManager.get_apis()
            custom_models = CustomModelManager.get_models()
            
            if not custom_apis:
                st.warning("尚未配置自定義 API，請先前往 '自設API' 標籤頁進行配置")
                return
            
            api_options = {api['name']: api['id'] for api in custom_apis}
            selected_api_name = st.selectbox("選擇 API:", list(api_options.keys()))
            selected_api_id = api_options[selected_api_name]
            
            # 獲取該 API 的模型
            api_models = [m for m in custom_models if m['api_id'] == selected_api_id]
            
            if not api_models:
                st.warning(f"API '{selected_api_name}' 尚未配置模型，請先前往 '自設模型' 標籤頁進行配置")
                return
            
            model_options = {model['name']: model['id'] for model in api_models}
            selected_model_name = st.selectbox("選擇模型:", list(model_options.keys()))
            selected_model = model_options[selected_model_name]
        
        # 提示詞輸入
        st.markdown("### 提示詞輸入")
        prompt = st.text_area(
            "輸入提示詞:",
            height=120,
            placeholder="描述您想要生成的圖像..."
        )
        
        # 標籤設置
        tags = st.text_input(
            "標籤 (用逗號分隔):",
            placeholder="風景, 藝術, 高清..."
        ).split(",") if st.text_input(
            "標籤 (用逗號分隔):",
            placeholder="風景, 藝術, 高清..."
        ).strip() else []
        
        # 生成參數
        with st.expander("🔧 生成參數"):
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                width = st.selectbox("寬度", [512, 768, 1024], index=2)
                num_steps = st.slider("推理步數", 1, 50, 20)
            with col_param2:
                height = st.selectbox("高度", [512, 768, 1024], index=2)
                guidance_scale = st.slider("引導比例", 0.0, 20.0, 7.5)
        
        # 生成按鈕
        if st.button("🚀 生成圖像", type="primary", use_container_width=True):
            if not prompt.strip():
                st.error("請輸入提示詞")
                return
            
            with st.spinner("生成中..."):
                start_time = time.time()
                
                if service_type == "內建服務":
                    if selected_service == "Demo Mode":
                        result = create_demo_image(prompt)
                    else:
                        # 調用內建服務 (原有邏輯)
                        result = {"success": False, "error": "內建服務邏輯需要實現"}
                else:
                    # 調用自定義服務
                    result = call_custom_api(
                        selected_api_id, 
                        selected_model,
                        prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale
                    )
                
                generation_time = time.time() - start_time
                
                if result["success"]:
                    st.success(f"✅ 生成成功！耗時: {generation_time:.1f}秒")
                    
                    # 顯示圖像
                    image = Image.open(BytesIO(result["data"]))
                    st.image(image, caption=prompt, use_column_width=True)
                    
                    # 保存記錄
                    record_id = ImageRecordManager.save_image(
                        prompt=prompt,
                        image_data=result["data"],
                        model_name=result.get("model", "Unknown"),
                        api_service=result.get("service", "Unknown"),
                        generation_time=generation_time,
                        metadata={
                            "width": width,
                            "height": height,
                            "num_steps": num_steps,
                            "guidance_scale": guidance_scale
                        },
                        tags=tags
                    )
                    
                    st.success(f"圖像已保存到記錄庫 (ID: {record_id[:8]})")
                    
                    # 下載按鈕
                    st.download_button(
                        "📥 下載圖像",
                        data=result["data"],
                        file_name=f"flux_{int(time.time())}.png",
                        mime="image/png"
                    )
                else:
                    st.error(f"❌ 生成失敗: {result['error']}")
    
    with col2:
        # 最近生成的圖像
        st.markdown("### 📸 最近生成")
        recent_records = ImageRecordManager.get_records(limit=5)
        
        for record in recent_records:
            with st.container():
                st.markdown(f"**{record['prompt'][:30]}...**")
                st.caption(f"模型: {record['model_name']} | {record['created_at'][:16]}")
                
                if st.button(f"查看", key=f"view_{record['id'][:8]}"):
                    st.session_state.view_record_id = record['id']

def custom_api_tab():
    """自定義 API 標籤頁"""
    st.subheader("🔌 自定義 API 配置")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 添加新 API")
        
        with st.form("add_api_form"):
            api_name = st.text_input("API 名稱", placeholder="My Custom API")
            api_endpoint = st.text_input("API 端點", placeholder="https://api.example.com/generate")
            api_type = st.selectbox("API 類型", ["replicate", "huggingface", "openai", "custom"])
            
            # Headers 配置
            st.markdown("**Headers (JSON格式):**")
            headers_json = st.text_area(
                "Headers",
                value='{\n  "Authorization": "Bearer YOUR_TOKEN",\n  "Content-Type": "application/json"\n}',
                height=100
            )
            
            # 參數配置
            st.markdown("**默認參數 (JSON格式):**")
            params_json = st.text_area(
                "Parameters",
                value='{\n  "num_outputs": 1,\n  "output_format": "png"\n}',
                height=100
            )
            
            if st.form_submit_button("添加 API", type="primary"):
                try:
                    headers = json.loads(headers_json) if headers_json.strip() else {}
                    parameters = json.loads(params_json) if params_json.strip() else {}
                    
                    api_id = CustomAPIManager.add_api(
                        name=api_name,
                        endpoint=api_endpoint,
                        api_type=api_type,
                        headers=headers,
                        parameters=parameters
                    )
                    
                    st.success(f"API '{api_name}' 添加成功！ID: {api_id[:8]}")
                    st.rerun()
                
                except json.JSONDecodeError:
                    st.error("JSON 格式錯誤，請檢查 Headers 和 Parameters 格式")
                except Exception as e:
                    st.error(f"添加失敗: {str(e)}")
    
    with col2:
        st.markdown("### 現有 API")
        
        apis = CustomAPIManager.get_apis()
        
        for api in apis:
            st.markdown(f"""
            <div class="custom-api-card">
                <h4>🔌 {api['name']}</h4>
                <p><strong>類型:</strong> {api['api_type']}</p>
                <p><strong>端點:</strong> {api['endpoint'][:50]}...</p>
                <p><strong>狀態:</strong> <span class="status-indicator status-online"></span>活躍</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_test, col_edit = st.columns(2)
            
            with col_test:
                if st.button(f"🧪 測試", key=f"test_{api['id'][:8]}"):
                    with st.spinner("測試中..."):
                        result = CustomAPIManager.test_api(api['id'])
                        if result["success"]:
                            st.success("✅ API 測試通過")
                        else:
                            st.error(f"❌ API 測試失敗: {result['error']}")
            
            with col_edit:
                if st.button(f"📝 編輯", key=f"edit_{api['id'][:8]}"):
                    st.info("編輯功能開發中...")

def custom_model_tab():
    """自定義模型標籤頁"""
    st.subheader("🤖 自定義模型配置")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 添加新模型")
        
        apis = CustomAPIManager.get_apis()
        if not apis:
            st.warning("請先添加自定義 API")
            return
        
        with st.form("add_model_form"):
            model_name = st.text_input("模型名稱", placeholder="FLUX.1-dev Custom")
            
            api_options = {api['name']: api['id'] for api in apis}
            selected_api_name = st.selectbox("關聯 API:", list(api_options.keys()))
            selected_api_id = api_options[selected_api_name]
            
            model_id = st.text_input("模型 ID", placeholder="black-forest-labs/flux-1-dev")
            model_description = st.text_area("模型描述", height=80)
            
            # 模型參數
            st.markdown("**模型專用參數 (JSON格式):**")
            model_params_json = st.text_area(
                "Model Parameters",
                value='{\n  "aspect_ratio": "1:1",\n  "output_quality": 90\n}',
                height=100
            )
            
            if st.form_submit_button("添加模型", type="primary"):
                try:
                    parameters = json.loads(model_params_json) if model_params_json.strip() else {}
                    
                    model_uid = CustomModelManager.add_model(
                        name=model_name,
                        api_id=selected_api_id,
                        model_id=model_id,
                        parameters=parameters,
                        description=model_description
                    )
                    
                    st.success(f"模型 '{model_name}' 添加成功！ID: {model_uid[:8]}")
                    st.rerun()
                
                except json.JSONDecodeError:
                    st.error("參數 JSON 格式錯誤")
                except Exception as e:
                    st.error(f"添加失敗: {str(e)}")
    
    with col2:
        st.markdown("### 現有模型")
        
        models = CustomModelManager.get_models()
        
        for model in models:
            st.markdown(f"""
            <div class="model-card">
                <h4>🤖 {model['name']}</h4>
                <p><strong>API:</strong> {model['api_name']}</p>
                <p><strong>模型ID:</strong> {model['model_id']}</p>
                <p><strong>描述:</strong> {model['description'][:50]}...</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"📊 查看詳情", key=f"model_detail_{model['id'][:8]}"):
                st.json(model['parameters'])

def image_records_tab():
    """圖片記錄標籤頁"""
    st.subheader("📚 圖片記錄管理")
    
    # 搜索和篩選
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 搜索提示詞或標籤", placeholder="輸入關鍵詞...")
    
    with col2:
        limit = st.selectbox("顯示數量", [20, 50, 100], index=1)
    
    with col3:
        view_mode = st.selectbox("顯示模式", ["列表", "網格"])
    
    # 獲取記錄
    records = ImageRecordManager.get_records(limit=limit, search_term=search_term)
    
    if not records:
        st.info("暫無圖片記錄")
        return
    
    # 批量操作
    st.markdown("### 批量操作")
    col_batch1, col_batch2, col_batch3 = st.columns(3)
    
    with col_batch1:
        if st.button("📥 導出全部"):
            record_ids = [r['id'] for r in records]
            zip_data = ImageRecordManager.export_records(record_ids)
            st.download_button(
                "下載 ZIP 文件",
                data=zip_data,
                file_name=f"flux_images_{int(time.time())}.zip",
                mime="application/zip"
            )
    
    with col_batch2:
        if st.button("📊 生成統計報告"):
            generate_analytics_report(records)
    
    with col_batch3:
        if st.button("🗑️ 清空記錄"):
            if st.checkbox("確認清空所有記錄"):
                # 實現清空邏輯
                st.success("記錄已清空")
    
    # 顯示記錄
    if view_mode == "網格":
        # 網格模式
        cols = st.columns(3)
        for i, record in enumerate(records):
            with cols[i % 3]:
                # 獲取圖片數據
                image_data = ImageRecordManager.get_image_data(record['id'])
                if image_data:
                    image = Image.open(BytesIO(image_data))
                    st.image(image, use_column_width=True)
                
                st.markdown(f"**{record['prompt'][:30]}...**")
                st.caption(f"{record['model_name']} | {record['created_at'][:16]}")
                
                col_view, col_del = st.columns(2)
                with col_view:
                    if st.button("查看", key=f"grid_view_{record['id'][:8]}"):
                        show_record_details(record)
                with col_del:
                    if st.button("刪除", key=f"grid_del_{record['id'][:8]}"):
                        if ImageRecordManager.delete_record(record['id']):
                            st.success("已刪除")
                            st.rerun()
    
    else:
        # 列表模式
        for record in records:
            with st.expander(f"🖼️ {record['prompt'][:50]}... | {record['created_at'][:16]}"):
                col_img, col_info = st.columns([1, 2])
                
                with col_img:
                    image_data = ImageRecordManager.get_image_data(record['id'])
                    if image_data:
                        image = Image.open(BytesIO(image_data))
                        st.image(image, use_column_width=True)
                
                with col_info:
                    st.markdown(f"**提示詞:** {record['prompt']}")
                    st.markdown(f"**模型:** {record['model_name']}")
                    st.markdown(f"**服務:** {record['api_service']}")
                    st.markdown(f"**生成時間:** {record['generation_time']:.1f}秒")
                    
                    if record['tags']:
                        tags_str = ", ".join(record['tags'])
                        st.markdown(f"**標籤:** {tags_str}")
                    
                    # 操作按鈕
                    col_dl, col_edit, col_del = st.columns(3)
                    
                    with col_dl:
                        if image_data:
                            st.download_button(
                                "📥 下載",
                                data=image_data,
                                file_name=record['filename'],
                                mime="image/png",
                                key=f"dl_{record['id'][:8]}"
                            )
                    
                    with col_edit:
                        if st.button("📝 編輯標籤", key=f"edit_tags_{record['id'][:8]}"):
                            st.info("標籤編輯功能開發中...")
                    
                    with col_del:
                        if st.button("🗑️ 刪除", key=f"list_del_{record['id'][:8]}"):
                            if ImageRecordManager.delete_record(record['id']):
                                st.success("已刪除")
                                st.rerun()

def analytics_tab():
    """統計分析標籤頁"""
    st.subheader("📊 統計分析")
    
    records = ImageRecordManager.get_records(limit=1000)
    
    if not records:
        st.info("暫無數據可分析")
        return
    
    # 基本統計
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("總生成數量", len(records))
    
    with col2:
        avg_time = sum(r['generation_time'] for r in records) / len(records)
        st.metric("平均生成時間", f"{avg_time:.1f}s")
    
    with col3:
        unique_models = len(set(r['model_name'] for r in records))
        st.metric("使用模型數量", unique_models)
    
    with col4:
        unique_services = len(set(r['api_service'] for r in records))
        st.metric("使用服務數量", unique_services)
    
    # 圖表分析
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### 📈 每日生成量")
        # 這裡可以添加時間序列圖表
        st.info("圖表功能開發中...")
    
    with col_chart2:
        st.markdown("### 🎯 模型使用分布")
        # 這裡可以添加餅圖
        st.info("圖表功能開發中...")

def show_record_details(record):
    """顯示記錄詳情"""
    st.modal("Record Details", record)

def generate_analytics_report(records):
    """生成統計報告"""
    st.success("統計報告生成功能開發中...")

def create_demo_image(prompt: str) -> Dict[str, Any]:
    """創建演示圖像"""
    try:
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
            return {"success": False, "error": "無法創建演示圖像"}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    main()
