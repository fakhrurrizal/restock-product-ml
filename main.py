import os
import shutil
import uvicorn
import multiprocessing
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.predictor import RestockPredictor
from src.config import Config

app = FastAPI(title="AI Stock Service")
config = Config()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {
    "analysis": None,
    "daily_sales": None,
    "raw_df": None
}

predictor = RestockPredictor()

def load_existing_data():
    """Memuat data otomatis ke RAM. Hanya dipanggil oleh proses utama."""
    file_path = os.path.join(config.UPLOAD_DIR, "active_dataset.csv")
    if os.path.exists(file_path):
        try:
            print(f"\n[STARTUP] Memuat dataset aktif: {file_path}")
            df = DataLoader().load_data(file_path)
            preprocessor = DataPreprocessor()
            _, daily_sales, product_analysis = preprocessor.preprocess_data(df)
            
            if product_analysis is not None:
                state["analysis"] = product_analysis
                state["daily_sales"] = daily_sales
                state["raw_df"] = df
                print("[STARTUP] Sukses memuat data lama ke RAM\n")
        except Exception as e:
            print(f"[STARTUP] Gagal memuat data lama: {str(e)}")

class ChatInput(BaseModel):
    message: str

@app.post("/upload-train")
async def upload_and_auto_train(
    file: UploadFile = File(...), 
    model_type: str = Query("SARIMA", enum=["SARIMA", "ARIMA"])
):
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(config.UPLOAD_DIR, "active_dataset.csv")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        df = DataLoader().load_data(file_path)
        preprocessor = DataPreprocessor()
        _, daily_sales, product_analysis = preprocessor.preprocess_data(df)
        
        state["analysis"] = product_analysis
        state["daily_sales"] = daily_sales
        state["raw_df"] = df
        
        trainer = ModelTrainer()
        trained_count, failed_list = trainer.train_all(df, model_type=model_type) 
        
        predictor.available_models = predictor._refresh_model_list()
        model_files = sorted(list(predictor.available_models))
        
        sku_col = next((c for c in df.columns if 'SKU' in c), "SKU")
        total_unique_sku = df[sku_col].nunique()

        return {
            "status": "Success",
            "message": f"Proses selesai menggunakan algoritma {model_type}",
            "execution_summary": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_algorithm": model_type,
                "total_sku_detected": int(total_unique_sku),
                "total_models_created": int(trained_count),
                "total_failed": int(len(failed_list)),
                "source_file": file.filename
            },
            "failed_details": failed_list,
            "output_details": [
                {
                    "sku_file": m,
                    "status": "Active/Ready",
                    "last_updated": datetime.now().strftime("%Y-%m-%d")
                } for m in model_files
            ]
        }
    except Exception as e:
        print(f"Upload & Train Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses data: {str(e)}")

@app.post("/chat")
async def chat_with_ai(input: ChatInput):
    if state["raw_df"] is None:
        return {
            "type": "text", 
            "status": "error",
            "message": "Data belum diunggah. Silakan upload file CSV terlebih dahulu."
        }
    
    try:
        response = predictor.process_natural_language(
            request_input=input.model_dump(),
            product_info=state["analysis"], 
            daily_sales_dict=state["daily_sales"],
            raw_df=state["raw_df"]
        )
        return response
    except Exception as e:
        print(f"Chat Engine Error: {str(e)}")
        return {"type": "text", "status": "error", "message": f"Kesalahan internal: {str(e)}"}

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    load_existing_data()
    
    # 3. Jalankan Uvicorn
    print("AI Stock Service is Starting")
    uvicorn.run(app, host="0.0.0.0", port=8000)