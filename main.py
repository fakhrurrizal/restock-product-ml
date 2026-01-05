import os
import shutil
import uvicorn
import glob
import json
import asyncio
from datetime import datetime
from typing import List, Optional, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.predictor import RestockPredictor
from src.config import Config

app = FastAPI(title="AI Stock Service")
config = Config()

origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
origins = [origin.strip() for origin in origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {
    "analysis": None,
    "daily_sales": None,
    "raw_df": None,
    "global_metrics": {},
    "trained_details": [],
    "model_info": {
        "model_type": None,
        "order": "(1, 1, 1)",
        "seasonal_order": "(1, 1, 1, 7)"
    },
    "progress": {"percent": 0, "status": "Membaca File..."}
}

predictor = RestockPredictor()

def load_existing_data():
    file_path = os.path.join(config.UPLOAD_DIR, "active_dataset.csv")
    if os.path.exists(file_path):
        try:
            df = DataLoader().load_data(file_path)
            preprocessor = DataPreprocessor()
            _, daily_sales, product_analysis = preprocessor.preprocess_data(df)
            
            if product_analysis is not None:
                state["analysis"] = product_analysis
                state["daily_sales"] = daily_sales
                state["raw_df"] = df
        except Exception as e:
            print(f"Error loading existing data: {e}")

class ChatHistoryItem(BaseModel):
    user: str
    bot: Optional[str] = None
    data: Optional[List[Any]] = None

class ChatInput(BaseModel):
    message: str
    history: Optional[List[ChatHistoryItem]] = []

def run_training_process(file_path: str, model_type: str):
    global state
    try:
        state["model_info"]["model_type"] = model_type
        state["progress"] = {"percent": 5, "status": "Membaca data..."}
        df = DataLoader().load_data(file_path)
        
        state["progress"] = {"percent": 15, "status": "Preprocessing data..."}
        preprocessor = DataPreprocessor()
        _, daily_sales, product_analysis = preprocessor.preprocess_data(df)
        
        state["analysis"] = product_analysis
        state["daily_sales"] = daily_sales
        state["raw_df"] = df

        state["progress"] = {"percent": 30, "status": f"Training {model_type} dimulai..."}
        trainer = ModelTrainer()
        
        def progress_callback(current, total):
            p = 30 + int((current / total) * 65)
            state["progress"] = {"percent": p, "status": f"Training {current}/{total} SKU..."}

        count, failed, details, global_eval = trainer.train_all(
            df, 
            model_type=model_type, 
            callback=progress_callback
        ) 
        
        state["trained_details"] = details
        state["global_metrics"] = global_eval
        
        predictor.available_models = predictor._refresh_model_list()
        state["progress"] = {"percent": 100, "status": "Selesai"}

    except Exception as e:
        state["progress"] = {"percent": 0, "status": f"Error: {str(e)}"}

@app.get("/train-progress")
async def train_progress():
    async def event_generator():
        while True:
            yield f"data: {json.dumps(state['progress'])}\n\n"
            if state["progress"]["percent"] >= 100 or "Error" in state["progress"]["status"]:
                break
            await asyncio.sleep(0.8) 
    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no" 
        })

@app.post("/upload-train")
async def upload_and_auto_train(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    model_type: str = Query("SARIMA", enum=["SARIMA", "ARIMA"])
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Hanya file CSV yang diperbolehkan")

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(config.UPLOAD_DIR, "active_dataset.csv")
    state["progress"] = {"percent": 0, "status": "Inisialisasi upload..."}
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan file: {str(e)}")
    
    background_tasks.add_task(run_training_process, file_path, model_type)
    return {"status": "Started", "message": "Proses training berjalan."}

@app.post("/chat")
async def chat_with_ai(input: ChatInput):
    if state["raw_df"] is None:
        return {
            "type": "text", 
            "status": "error",
            "message": "Model belum siap. Silakan upload dataset terlebih dahulu."
        }
    
    try:
        response = predictor.process_natural_language(
            request_input={
                "message": input.message,
                "history": [h.dict() for h in input.history]
            },
            product_info=state["analysis"], 
            daily_sales_dict=state["daily_sales"],
            raw_df=state["raw_df"]
        )
        return response
    except Exception as e:
        return {"type": "text", "status": "error", "message": f"Kesalahan internal AI: {str(e)}"}
    
@app.get("/check-status")
async def check_status():
    is_ready = state["raw_df"] is not None
    total_processed = len(state["analysis"]) if is_ready else 0
    success_count = len(state["trained_details"])
    
    return {
        "is_trained": is_ready,
        "total_sku": total_processed,
        "last_status": state["progress"]["status"],
        "model_info": {
            "selected_model": state["model_info"]["model_type"],
            "order": state["model_info"]["order"],
            "seasonal_order": state["model_info"]["seasonal_order"] if state["model_info"]["model_type"] == "SARIMA" else None
        },
        "global_evaluation": state["global_metrics"],
        "summary": {
            "success": success_count,
            "failed": max(0, total_processed - success_count)
        }
    }

@app.delete("/reset-data")
async def reset_data():
    global state
    try:
        file_path = os.path.join(config.UPLOAD_DIR, "active_dataset.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        model_files = glob.glob(os.path.join("models", "*"))
        for f in model_files:
            if os.path.isfile(f):
                os.remove(f)
        
        state.update({
            "analysis": None,
            "daily_sales": None,
            "raw_df": None,
            "global_metrics": {},
            "trained_details": [],
            "model_info": {"model_type": None, "order": "(1, 1, 1)", "seasonal_order": "(1, 1, 1, 7)"},
            "progress": {"percent": 0, "status": "Membaca File..."}
        })
        predictor.available_models = []
        return {"message": "Data berhasil dihapus"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    load_existing_data()
    uvicorn.run(app, host="0.0.0.0", port=8000)