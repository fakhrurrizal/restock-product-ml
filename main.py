import os
import shutil
import uvicorn
import glob
import json
import asyncio
import gc
from datetime import datetime
from typing import List, Optional, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from multiprocessing import Manager

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
    "progress": {"percent": 0, "status": "Ready"}
}

mp_manager = None
training_active_flag = None
is_training_active = False
predictor = RestockPredictor()

async def internal_cleanup():
    global state
    try:
        file_path = os.path.join(config.UPLOAD_DIR, "active_dataset.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        model_files = glob.glob(os.path.join(config.MODEL_PATH, "*.pkl"))
        for f in model_files:
            try:
                os.remove(f)
            except:
                pass
        
        state.update({
            "analysis": None,
            "daily_sales": None,
            "raw_df": None,
            "global_metrics": {},
            "trained_details": [],
            "model_info": {"model_type": None, "order": "(1, 1, 1)", "seasonal_order": "(1, 1, 1, 7)"},
            "progress": {"percent": 0, "status": "Ready"}
        })
        predictor.available_models = []
        gc.collect()
        return True
    except Exception as e:
        return False

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
        except:
            pass

class ChatHistoryItem(BaseModel):
    user: str
    bot: Optional[str] = None
    data: Optional[List[Any]] = None

class ChatInput(BaseModel):
    message: str
    history: Optional[List[ChatHistoryItem]] = []

async def run_training_process(file_path: str, model_type: str):
    global state, is_training_active, training_active_flag
    is_training_active = True
    
    if training_active_flag is not None:
        training_active_flag.value = True
    
    try:
        state["model_info"]["model_type"] = model_type
        state["progress"] = {"percent": 1, "status": "Inisialisasi..."}
        
        df = await asyncio.to_thread(DataLoader().load_data, file_path)
        
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

        count, failed, details, global_eval = await asyncio.to_thread(
            trainer.train_all, df, model_type, progress_callback, training_active_flag
        ) 
        
        if training_active_flag and not training_active_flag.value:
            raise InterruptedError("STOP_REQUESTED")

        state["trained_details"] = details
        state["global_metrics"] = global_eval
        predictor.available_models = predictor._refresh_model_list()
        state["progress"] = {"percent": 100, "status": "Selesai"}

    except InterruptedError:
        state["progress"] = {"percent": 0, "status": "Training dihentikan"}
    except Exception as e:
        state["progress"] = {"percent": 0, "status": f"Error: {str(e)}"}
    finally:
        is_training_active = False

@app.get("/train-progress")
async def train_progress():
    async def event_generator():
        while True:
            current_progress = state["progress"]
            yield f"data: {json.dumps(current_progress)}\n\n"
            
            status_lower = current_progress["status"].lower()
            if current_progress["percent"] >= 100 or "error" in status_lower:
                break
            if "dihentikan" in status_lower:
                await asyncio.sleep(1)
                break
            await asyncio.sleep(0.5) 
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/check-status")
async def check_status():
    available_models = predictor._refresh_model_list()
    is_trained = (state["raw_df"] is not None and state["analysis"] is not None and len(available_models) > 0)
    total_processed = len(state["analysis"]) if state["analysis"] is not None else 0
    total_rows = len(state["raw_df"]) if state["raw_df"] is not None else 0
    success_count = len(state["trained_details"])
    global_metrics = state["global_metrics"] if state["global_metrics"] else {"mae": None, "rmse": None, "mape": "N/A"}

    return {
        "is_trained": is_trained,
        "is_active": is_training_active,
        "total_sku": total_processed,
        "total_rows": total_rows,
        "last_status": state["progress"]["status"],
        "model_info": {
            "selected_model": state["model_info"]["model_type"],
            "order": state["model_info"]["order"],
            "seasonal_order": state["model_info"]["seasonal_order"] if state["model_info"]["model_type"] == "SARIMA" else None
        },
        "global_evaluation": global_metrics,
        "summary": {
            "success": success_count,
            "failed": max(0, total_processed - success_count)
        }
    }

@app.post("/stop-training")
async def stop_training():
    global is_training_active, training_active_flag
    if is_training_active and training_active_flag is not None:
        training_active_flag.value = False
        state["progress"] = {"percent": 0, "status": "Training dihentikan"}
        await internal_cleanup()
        is_training_active = False
        return {"status": "stopped"}
    return {"status": "idle"}

@app.post("/upload-train")
async def upload_and_auto_train(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    model_type: str = Query("SARIMA", enum=["SARIMA", "ARIMA"])
):
    global is_training_active
    if is_training_active:
        raise HTTPException(status_code=400, detail="Training sedang berjalan.")
    file_path = os.path.join(config.UPLOAD_DIR, "active_dataset.csv")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    background_tasks.add_task(run_training_process, file_path, model_type)
    return {"status": "Started"}

@app.post("/chat")
async def chat_with_ai(input: ChatInput):
    if state["raw_df"] is None:
        return {"type": "text", "status": "error", "message": "Data belum siap."}
    response = await asyncio.to_thread(
        predictor.process_natural_language,
        {"message": input.message, "history": [h.dict() for h in input.history]},
        state["analysis"], state["daily_sales"], state["raw_df"]
    )
    return response

@app.delete("/reset-data")
async def reset_data():
    global is_training_active, training_active_flag
    if is_training_active and training_active_flag is not None:
        training_active_flag.value = False
        await asyncio.sleep(0.5)
        is_training_active = False
    await internal_cleanup()
    return {"message": "Data dibersihkan"}

if __name__ == "__main__":
    mp_manager = Manager()
    training_active_flag = mp_manager.Value('b', True)
    load_existing_data()
    uvicorn.run(app, host="0.0.0.0", port=8000)