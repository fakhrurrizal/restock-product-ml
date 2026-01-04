import joblib
import os
import pandas as pd
import numpy as np
import warnings
import gc
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from concurrent.futures import ProcessPoolExecutor
from src.config import Config
from tqdm import tqdm

warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self):
        self.config = Config()
        if not os.path.exists(self.config.MODEL_PATH):
            os.makedirs(self.config.MODEL_PATH, exist_ok=True)

    def calculate_mape(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true > 0
        if not np.any(mask):
            return 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def auto_cleanup_models(self, active_skus):
        existing_files = os.listdir(self.config.MODEL_PATH)
        active_safe_filenames = {f"{''.join([c if c.isalnum() else '_' for c in str(sku)])}.pkl" for sku in active_skus}
        
        deleted_count = 0
        for file in existing_files:
            if file.endswith(".pkl") and file not in active_safe_filenames:
                try:
                    os.remove(os.path.join(self.config.MODEL_PATH, file))
                    deleted_count += 1
                except:
                    pass
        return deleted_count

    def _train_worker(self, args):
        sku, daily_series, model_type, model_path = args
        try:
            data_values = daily_series.values.astype(float)
            
            stats = {
                "total_qty": int(np.sum(data_values)),
                "avg_daily": float(np.mean(data_values[data_values > 0])) if np.any(data_values > 0) else 0,
                "active_days": int(len(data_values[data_values > 0])),
                "metrics": None
            }

            if stats["active_days"] < 4:
                return False, sku, "Data transaksi aktif < 4", stats

            split_idx = int(len(data_values) * 0.8)
            if split_idx < 1: split_idx = len(data_values) - 1
            
            train_data = data_values[:split_idx]
            test_data = data_values[split_idx:]
            
            model_fit = None
            reason = "Unknown Error"

            try:
                if model_type.upper() == "SARIMA" and len(train_data) > 14:
                    model = SARIMAX(train_data, 
                                    order=(1, 1, 1), 
                                    seasonal_order=(1, 1, 1, 7),
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False)
                else:
                    model = SARIMAX(train_data, order=(1, 1, 1), enforce_stationarity=False)
                
                model_fit = model.fit(disp=False, maxiter=20)
                predictions = model_fit.get_forecast(steps=len(test_data)).predicted_mean
                
                mae = mean_absolute_error(test_data, predictions)
                rmse = np.sqrt(mean_squared_error(test_data, predictions))
                mape = self.calculate_mape(test_data, predictions)

                stats["metrics"] = {
                    "mae": round(float(mae), 2),
                    "rmse": round(float(rmse), 2),
                    "mape": f"{round(float(mape), 2)}%"
                }

            except Exception as e:
                try:
                    model_alt = SimpleExpSmoothing(train_data).fit(smoothing_level=0.3, optimized=False)
                    model_fit = model_alt
                    stats["metrics"] = {"mae": 0, "rmse": 0, "mape": "N/A (Fallback)"}
                except:
                    return False, sku, f"Training gagal: {str(e)}", stats

            if model_fit:
                final_model = SARIMAX(data_values, order=(1,1,1), enforce_stationarity=False).fit(disp=False, maxiter=10)
                safe_sku = "".join([c if c.isalnum() else "_" for c in str(sku)])
                file_path = os.path.join(model_path, f"{safe_sku}.pkl")
                joblib.dump(final_model, file_path)
                
                del model_fit, final_model
                gc.collect()
                return True, sku, "Success", stats
                
        except Exception as e:
            return False, sku, f"System Error: {str(e)}", {}
        
        return False, sku, reason, stats

    def train_all(self, raw_df, model_type="SARIMA", callback=None):
        df = raw_df.copy()

        c_sku = next((c for c in df.columns if 'SKU' in c), None)
        c_qty = next((c for c in df.columns if 'Jumlah' in c), None)
        c_time = next((c for c in df.columns if 'Waktu' in c or 'Tanggal' in c), None)

        if not all([c_sku, c_qty, c_time]):
            return 0, [{"sku": "N/A", "reason": "Kolom tidak ditemukan"}], [], {}

        df[c_time] = pd.to_datetime(df[c_time], errors='coerce')
        df[c_sku] = df[c_sku].astype(str).str.strip()
        df = df.dropna(subset=[c_time, c_sku])
        df = df[~df[c_sku].isin(['-', '', 'nan', 'None'])]

        full_series = df.groupby([c_sku, pd.Grouper(key=c_time, freq='D')])[c_qty].sum().unstack(level=0).fillna(0)
        
        sku_counts = (full_series > 0).sum()
        active_sku_list = sku_counts[sku_counts >= 4].index.tolist()
        
        self.auto_cleanup_models(active_sku_list)

        tasks = []
        for sku in active_sku_list:
            tasks.append((sku, full_series[sku], model_type, self.config.MODEL_PATH))

        trained_count = 0
        trained_details = []
        failed_details = []
        
        all_mae = []
        all_rmse = []
        all_mape = []
        
        inactive_skus = sku_counts[sku_counts < 4].index.tolist()
        for s in inactive_skus:
            failed_details.append({"sku": s, "reason": "Data terlalu sedikit (< 4 transaksi)", "stats": {}})

        max_workers = 2 
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = executor.map(self._train_worker, tasks)
            
            for i, result in enumerate(tqdm(futures, total=len(tasks), desc="Training Models"), 1):
                is_success, sku_name, reason, stats = result
                if is_success:
                    trained_count += 1
                    
                    if stats.get("metrics") and isinstance(stats["metrics"]["mape"], str) and "%" in stats["metrics"]["mape"]:
                        all_mae.append(stats["metrics"]["mae"])
                        all_rmse.append(stats["metrics"]["rmse"])
                        all_mape.append(float(stats["metrics"]["mape"].replace('%', '')))
                    
                    trained_details.append({
                        "sku": sku_name,
                        "metrics": stats.get("metrics"),
                        "stats": stats
                    })
                else:
                    failed_details.append({
                        "sku": sku_name, 
                        "reason": reason,
                        "stats": stats if stats else {}
                    })
                
                if callback:
                    callback(i, len(tasks))
                
                if i % 20 == 0:
                    gc.collect()

        global_metrics = {
            "mae": round(float(np.mean(all_mae)), 2) if all_mae else 0,
            "rmse": round(float(np.mean(all_rmse)), 2) if all_rmse else 0,
            "mape": f"{round(float(np.mean(all_mape)), 2)}%" if all_mape else "0%"
        }

        return trained_count, failed_details, trained_details, global_metrics