import joblib
import os
import pandas as pd
import numpy as np
import warnings
import gc
from statsmodels.tsa.statespace.sarimax import SARIMAX
from concurrent.futures import ProcessPoolExecutor
from src.config import Config
from tqdm import tqdm

warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self):
        self.config = Config()
        if not os.path.exists(self.config.MODEL_PATH):
            os.makedirs(self.config.MODEL_PATH, exist_ok=True)

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
            
            if len(data_values[data_values > 0]) < 1:
                return False, sku, "Data transaksi aktif < 1"

            model_fit = None
            reason = "Unknown Error"
            
            try:
                if model_type.upper() == "SARIMA":
                    model = SARIMAX(data_values, 
                                    order=(1, 1, 1), 
                                    seasonal_order=(1, 1, 0, 7) if len(data_values) > 14 else None,
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False)
                else:
                    model = SARIMAX(data_values, order=(1, 1, 1))
                
                model_fit = model.fit(disp=False, maxiter=20)
                test_check = model_fit.get_forecast(steps=1).predicted_mean
                
                if np.isnan(test_check) or test_check[0] > 10000 or test_check[0] < 0:
                    model_fit = None
                    reason = "Hasil tidak masuk akal (NaN/Outlier)"
            except Exception as e:
                model_fit = None
                reason = f"Gagal konvergensi: {str(e)}"

            if model_fit is None:
                try:
                    model_alt = SARIMAX(data_values, order=(1, 0, 0))
                    model_fit = model_alt.fit(disp=False, maxiter=15)
                except Exception as e:
                    return False, sku, f"Recovery gagal: {str(e)}"

            if model_fit:
                safe_sku = "".join([c if c.isalnum() else "_" for c in str(sku)])
                file_path = os.path.join(model_path, f"{safe_sku}.pkl")
                joblib.dump(model_fit, file_path)
                
                del model_fit
                gc.collect()
                return True, sku, "Success"
                
        except Exception as e:
            return False, sku, f"System Error: {str(e)}"
        
        return False, sku, reason

    def train_all(self, raw_df, model_type="SARIMA", callback=None):
        df = raw_df.copy()

        c_sku = next((c for c in df.columns if 'SKU' in c), None)
        c_qty = next((c for c in df.columns if 'Jumlah' in c), None)
        c_time = next((c for c in df.columns if 'Waktu' in c or 'Tanggal' in c), None)

        if not all([c_sku, c_qty, c_time]):
            return 0, [{"sku": "N/A", "reason": "Kolom tidak ditemukan"}]

        df[c_time] = pd.to_datetime(df[c_time], errors='coerce')
        df[c_sku] = df[c_sku].astype(str).str.strip()
        df = df.dropna(subset=[c_time, c_sku])
        df = df[~df[c_sku].isin(['-', '', 'nan', 'None'])]

        sku_counts = df.groupby(c_sku).size()
        active_sku_list = sku_counts[sku_counts >= 4].index.tolist()
        
        self.auto_cleanup_models(active_sku_list)

        tasks = []
        for sku in tqdm(active_sku_list, desc="Preprocessing Data", unit="sku"):
            sku_data = df[df[c_sku] == sku]
            daily_series = sku_data.set_index(c_time)[c_qty].resample('D').sum().fillna(0)
            tasks.append((sku, daily_series, model_type, self.config.MODEL_PATH))

        trained_count = 0
        failed_details = []
        
        inactive_skus = sku_counts[sku_counts < 1].index.tolist()
        for s in inactive_skus:
            failed_details.append({"sku": s, "reason": "Data terlalu sedikit (< 1 transaksi)"})

        max_workers = 2 
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = executor.map(self._train_worker, tasks)
            
            for i, result in enumerate(tqdm(futures, total=len(tasks), desc="Training Models"), 1):
                is_success, sku_name, reason = result
                if is_success:
                    trained_count += 1
                else:
                    failed_details.append({"sku": sku_name, "reason": reason})
                
                if callback:
                    callback(i, len(tasks))
                
                if i % 10 == 0:
                    gc.collect()

        return trained_count, failed_details