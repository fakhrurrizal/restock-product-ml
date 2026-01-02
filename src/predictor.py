import joblib
import os
import pandas as pd
import re
import numpy as np
import json
from collections import OrderedDict
from src.config import Config
from datetime import datetime

class RestockPredictor:
    def __init__(self):
        self.config = Config()
        self.model_cache = OrderedDict()
        self.cache_limit = 500
        self.learning_file = os.path.join("data", "dynamic_patterns.json")
        self.available_models = self._refresh_model_list()
        self.keywords = self._load_all_patterns()

    def _refresh_model_list(self):
        if os.path.exists(self.config.MODEL_PATH):
            return set(os.listdir(self.config.MODEL_PATH))
        return set()

    def _load_all_patterns(self):
        base_patterns = {
            'rekomendasi_restock': [
                'restock', 'stok', 'beli', 'kulakan', 'order', 'tambah', 'kurang', 
                'habis', 'kritis', 'urgent', 'segera', 'menipis', 'rekomendasi', 
                'saran', 'prediksi stok', 'belanja', 'inventori', 'gudang'
            ],
            'top_products': [
                'terlaris', 'laku', 'best seller', 'top', 'banyak', 'terbanyak', 
                'unggulan', 'favorit', 'juara', 'populer', 'paling', 'item utama',
                'paling laku', 'penjualan tertinggi', 'produk emas', 'terjual'
            ],
            'trend_analysis': [
                'tren', 'analisa', 'analisis', 'grafik', 'perkembangan', 'historis', 
                'performa', 'evaluasi', 'laporan', 'statistik', 'riwayat', 'naik turun',
                'tracking', 'pantau', 'cek data', 'perbandingan'
            ],
            'summary': [
                'ringkasan', 'summary', 'overview', 'dashboard', 'total', 'rekap', 
                'kesimpulan', 'poin penting', 'seluruh', 'semua data'
            ]
        }

        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r') as f:
                    content = f.read()
                    if content:
                        dynamic_data = json.loads(content)
                        for category in base_patterns:
                            if category in dynamic_data:
                                combined = set(base_patterns[category] + dynamic_data[category])
                                base_patterns[category] = list(combined)
            except:
                pass
        return base_patterns

    def _learn_new_pattern(self, category, word):
        word = word.lower().strip()
        if not word or len(word) < 3: 
            return 
        
        dynamic_data = {}
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r') as f:
                    content = f.read()
                    if content:
                        dynamic_data = json.loads(content)
            except:
                dynamic_data = {}

        if category not in dynamic_data:
            dynamic_data[category] = []
        
        if word not in dynamic_data[category] and word not in self.keywords.get(category, []):
            dynamic_data[category].append(word)
            os.makedirs(os.path.dirname(self.learning_file), exist_ok=True)
            with open(self.learning_file, 'w') as f:
                json.dump(dynamic_data, f, indent=4)
            self.keywords = self._load_all_patterns()

    def get_forecast_series(self, sku_id):
        if not sku_id: return []
        
        if sku_id in self.model_cache:
            self.model_cache.move_to_end(sku_id)
            model = self.model_cache[sku_id]
        else:
            safe_sku = "".join([c if c.isalnum() else "_" for c in str(sku_id)])
            filename = f"{safe_sku}.pkl"
            if filename not in self.available_models: return []
            try:
                model = joblib.load(os.path.join(self.config.MODEL_PATH, filename))
                self.model_cache[sku_id] = model
                if len(self.model_cache) > self.cache_limit:
                    self.model_cache.popitem(last=False)
            except: return []

        try:
            forecast = model.get_forecast(steps=7)
            return [{"name": f"H+{i+1}", "value": int(round(max(0, float(val))))} 
                    for i, val in enumerate(forecast.predicted_mean)]
        except: return []

    def get_forecast(self, sku_id):
        series = self.get_forecast_series(sku_id)
        if not series: return 0
        return sum(item['value'] for item in series)

    def extract_number(self, text):
        nums = re.findall(r'\d+', text)
        val = int(nums[0]) if nums else None
        return val

    def process_natural_language(self, request_input, product_info, daily_sales_dict, raw_df):
        prompt = ""
        history = []
        
        if isinstance(request_input, dict):
            prompt = request_input.get("message", "")
            history = request_input.get("history", [])
        else:
            prompt = str(request_input)

        prompt_lower = prompt.lower().strip()
        words_in_chat = re.findall(r'\w+', prompt_lower)
        
        num_extracted = self.extract_number(prompt_lower)
        limit = min(num_extracted, 50) if num_extracted else 5

        last_data = None
        if history:
            for h in reversed(history):
                if isinstance(h, dict) and h.get('data'):
                    last_data = h['data']
                    break

        if last_data and num_extracted and any(x in prompt_lower for x in ['saja', 'tampilkan', 'ambil', 'data']):
            return {
                "type": "multi_visual", "status": "success",
                "message": f"Menampilkan {limit} data sesuai permintaan Anda:",
                "data": last_data[:limit],
                "charts": [] 
            }

        command_type = None
        for p_type, key_list in self.keywords.items():
            if any(word in words_in_chat for word in key_list):
                command_type = p_type
                break

        if not command_type:
            if any(x in prompt_lower for x in ['hitung', 'itungan', 'itungin']):
                command_type = 'trend_analysis'
                self._learn_new_pattern('trend_analysis', 'hitung')
            elif 'paling' in prompt_lower and any(x in prompt_lower for x in ['banyak', 'laku', 'terjual']):
                command_type = 'top_products'
                self._learn_new_pattern('top_products', 'paling')

        if not command_type:
            return {
                "type": "text", "status": "error",
                "message": f"Maaf, saya tidak mengenali perintah: '{prompt}'. Coba tanya 'Produk terlaris' atau 'Rekomendasi restock'."
            }

        c_sku = next((c for c in raw_df.columns if 'SKU' in c), 'Nomor Referensi SKU')
        c_qty = next((c for c in raw_df.columns if 'Jumlah' in c), 'Jumlah')
        c_nama = next((c for c in raw_df.columns if 'Nama Produk' in c), 'Nama Produk')
        c_var = next((c for c in raw_df.columns if 'Variasi' in c), 'Nama Variasi')
        c_waktu = next((c for c in raw_df.columns if 'Waktu' in c or 'Tanggal' in c), 'Waktu Pesanan Dibuat')
        c_bayar = next((c for c in raw_df.columns if 'Total' in c or 'Bayar' in c), 'Total Pembayaran')

        df_work = raw_df.copy()
        df_work[c_var] = df_work[c_var].fillna('-').replace('', '-')
        
        if not pd.api.types.is_datetime64_any_dtype(df_work[c_waktu]):
            df_work[c_waktu] = pd.to_datetime(df_work[c_waktu], errors='coerce')

        if not pd.api.types.is_numeric_dtype(df_work[c_bayar]):
            df_work[c_bayar] = pd.to_numeric(df_work[c_bayar].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce').fillna(0)

        months_id = {'januari':1, 'februari':2, 'maret':3, 'april':4, 'mei':5, 'juni':6,
                     'juli':7, 'agustus':8, 'september':9, 'oktober':10, 'november':11, 'desember':12}
        detected_months = [m for m in months_id if m in prompt_lower]
        time_label = "Keseluruhan"

        if len(detected_months) >= 2:
            m1, m2 = months_id[detected_months[0]], months_id[detected_months[1]]
            df_work = df_work[(df_work[c_waktu].dt.month >= min(m1,m2)) & (df_work[c_waktu].dt.month <= max(m1,m2))]
            time_label = f"{detected_months[0].title()} - {detected_months[1].title()}"
        elif len(detected_months) == 1:
            df_work = df_work[df_work[c_waktu].dt.month == months_id[detected_months[0]]]
            time_label = f"Bulan {detected_months[0].title()}"

        if command_type == 'top_products':
            res = df_work.groupby([c_nama, c_var, c_sku]).agg({c_qty: 'sum'}).reset_index()
            res = res.sort_values(by=c_qty, ascending=False).head(limit)
            chart_data = [{"name": str(r[c_nama]), "value": int(r[c_qty])} for _, r in res.iterrows()]
            return {
                "type": "multi_visual", "status": "success",
                "message": f"Daftar {limit} Produk Terlaris ({time_label}):",
                "data": [{"produk": str(r[c_nama]), "variasi": str(r[c_var]), "sku": str(r[c_sku]), "total": int(r[c_qty])} for _, r in res.iterrows()],
                "charts": [{"title": "Volume Penjualan Teratas", "type": "bar", "data": chart_data}]
            }

        if command_type == 'rekomendasi_restock':
            unique_prods = df_work.groupby([c_nama, c_var, c_sku])[c_qty].sum().reset_index().sort_values(by=c_qty, ascending=False).head(50)
            results = []
            all_charts = []
            for _, row in unique_prods.iterrows():
                forecast_data = self.get_forecast_series(row[c_sku])
                if forecast_data:
                    f_total = sum(d['value'] for d in forecast_data)
                    results.append({
                        "sku": str(row[c_sku]), "nama_produk": str(row[c_nama]), "variasi": str(row[c_var]),
                        "prediksi_7_hari": int(f_total),
                        "urgensi": "KRITIS" if f_total > (int(row[c_qty])/4) else "NORMAL"
                    })
                    if len(all_charts) < 2:
                        all_charts.append({
                            "title": f"Forecast: {str(row[c_nama])[:20]}",
                            "type": "line",
                            "data": forecast_data
                        })
            return {
                "type": "multi_visual", "status": "success",
                "message": "Analisis Prediksi Stok 7 Hari Ke Depan:",
                "data": sorted(results, key=lambda x: x['prediksi_7_hari'], reverse=True)[:limit],
                "charts": all_charts
            }

        if command_type == 'trend_analysis':
            daily_trend = df_work.groupby(df_work[c_waktu].dt.date)[c_qty].sum().reset_index()
            daily_trend.columns = ['date', 'qty']
            chart_data = [{"name": str(r['date']), "value": int(r['qty'])} for _, r in daily_trend.tail(15).iterrows()]
            res_table = df_work.groupby([c_nama, c_var, c_sku])[c_qty].sum().reset_index().sort_values(by=c_qty, ascending=False).head(limit)
            return {
                "type": "multi_visual", "status": "success",
                "message": f"Analisis Tren Penjualan Periode {time_label}:",
                "data": [{"produk": str(r[c_nama]), "variasi": str(r[c_var]), "total": int(r[c_qty])} for _, r in res_table.iterrows()],
                "charts": [{"title": "Grafik Penjualan 15 Hari Terakhir", "type": "line", "data": chart_data}]
            }

        if command_type == 'summary':
            t_qty = int(df_work[c_qty].sum())
            t_bayar = float(df_work[c_bayar].sum())
            t_sku = int(df_work[c_sku].nunique())
            
            summary_data = {
                "total_terjual": t_qty,
                "omzet": f"Rp {t_bayar:,.0f}",
                "produk_unik": t_sku
            }
            
            df_work['week'] = df_work[c_waktu].dt.isocalendar().week
            weekly_omzet = df_work.groupby('week')[c_bayar].sum().tail(5).reset_index()
            chart_omzet = [{"name": f"W-{int(r['week'])}", "value": float(r[c_bayar])} for _, r in weekly_omzet.iterrows()]
            
            return {
                "type": "multi_visual", "status": "success",
                "message": f"Ringkasan Performa Toko ({time_label}):",
                "summary": summary_data,
                "charts": [{"title": "Performa Omzet Mingguan", "type": "bar", "data": chart_omzet}]
            }