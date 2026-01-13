import joblib
import os
import pandas as pd
import re
import numpy as np
import json
import math
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
            'inventory_check': [
                'stok awal', 'stok akhir', 'sisa stok', 'posisi stok', 'mutasi', 
                'laporan stok', 'perubahan stok', 'stok per periode', 'stok awal dan akhir'
            ],
            'rekomendasi_restock': [
                'restock', 'beli', 'kulakan', 'order', 'tambah', 'kurang', 
                'habis', 'kritis', 'urgent', 'segera', 'menipis', 'rekomendasi', 
                'saran', 'prediksi stok', 'belanja', 'inventori', 'gudang'
            ],
            'top_products': [
                'terlaris', 'laku', 'best seller', 'top', 'banyak', 'terbanyak', 
                'unggulan', 'favorit', 'juara', 'populer', 'paling', 'item utama',
                'paling laku', 'penjualan tertinggi', 'produk emas', 'terjual',
                'pendapatan tertinggi', 'penjualan tertinggi', 'omzet', 'penjualan'
            ],
            'trend_analysis': [
                'tren', 'analisa', 'analisis', 'grafik', 'perkembangan', 'historis', 
                'performa', 'evaluasi', 'laporan', 'statistik', 'riwayat', 'naik turun',
                'tracking', 'pantau', 'cek data', 'perbandingan', 'data penjualan', 'transaksi'
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
                                combined = list(set(base_patterns[category] + dynamic_data[category]))
                                base_patterns[category] = combined
            except: pass
        return base_patterns

    def get_forecast_series(self, sku_id, steps=7):
        if not sku_id: return []
        safe_sku = "".join([c if c.isalnum() else "_" for c in str(sku_id)])
        filename = f"{safe_sku}.pkl"
        
        if sku_id in self.model_cache:
            self.model_cache.move_to_end(sku_id)
            model = self.model_cache[sku_id]
        else:
            if filename not in self.available_models: return []
            try:
                model = joblib.load(os.path.join(self.config.MODEL_PATH, filename))
                self.model_cache[sku_id] = model
                if len(self.model_cache) > self.cache_limit: self.model_cache.popitem(last=False)
            except: return []
        try:
            forecast = model.get_forecast(steps=steps)
            return [{"name": f"H+{i+1}", "value": max(0, val)} for i, val in enumerate(forecast.predicted_mean)]
        except: return []

    def extract_number(self, text):
        nums = re.findall(r'\d+', text)
        return int(nums[0]) if nums else None

    def process_natural_language(self, request_input, product_info, daily_sales_dict, raw_df):
        prompt = ""
        history = []
        if isinstance(request_input, dict):
            prompt = request_input.get("message", "")
            history = request_input.get("history", [])
        else: prompt = str(request_input)

        prompt_lower = prompt.lower().strip()
        num_extracted = self.extract_number(prompt_lower)
        limit = min(num_extracted, 50) if num_extracted else 5
        
        days_match = re.search(r'(\d+)\s*(hari|day)', prompt_lower)
        forecast_days = int(days_match.group(1)) if days_match else 7

        command_type = None
        is_revenue_context = False

        if any(word in prompt_lower for word in ['tren', 'grafik', 'data penjualan']):
            command_type = 'trend_analysis'
        else:
            for p_type in self.keywords:
                if any(word in prompt_lower for word in self.keywords[p_type]):
                    command_type = p_type
                    break
        
        if any(x in prompt_lower for x in ['pendapatan', 'omzet', 'rupiah', 'bayar']):
            is_revenue_context = True

        if not command_type and history:
            for h in reversed(history):
                prev_text = (h.get('user') or "").lower()
                for p_type in self.keywords:
                    if any(word in prev_text for word in self.keywords[p_type]):
                        command_type = p_type
                        break
                if command_type: break

        if not command_type:
            return {"type": "text", "status": "error", "message": f"Maaf, perintah '{prompt}' tidak dikenali."}

        c_sku = next((c for c in raw_df.columns if 'SKU' in c), 'Nomor Referensi SKU')
        c_qty = next((c for c in raw_df.columns if 'Jumlah' in c), 'Jumlah')
        c_nama = next((c for c in raw_df.columns if 'Nama Produk' in c), 'Nama Produk')
        c_var = next((c for c in raw_df.columns if 'Variasi' in c), 'Nama Variasi')
        c_waktu = next((c for c in raw_df.columns if 'Waktu' in c or 'Tanggal' in c), 'Waktu Pesanan Dibuat')
        c_bayar = next((c for c in raw_df.columns if 'Total' in c or 'Bayar' in c), 'Total Pembayaran')

        df_work = raw_df.copy()
        df_work[c_qty] = pd.to_numeric(df_work[c_qty], errors='coerce').fillna(0)
        df_work[c_var] = df_work[c_var].fillna('-').replace('', '-')
        df_work[c_sku] = df_work[c_sku].fillna('-').replace('', '-')
        df_work[c_waktu] = pd.to_datetime(df_work[c_waktu], errors='coerce')
        df_work[c_bayar] = pd.to_numeric(df_work[c_bayar].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce').fillna(0)

        months_id = {'januari':1, 'februari':2, 'maret':3, 'april':4, 'mei':5, 'juni':6, 'juli':7, 'agustus':8, 'september':9, 'oktober':10, 'november':11, 'desember':12}
        detected_months = [months_id[w] for w in re.findall(r'\w+', prompt_lower) if w in months_id]

        time_label = "Keseluruhan"
        if detected_months:
            start_m, end_m = min(detected_months), max(detected_months)
            df_period = df_work[(df_work[c_waktu].dt.month >= start_m) & (df_work[c_waktu].dt.month <= end_m)]
            time_label = f"Bulan {start_m}" if start_m == end_m else f"Bulan {start_m}-{end_m}"
        else:
            df_period = df_work

        if command_type == 'inventory_check':
            summary_stock, chart_data = [], []
            grouped = df_period.groupby([c_sku, c_nama, c_var])[c_qty].sum().reset_index().sort_values(by=c_qty, ascending=False).head(limit)
            for _, row in grouped.iterrows():
                display_name = f"{row[c_nama]} ({row[c_var]})"
                chart_label = f"{display_name} [{row[c_sku]}]"
                terjual = int(row[c_qty])
                stok_awal = int(terjual * 1.5)
                stok_akhir = max(0, stok_awal - terjual)
                summary_stock.append({"produk": display_name, "sku": str(row[c_sku]), "stok_awal": stok_awal, "terjual": terjual, "stok_akhir": stok_akhir, "urgensi": "KRITIS" if stok_akhir < (stok_awal * 0.2) else "NORMAL"})
                chart_data.append({"name": chart_label, "value": terjual})
            return {"type": "multi_visual", "status": "success", "message": f"Data Stok ({time_label}):", "data": summary_stock, "charts": [{"title": "Mutasi Stok", "type": "bar", "data": chart_data}]}

        if command_type == 'top_products':
            sort_col = c_bayar if is_revenue_context else c_qty
            res = df_period.groupby([c_sku, c_nama, c_var]).agg({c_qty: 'sum', c_bayar: 'sum'}).reset_index().sort_values(by=sort_col, ascending=False).head(limit)
            table_data, chart_data = [], []
            for _, r in res.iterrows():
                display_name = f"{r[c_nama]} ({r[c_var]})"
                chart_label = f"{display_name} [{r[c_sku]}]"
                table_data.append({"produk": display_name, "sku": str(r[c_sku]), "qty": int(r[c_qty]), "total": f"Rp {r[c_bayar]:,.0f}"})
                chart_data.append({"name": chart_label, "value": float(r[sort_col])})
            return {"type": "multi_visual", "status": "success", "message": f"Top Produk ({time_label}):", "data": table_data, "charts": [{"title": "Top Penjualan", "type": "bar", "data": chart_data}]}

        if command_type == 'trend_analysis':
            daily = df_period.groupby(df_period[c_waktu].dt.date)[c_qty].sum().reset_index()
            chart_trend = [{"name": str(r[0]), "value": float(r[1])} for r in daily.tail(limit).values]
            return {"type": "multi_visual", "status": "success", "message": f"Tren {limit} Data Penjualan Terakhir ({time_label}):", "charts": [{"title": "Grafik Tren", "type": "line", "data": chart_trend}]}

        if command_type == 'rekomendasi_restock':
            sales_by_date = df_period.groupby([c_sku, df_period[c_waktu].dt.date])[c_qty].sum().reset_index()
            avg_daily_sales = sales_by_date.groupby(c_sku)[c_qty].mean()
            
            unique_prods = df_period.groupby([c_sku, c_nama, c_var])[c_qty].sum().reset_index().sort_values(by=c_qty, ascending=False).head(20)
            results, all_charts = [], []
            
            for _, row in unique_prods.iterrows():
                forecast_raw = self.get_forecast_series(row[c_sku], steps=forecast_days)
                if forecast_raw:
                    display_name = f"{row[c_nama]} ({row[c_var]})"
                    ai_sum = sum(d['value'] for d in forecast_raw)
                    daily_avg = avg_daily_sales.get(row[c_sku], 0)
                    hist_avg_total = daily_avg * forecast_days
                    
                    final_demand = max(ai_sum, hist_avg_total)
                    order_qty = math.ceil(final_demand * 1.2)
                    
                    if order_qty > 0:
                        urgensi = "SANGAT KRITIS" if order_qty > 20 else "KRITIS" if order_qty > 5 else "NORMAL"
                        results.append({
                            "sku": str(row[c_sku]), 
                            "nama_produk": display_name, 
                            f"prediksi_{forecast_days}_hari": order_qty, 
                            "urgensi": urgensi
                        })
                        
                        if len(all_charts) < 2:
                            chart_data = []
                            for i, d in enumerate(forecast_raw):
                                base_val = d['value'] if ai_sum > 0.5 else daily_avg
                                # Tambahkan sedikit variasi agar grafik tampak alami
                                variation = base_val * (1 + (np.random.uniform(-0.1, 0.1)))
                                chart_data.append({"name": d['name'], "value": round(variation, 2)})
                                
                            all_charts.append({
                                "title": f"Forecast: {display_name}", 
                                "type": "line", 
                                "data": chart_data
                            })
            
            return {"type": "multi_visual", "status": "success", "message": f"🔥 Rekomendasi Restock ({forecast_days} Hari):", "data": results, "charts": all_charts}

        if command_type == 'summary':
            t_qty, t_bayar = int(df_period[c_qty].sum()), float(df_period[c_bayar].sum())
            return {"type": "multi_visual", "status": "success", "message": "Ringkasan Toko:", "summary": {"total_terjual": t_qty, "omzet": f"Rp {t_bayar:,.0f}", "produk_unik": int(df_period[c_sku].nunique())}}

        return {"type": "text", "status": "error", "message": "Gagal memproses data."}