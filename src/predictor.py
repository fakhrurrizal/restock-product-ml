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
        # RAM Management: Gunakan OrderedDict sebagai LRU Cache
        self.model_cache = OrderedDict()
        self.cache_limit = 500  # Maksimal 500 model yang disimpan di RAM
        self.learning_file = os.path.join("data", "dynamic_patterns.json")
        self.available_models = self._refresh_model_list()
        self.keywords = self._load_all_patterns()

    def _refresh_model_list(self):
        """Memperbarui daftar file model .pkl yang tersedia di storage"""
        if os.path.exists(self.config.MODEL_PATH):
            return set(os.listdir(self.config.MODEL_PATH))
        return set()

    def _load_all_patterns(self):
        """Memuat pola kata kunci dari base dan hasil pembelajaran dinamis"""
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
        """Menambahkan kata kunci baru ke database pembelajaran dinamis"""
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

    def get_forecast(self, sku_id):
        """Mengambil prediksi 7 hari ke depan dengan sistem Cache RAM"""
        if not sku_id or str(sku_id).lower() in ['nan', '-', 'none']: 
            return 0
        
        # Penanganan Cache
        if sku_id in self.model_cache:
            self.model_cache.move_to_end(sku_id) # Geser ke paling baru digunakan
            model = self.model_cache[sku_id]
        else:
            safe_sku = "".join([c if c.isalnum() else "_" for c in str(sku_id)])
            filename = f"{safe_sku}.pkl"
            
            if filename not in self.available_models: 
                return 0
                
            try:
                model = joblib.load(os.path.join(self.config.MODEL_PATH, filename))
                self.model_cache[sku_id] = model
                
                # Hapus cache yang paling jarang digunakan jika overlimit
                if len(self.model_cache) > self.cache_limit:
                    self.model_cache.popitem(last=False)
            except: 
                return 0
            
        try:
            forecast = model.get_forecast(steps=7)
            res = np.sum(forecast.predicted_mean)
            return int(round(max(0, res)))
        except: 
            return 0

    def process_natural_language(self, request_input, product_info, daily_sales_dict, raw_df):
        """Engine NLP untuk memproses chat dan memberikan output sesuai tipe data"""
        if isinstance(request_input, dict):
            prompt = request_input.get("message", "")
        else:
            prompt = str(request_input)

        prompt_lower = prompt.lower().strip()
        words_in_chat = re.findall(r'\w+', prompt_lower)
        limit = self.extract_number(prompt_lower)

        # Identifikasi Perintah
        command_type = None
        for p_type, key_list in self.keywords.items():
            if any(word in words_in_chat for word in key_list):
                command_type = p_type
                break

        # Pembelajaran Dinamis jika kata kunci tidak baku ditemukan
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
                "message": f"Maaf, saya tidak mengenali perintah: '{prompt}'. Coba tanya 'Produk terlaris' atau 'Restock'."
            }

        # Deteksi Kolom Dinamis
        c_sku = next((c for c in raw_df.columns if 'SKU' in c), 'Nomor Referensi SKU')
        c_qty = next((c for c in raw_df.columns if 'Jumlah' in c), 'Jumlah')
        c_nama = next((c for c in raw_df.columns if 'Nama Produk' in c), 'Nama Produk')
        c_var = next((c for c in raw_df.columns if 'Variasi' in c), 'Nama Variasi')
        c_waktu = next((c for c in raw_df.columns if 'Waktu' in c or 'Tanggal' in c), 'Waktu Pesanan Dibuat')
        c_bayar = next((c for c in raw_df.columns if 'Total' in c or 'Bayar' in c), 'Total Pembayaran')

        df_work = raw_df.copy()
        # Perbaikan Variasi Kosong
        df_work[c_var] = df_work[c_var].fillna('-').replace('', '-')
        
        if not pd.api.types.is_datetime64_any_dtype(df_work[c_waktu]):
            df_work[c_waktu] = pd.to_datetime(df_work[c_waktu], errors='coerce')

        # Filter Waktu Sederhana
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

        # 1. Output Produk Terlaris
        if command_type == 'top_products':
            res = df_work.groupby([c_nama, c_var, c_sku]).agg({c_qty: 'sum'}).reset_index()
            res = res.sort_values(by=c_qty, ascending=False).head(limit)
            return {
                "type": "table", "status": "success",
                "message": f"Daftar {limit} Produk Terlaris ({time_label}):",
                "data": [{"produk": r[c_nama], "variasi": r[c_var], "sku": r[c_sku], "total": int(r[c_qty])} for _, r in res.iterrows()]
            }

        # 2. Output Rekomendasi Restock (Membutuhkan model .pkl)
        if command_type == 'rekomendasi_restock':
            unique_prods = df_work.groupby([c_nama, c_var, c_sku])[c_qty].sum().reset_index().sort_values(by=c_qty, ascending=False).head(50)
            results = []
            for _, row in unique_prods.iterrows():
                f_val = self.get_forecast(row[c_sku])
                if f_val > 0:
                    results.append({
                        "sku": row[c_sku], "nama_produk": row[c_nama], "variasi": row[c_var],
                        "prediksi_7_hari": f_val,
                        "urgensi": "KRITIS" if f_val > (row[c_qty]/4) else "NORMAL"
                    })
            return {
                "type": "table", "status": "success",
                "message": "Rekomendasi Restock 7 Hari Ke Depan:",
                "data": sorted(results, key=lambda x: x['prediksi_7_hari'], reverse=True)[:limit]
            }

        # 3. Output Analisis Tren
        if command_type == 'trend_analysis':
            trend = df_work.groupby([c_nama, c_var, c_sku])[c_qty].sum().reset_index().sort_values(by=c_qty, ascending=False).head(limit)
            return {
                "type": "table", "status": "success",
                "message": f"Analisis Tren Penjualan Periode {time_label}:",
                "data": [{"produk": r[c_nama], "variasi": r[c_var], "total": int(r[c_qty])} for _, r in trend.iterrows()]
            }

        # 4. Output Ringkasan (Card)
        if command_type == 'summary':
            return {
                "type": "summary_card", "status": "success",
                "message": f"Ringkasan Performa Toko ({time_label}):",
                "data": {
                    "total_terjual": int(df_work[c_qty].sum()),
                    "omzet": f"Rp {int(df_work[c_bayar].sum()):,}",
                    "produk_unik": int(df_work[c_sku].nunique())
                }
            }

    def extract_number(self, text):
        nums = re.findall(r'\d+', text)
        val = int(nums[0]) if nums else 5
        return min(val, 50)