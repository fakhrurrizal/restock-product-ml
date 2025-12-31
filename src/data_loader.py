import pandas as pd
import numpy as np
from src.config import Config

class DataLoader:
    def __init__(self):
        self.config = Config()

    def load_data(self, file_path):
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            
            # 1. Standarisasi Kolom Waktu
            col_time = next((c for c in df.columns if 'Waktu Pesanan Dibuat' in c or 'Tanggal' in c), None)
            if col_time:
                df[col_time] = pd.to_datetime(df[col_time], errors='coerce')
            
            # 2. Pembersihan Angka (Mencegah error format string Rp atau titik ribuan)
            col_qty = next((c for c in df.columns if 'Jumlah' in c), None)
            col_pay = next((c for c in df.columns if 'Total Pembayaran' in c), None)
            
            for col in [col_qty, col_pay]:
                if col in df.columns and df[col].dtype == 'object':
                    # Hapus simbol non-numerik kecuali titik/koma jika ada
                    df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # 3. Hapus baris yang tidak memiliki SKU atau Waktu (Data Sampah)
            col_sku = next((c for c in df.columns if 'SKU' in c), None)
            if col_sku and col_time:
                df = df.dropna(subset=[col_sku, col_time])
            
            return df
        except Exception as e:
            raise Exception(f"Gagal memuat dan membersihkan data: {str(e)}")