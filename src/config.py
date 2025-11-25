# Konfigurasi global untuk project
import os
from datetime import datetime

class Config:
    # Path configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    
    # Model configuration
    DEFAULT_MODEL_TYPE = 'SARIMA'  # 'ARIMA' or 'SARIMA'
    ARIMA_ORDER = (1, 1, 1)
    SARIMA_ORDER = (1, 1, 1)
    SARIMA_SEASONAL_ORDER = (1, 1, 1, 7)  # Weekly seasonality
    
    # Training configuration
    TEST_SIZE = 0.2
    TARGET_COLUMN = 'Jumlah'
    DATE_COLUMN = 'Waktu Pesanan Dibuat'
    
    # Prediction configuration
    DEFAULT_PREDICTION_DAYS = 30
    SAFETY_STOCK_RATIO = 0.2
    
    COLUMN_MAPPING = {
        'date_columns': ['waktu pesanan dibuat', 'tanggal', 'date', 'order_date'],
        'sku_columns': ['sku induk', 'sku_induk', 'sku', 'product_id', 'kode produk'],
        'quantity_columns': ['jumlah', 'quantity', 'qty', 'qty_sold'],
        'product_columns': ['nama produk', 'product', 'product_name', 'item'],
        'price_columns': ['harga setelah diskon', 'price', 'harga', 'unit_price']
    }
    
    # Feature columns dari dataset Shopee
    FEATURE_COLUMNS = [
        'No. Pesanan', 'Status Pesanan', 'Alasan Pembatalan', 
        'Status Pembatalan/ Pengembalian', 'No. Resi', 'Opsi Pengiriman',
        'Antar ke counter/ pick-up', 'Pesanan Harus Dikirimkan Sebelum',
        'Waktu Pengiriman Diatur', 'Waktu Pesanan Dibuat', 
        'Waktu Pembayaran Dilakukan', 'Metode Pembayaran', 'SKU Induk',
        'Nama Produk', 'Nomor Referensi SKU', 'Nama Variasi', 'Harga Awal',
        'Harga Setelah Diskon', 'Jumlah', 'Returned quantity', 
        'Total Harga Produk', 'Total Diskon', 'Diskon Dari Penjual',
        'Diskon Dari Shopee', 'Berat Produk', 'Jumlah Produk di Pesan',
        'Total Berat', 'Voucher Ditanggung Penjual', 'Cashback Koin',
        'Voucher Ditanggung Shopee', 'Paket Diskon', 
        'Paket Diskon (Diskon dari Shopee)', 
        'Paket Diskon (Diskon dari Penjual)', 'Potongan Koin Shopee',
        'Diskon Kartu Kredit', 'Ongkos Kirim Dibayar oleh Pembeli',
        'Estimasi Potongan Biaya Pengiriman', 'Ongkos Kirim Pengembalian Barang',
        'Total Pembayaran', 'Perkiraan Ongkos Kirim', 'Catatan dari Pembeli',
        'Catatan', 'Username (Pembeli)', 'Nama Penerima', 'No. Telepon',
        'Alamat Pengiriman', 'Kota/Kabupaten', 'Provinsi', 'Waktu Pesanan Selesai'
    ]
    
    @classmethod
    def ensure_directories(cls):
        """Membuat direktori yang diperlukan"""
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(cls.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)