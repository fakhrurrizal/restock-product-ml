import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads/")
    
    DATE_COLUMN = 'Waktu Pesanan Dibuat'
    TARGET_COLUMN = 'Jumlah'
    SKU_COLUMN = 'SKU Induk'
    PRODUCT_NAME = 'Nama Produk'
    STATUS_COLUMN = 'Status Pesanan'
    
    FORECAST_PERIOD = 7  