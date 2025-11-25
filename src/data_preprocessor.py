import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config

class DataPreprocessor:
    def __init__(self):
        self.config = Config()
    
    def __init__(self):
        self.config = Config()
    
    def preprocess_data(self, df):
        """
        Preprocessing data untuk dataset real
        """
        print(" Memulai preprocessing data...")
        
        # 1. Cleaning data
        df_clean = self._clean_data(df)
        
        # 2. Filter data
        df_filtered = self._filter_data(df_clean)
        
        # 3. Transform data
        df_transformed = self._transform_data(df_filtered)
        
        # 4. Create time series data
        time_series = self._create_time_series(df_transformed)
        
        # 5. Create daily sales data
        daily_sales = self._create_daily_sales(df_transformed)
        
        # 6. Product analysis
        product_analysis = self._analyze_products(df_transformed)
        
        print(" Preprocessing data selesai")
        return time_series, daily_sales, product_analysis
    
    def _clean_data(self, df):
        """Pembersihan data untuk dataset real"""
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        print(f" Cleaning data: {initial_rows} rows initially")
        
        # Handle missing values di kolom penting
        important_cols = []
        if 'Waktu Pesanan Dibuat' in df_clean.columns:
            important_cols.append('Waktu Pesanan Dibuat')
        if 'Jumlah' in df_clean.columns:
            important_cols.append('Jumlah')
        
        if important_cols:
            df_clean = df_clean.dropna(subset=important_cols)
            print(f"   Removed rows with missing important columns: {initial_rows - len(df_clean)}")
        
        # Hapus duplikat
        initial_after_missing = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"   Removed duplicates: {initial_after_missing - len(df_clean)}")
        
        # Clean numeric columns
        if 'Jumlah' in df_clean.columns:
            df_clean['Jumlah'] = pd.to_numeric(df_clean['Jumlah'], errors='coerce')
            # Remove negative quantities
            df_clean = df_clean[df_clean['Jumlah'] > 0]
            print(f"   Removed invalid quantities: {len(df_clean)} rows remaining")
        
        print(f"   Final: {len(df_clean)} rows ({initial_rows - len(df_clean)} removed)")
        return df_clean
    
    def _transform_data(self, df):
        """Transformasi data dengan auto-detect kolom"""
        df_transformed = df.copy()
        
        print("    Auto-detecting columns...")
        
        # Auto-detect kolom tanggal
        date_columns = [col for col in df_transformed.columns if 'tanggal' in col.lower() or 'waktu' in col.lower() or 'date' in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            print(f"      Using date column: '{date_col}'")
            df_transformed['Waktu Pesanan Dibuat'] = pd.to_datetime(df_transformed[date_col], errors='coerce')
        else:
            print("      ⚠️ No date column found")
            df_transformed['Waktu Pesanan Dibuat'] = pd.to_datetime('today')
        
        # Auto-detect kolom SKU
        sku_columns = [col for col in df_transformed.columns if 'sku induk' in col.lower() or 'sku_induk' in col.lower() or 'sku' in col.lower()]
        if sku_columns:
            sku_col = sku_columns[0]
            print(f"      Using SKU column: '{sku_col}'")
            df_transformed['SKU Induk'] = df_transformed[sku_col]
        else:
            print("      ⚠️ No SKU column found, generating from product name...")
            # Generate SKU dari nama produk atau nomor referensi
            if 'Nomor Referensi SKU' in df_transformed.columns:
                df_transformed['SKU Induk'] = df_transformed['Nomor Referensi SKU']
            elif 'Nama Produk' in df_transformed.columns:
                # Extract 2-4 huruf kapital dari nama produk sebagai SKU
                df_transformed['SKU Induk'] = df_transformed['Nama Produk'].str.extract(r'([A-Z]{2,4})')[0]
                df_transformed['SKU Induk'] = df_transformed['SKU Induk'].fillna('PROD')
            else:
                df_transformed['SKU Induk'] = 'SKU_' + df_transformed.index.astype(str)
        
        # Auto-detect kolom quantity
        qty_columns = [col for col in df_transformed.columns if 'jumlah' in col.lower() or 'quantity' in col.lower() or 'qty' in col.lower()]
        if qty_columns:
            qty_col = qty_columns[0]
            print(f"      Using quantity column: '{qty_col}'")
            df_transformed['Jumlah'] = pd.to_numeric(df_transformed[qty_col], errors='coerce')
        else:
            print("      ⚠️ No quantity column found")
            df_transformed['Jumlah'] = 1  # Default quantity
        
        # Auto-detect kolom produk
        product_columns = [col for col in df_transformed.columns if 'produk' in col.lower() or 'product' in col.lower() or 'nama' in col.lower()]
        if product_columns:
            product_col = product_columns[0]
            print(f"      Using product column: '{product_col}'")
            df_transformed['Nama Produk'] = df_transformed[product_col].astype(str).str.slice(0, 50)  # Potong nama panjang
        
        # Clean data
        df_transformed['Jumlah'] = pd.to_numeric(df_transformed['Jumlah'], errors='coerce').fillna(1)
        df_transformed['Jumlah'] = df_transformed['Jumlah'].clip(lower=1)  # Minimal 1
        
        # Tambahkan kolom tanggal untuk aggregasi
        df_transformed['Tanggal'] = df_transformed['Waktu Pesanan Dibuat'].dt.date
        
        print(f"      Final columns: {list(df_transformed.columns)}")
        return df_transformed
    
    def _filter_data(self, df):
        """Filter data berdasarkan status pesanan"""
        df_filtered = df.copy()
        
        # Filter hanya pesanan yang sukses
        if 'Status Pesanan' in df_filtered.columns:
            success_keywords = ['selesai', 'completed', 'dikirim', 'delivered', 'sampai']
            mask = df_filtered['Status Pesanan'].str.contains(
                '|'.join(success_keywords), case=False, na=False
            )
            df_filtered = df_filtered[mask]
        
        print(f" Data filtering: {len(df)} -> {len(df_filtered)} rows")
        return df_filtered
    
    def _create_time_series(self, df):
        """Membuat data time series untuk forecasting"""
        if 'Tanggal' not in df.columns:
            return pd.DataFrame()
        
        # Aggregasi harian
        daily_agg = df.groupby('Tanggal').agg({
            self.config.TARGET_COLUMN: 'sum',
            'Total Pembayaran': 'sum',
            'Nama Produk': 'count'  # Jumlah transaksi
        }).reset_index()
        
        daily_agg.columns = ['Tanggal', 'Total_Quantity', 'Total_Revenue', 'Transaction_Count']
        daily_agg['Tanggal'] = pd.to_datetime(daily_agg['Tanggal'])
        
        # Buat time series lengkap
        full_date_range = pd.date_range(
            start=daily_agg['Tanggal'].min(),
            end=daily_agg['Tanggal'].max(),
            freq='D'
        )
        
        time_series = pd.DataFrame({'Tanggal': full_date_range})
        time_series = time_series.merge(daily_agg, on='Tanggal', how='left')
        
        # Fill missing values dengan 0
        numeric_cols = ['Total_Quantity', 'Total_Revenue', 'Transaction_Count']
        time_series[numeric_cols] = time_series[numeric_cols].fillna(0)
        
        time_series = time_series.set_index('Tanggal').sort_index()
        return time_series
    
    def _create_daily_sales(self, df):
        """Membuat data penjualan harian per produk"""
        if 'Tanggal' not in df.columns or 'SKU Induk' not in df.columns:
            return pd.DataFrame()
        
        daily_sales = df.groupby(['Tanggal', 'SKU Induk', 'Nama Produk']).agg({
            self.config.TARGET_COLUMN: 'sum',
            'Total Pembayaran': 'sum',
            'Harga Setelah Diskon': 'mean'
        }).reset_index()
        
        daily_sales['Tanggal'] = pd.to_datetime(daily_sales['Tanggal'])
        return daily_sales
    
    def _analyze_products(self, df):
        """Analisis produk"""
        product_analysis = {}
        
        if 'SKU Induk' in df.columns and 'Nama Produk' in df.columns:
            # Top products by quantity
            top_products_qty = df.groupby(['SKU Induk', 'Nama Produk']).agg({
                self.config.TARGET_COLUMN: ['sum', 'count', 'mean'],
                'Total Pembayaran': 'sum'
            }).round(2)
            
            top_products_qty.columns = ['Total_Quantity', 'Transaction_Count', 'Avg_Quantity_Per_Order', 'Total_Revenue']
            product_analysis['top_products'] = top_products_qty.reset_index()
        
        return product_analysis
    
    def prepare_training_data(self, time_series, target_column='Total_Quantity', test_size=0.2):
        """
        Mempersiapkan data training dan testing
        
        Args:
            time_series (pd.DataFrame): Data time series
            target_column (str): Kolom target
            test_size (float): Proporsi data testing
            
        Returns:
            tuple: (train_data, test_data)
        """
        if target_column not in time_series.columns:
            raise ValueError(f"Target column '{target_column}' tidak ditemukan")
        
        # Split data
        split_idx = int(len(time_series) * (1 - test_size))
        train_data = time_series[target_column][:split_idx]
        test_data = time_series[target_column][split_idx:]
        
        print(f" Data splitting: Train={len(train_data)}, Test={len(test_data)}")
        return train_data, test_data