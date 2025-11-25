import pandas as pd
import numpy as np
import requests
from urllib.parse import urlparse
import os
from config import Config

class DataLoader:
    def __init__(self):
        self.config = Config()
    
    def load_from_url(self, url):
        """
        Load data dari URL (CSV atau Excel)
        
        Args:
            url (str): URL menuju file dataset
            
        Returns:
            pandas.DataFrame: Dataset yang telah diload
        """
        try:
            # Download file dari URL
            response = requests.get(url)
            response.raise_for_status()
            
            # Simpan file sementara
            temp_path = os.path.join(self.config.RAW_DATA_DIR, 'temp_dataset')
            
            # Tentukan ekstensi file
            parsed_url = urlparse(url)
            file_extension = os.path.splitext(parsed_url.path)[1].lower()
            
            if file_extension == '.csv':
                temp_path += '.csv'
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                df = pd.read_csv(temp_path)
            elif file_extension in ['.xlsx', '.xls']:
                temp_path += '.xlsx'
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                df = pd.read_excel(temp_path)
            else:
                raise ValueError(f"Format file tidak didukung: {file_extension}")
            
            print(f" Data berhasil diload: {df.shape}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data dari URL: {str(e)}")
            return None
    
    def __init__(self):
        self.config = Config()
    
    def load_from_file(self, file_path, encoding=None):
        """
        Load data dari file lokal dengan auto-detect encoding
        """
        try:
            if not os.path.exists(file_path):
                print(f"❌ File tidak ditemukan: {file_path}")
                return None
            
            # Auto-detect encoding untuk CSV
            if file_path.endswith('.csv') and encoding is None:
                encoding = self.detect_encoding(file_path)
                print(f" Detected encoding: {encoding}")
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                # Coba berbagai encoding
                encodings_to_try = [encoding, 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                if encoding:
                    encodings_to_try = [encoding] + [e for e in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252'] if e != encoding]
                
                for enc in encodings_to_try:
                    try:
                        df = pd.read_csv(file_path, encoding=enc)
                        print(f" Successfully loaded with encoding: {enc}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    print("❌ Failed to load with any encoding")
                    return None
                    
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Format file tidak didukung: {file_extension}")
            
            print(f" Data berhasil diload: {df.shape}")
            
            # Clean column names (handle spasi, karakter khusus)
            df.columns = [str(col).strip().replace('\n', ' ').replace('\r', ' ') for col in df.columns]
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data dari file: {str(e)}")
            return None
    
    def detect_encoding(self, file_path):
        """Detect file encoding"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                return result['encoding']
        except:
            return 'utf-8'
    
    def validate_dataset_structure(self, df):
        """
        Validasi dan adaptasi struktur dataset
        """
        if df is None or df.empty:
            return df
        
        df_clean = df.copy()
        
        # Mapping nama kolom alternatif
        column_mapping = {
            'tanggal': 'Waktu Pesanan Dibuat',
            'date': 'Waktu Pesanan Dibuat', 
            'order_date': 'Waktu Pesanan Dibuat',
            'product': 'Nama Produk',
            'product_name': 'Nama Produk',
            'item': 'Nama Produk',
            'sku': 'SKU Induk',
            'product_id': 'SKU Induk',
            'quantity': 'Jumlah',
            'qty': 'Jumlah',
            'amount': 'Jumlah',
            'price': 'Harga Setelah Diskon',
            'total_price': 'Total Pembayaran',
            'revenue': 'Total Pembayaran',
            'status': 'Status Pesanan'
        }
        
        # Rename columns jika ada yang match
        current_columns = [col.lower().strip() for col in df_clean.columns]
        for col in df_clean.columns:
            col_lower = col.lower().strip()
            if col_lower in column_mapping and column_mapping[col_lower] not in df_clean.columns:
                df_clean = df_clean.rename(columns={col: column_mapping[col_lower]})
                print(f" Renamed column: '{col}' -> '{column_mapping[col_lower]}'")
        
        # Validasi kolom penting
        important_columns = ['Waktu Pesanan Dibuat', 'Jumlah']
        missing_important = [col for col in important_columns if col not in df_clean.columns]
        
        if missing_important:
            print(f"⚠️  Missing important columns: {missing_important}")
            print(f"   Available columns: {list(df_clean.columns)}")
        
        return df_clean

    
    def validate_dataset(self, df):
        """
        Validasi struktur dataset dengan matching yang lebih fleksibel
        """
        validation_result = {
            'is_valid': True,
            'missing_columns': [],
            'data_types': {},
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
        # Check kolom penting dengan matching fleksibel
        important_columns = [self.config.DATE_COLUMN, self.config.TARGET_COLUMN, 'Nama Produk']
        
        for col in important_columns:
            # Cari kolom dengan nama yang mirip
            found = False
            for actual_col in df.columns:
                if col.lower() in actual_col.lower():
                    found = True
                    break
            
            if not found:
                validation_result['missing_columns'].append(col)
                validation_result['is_valid'] = False
        
        # Untuk SKU Induk, cari variasi nama
        sku_variations = ['sku induk', 'sku_induk', 'sku', 'product_id', 'kode produk']
        sku_found = False
        for sku_var in sku_variations:
            for actual_col in df.columns:
                if sku_var in actual_col.lower():
                    sku_found = True
                    print(f"    Found SKU column: '{actual_col}'")
                    break
            if sku_found:
                break
        
        if not sku_found:
            validation_result['missing_columns'].append('SKU Induk')
            print(f"   ⚠️  SKU column not found. Available columns: {list(df.columns)}")
        
        # Cek tipe data
        for col in df.columns:
            validation_result['data_types'][col] = str(df[col].dtype)
        
        return validation_result
    
    def get_dataset_info(self, df):
        """
        Mendapatkan informasi dataset dengan detil kolom
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'date_range': None,
            'total_sales': 0,
            'total_products': 0,
            'column_details': {}
        }
        
        # Tampilkan detail setiap kolom
        print(f"\n📋 COLUMN DETAILS:")
        for col in df.columns:
            non_null = df[col].count()
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            info['column_details'][col] = {
                'non_null': non_null,
                'null_count': null_count,
                'unique_count': unique_count,
                'dtype': str(df[col].dtype)
            }
            print(f"   • {col}: {non_null} non-null, {null_count} null, {unique_count} unique")
        
        # Cari kolom tanggal
        date_columns = [col for col in df.columns if 'tanggal' in col.lower() or 'waktu' in col.lower() or 'date' in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            print(f"   📅 Using date column: '{date_col}'")
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            date_range = df[date_col].agg(['min', 'max'])
            info['date_range'] = {
                'start': date_range['min'].strftime('%Y-%m-%d') if pd.notna(date_range['min']) else 'N/A',
                'end': date_range['max'].strftime('%Y-%m-%d') if pd.notna(date_range['max']) else 'N/A'
            }
        
        # Cari kolom quantity
        qty_columns = [col for col in df.columns if 'jumlah' in col.lower() or 'quantity' in col.lower() or 'qty' in col.lower()]
        if qty_columns:
            qty_col = qty_columns[0]
            print(f"    Using quantity column: '{qty_col}'")
            info['total_sales'] = int(df[qty_col].sum())
        
        # Cari kolom produk
        product_columns = [col for col in df.columns if 'produk' in col.lower() or 'product' in col.lower() or 'nama' in col.lower()]
        if product_columns:
            product_col = product_columns[0]
            print(f"   Using product column: '{product_col}'")
            info['total_products'] = df[product_col].nunique()
        
        return info