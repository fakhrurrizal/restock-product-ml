#!/usr/bin/env python3
"""
AUTO-MAGIC PREDICTOR - Langsung jawab pertanyaan tanpa step manual
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from predictor import RestockPredictor

class AutoMagicPredictor:
    def __init__(self, dataset_path=None):  # ⬅️ TETAP SAMA TAPI PERBAIKI METHOD auto_setup
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        self.predictor = RestockPredictor(self.model_trainer)
        
        self.dataset = None
        self.time_series_data = None
        self.daily_sales_data = None
        self.product_analysis = None
        self.is_ready = False
        self.using_real_data = False
        
        print("  Auto-Magic Predictor Initializing...")
        if dataset_path:
            self.auto_setup(dataset_path)
        else:
            self.auto_setup()
    
    def auto_setup(self, dataset_path=None):  # ⬅️ PARAMETER DIPINDAH KE SINI
        """Setup otomatis dengan dataset real atau sample"""
        try:
            # Cari dataset otomatis jika tidak ada path
            if dataset_path is None:
                dataset_path = self.find_dataset()
            
            if dataset_path and os.path.exists(dataset_path):
                print(f" Loading real dataset: {dataset_path}")
                self.dataset = self.data_loader.load_from_file(dataset_path)
                self.using_real_data = True
            else:
                print(" Creating sample data...")
                self.create_auto_sample_data()
                self.using_real_data = False
                
        except Exception as e:
            print(f"❌ Auto-setup error: {str(e)}")
    
    def create_auto_sample_data(self, days=180):
        """Buat sample data otomatis yang realistis"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Produk-produk realistic Shopee
        products = [
            {'name': 'Smartphone Android', 'sku': 'PHN-001', 'base_sales': 8, 'trend': 0.1, 'price': 2000000},
            {'name': 'Laptop Gaming', 'sku': 'LAP-002', 'base_sales': 3, 'trend': 0.05, 'price': 8000000},
            {'name': 'Headphone Wireless', 'sku': 'HP-003', 'base_sales': 15, 'trend': 0.2, 'price': 500000},
            {'name': 'Smart Watch', 'sku': 'SW-004', 'base_sales': 6, 'trend': 0.15, 'price': 1500000},
            {'name': 'Power Bank 10000mAh', 'sku': 'PB-005', 'base_sales': 12, 'trend': 0.08, 'price': 300000},
            {'name': 'Tablet Android', 'sku': 'TAB-006', 'base_sales': 4, 'trend': 0.12, 'price': 2500000},
            {'name': 'Kamera Digital', 'sku': 'CAM-007', 'base_sales': 2, 'trend': 0.03, 'price': 3500000},
            {'name': 'Speaker Bluetooth', 'sku': 'SPK-008', 'base_sales': 7, 'trend': 0.1, 'price': 800000},
        ]
        
        sales_data = []
        order_id = 1000
        
        for i, date in enumerate(dates):
            # Untuk setiap hari, generate beberapa transaksi
            daily_transactions = max(5, int(np.random.normal(15, 5)))
            
            for j in range(daily_transactions):
                product = np.random.choice(products)
                
                # Sales pattern yang realistis
                trend_component = product['base_sales'] + (product['trend'] * i)
                seasonal_component = 3 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
                noise_component = np.random.normal(0, 2)
                
                quantity = max(1, int(trend_component + seasonal_component + noise_component))
                price = product['price'] * (1 - np.random.uniform(0, 0.3))  # Discount
                
                sales_data.append({
                    'No. Pesanan': f'ORDER_{order_id}',
                    'Status Pesanan': 'selesai',
                    'Waktu Pesanan Dibuat': date.strftime('%Y-%m-%d'),
                    'Nama Produk': product['name'],
                    'SKU Induk': product['sku'],
                    'Jumlah': quantity,
                    'Harga Setelah Diskon': price,
                    'Total Pembayaran': quantity * price
                })
                order_id += 1
        
        self.dataset = pd.DataFrame(sales_data)
        print(f" Sample data created: {len(self.dataset)} transactions, {len(products)} products")
    
    def analyze_restock_urgent(self):
        """Analisis produk yang butuh restock URGENT - IMPROVED"""
        if not self.is_ready:
            return "System not ready. Please wait..."
        
        restock_needs = self.predictor.analyze_product_restock_needs(self.daily_sales_data)
        urgent_products = [p for p in restock_needs if p['urgency'] == 'HIGH']
        
        if not urgent_products:
            return "🎉 **SEMUA PRODUK AMAN!** Tidak ada yang butuh restock urgent."
        
        response = f" **PRODUK BUTUH RESTOCK URGENT** ({len(urgent_products)} produk)\n\n"
        
        for i, product in enumerate(urgent_products[:15], 1):
            response += f"{i}. {product['urgency_icon']} **{product['product_name']}**\n"
            response += f"   Stok: {product['current_stock']:.0f} unit\n"
            response += f"   Butuh: {product['restock_quantity']:.0f} unit\n" 
            response += f"   Stok habis dalam: {product['stock_cover_days']:.1f} hari\n"
            response += f"   Penjualan/hari: {product['avg_daily_sales']:.1f} unit\n"
            response += f"   SKU: {product['sku']}\n\n"
        
        total_restock = sum(p['restock_quantity'] for p in urgent_products)
        avg_cover_days = sum(p['stock_cover_days'] for p in urgent_products) / len(urgent_products)
        total_urgent_value = sum(p['total_revenue'] for p in urgent_products)
        
        response += f"**SUMMARY URGENT:**\n"
        response += f"   • Total restock needed: {total_restock:.0f} unit\n"
        response += f"   • Average stock cover: {avg_cover_days:.1f} days\n"
        response += f"   • Total revenue at risk: Rp {total_urgent_value:,.0f}\n"
        response += f"   • Most critical: {urgent_products[0]['product_name']} ({urgent_products[0]['stock_cover_days']:.1f} days)\n"
        
        response += f"\n**REKOMENDASI:**\n"
        response += f"   • Prioritaskan {urgent_products[0]['product_name']} (hampir habis)\n"
        response += f"   • Restock {len(urgent_products)} produk dalam 5 hari ke depan\n"
        response += f"   • Monitor produk dengan cover days < 7 hari\n"
        
        return response
    
    def analyze_all_restock(self):
        """Analisis semua produk yang butuh restock"""
        if not self.is_ready:
            return "❌ System not ready. Please wait..."
        
        restock_needs = self.predictor.analyze_product_restock_needs(self.daily_sales_data)
        
        if not restock_needs:
            return " **SEMUA PRODUK STOKNYA CUKUP!**"
        
        high_urgency = [p for p in restock_needs if p['urgency'] == 'HIGH']
        medium_urgency = [p for p in restock_needs if p['urgency'] == 'MEDIUM']
        low_urgency = [p for p in restock_needs if p['urgency'] == 'LOW']
        
        response = f" **ANALISIS RESTOCK LENGKAP**\n\n"
        
        response += f" **URGENT** ({len(high_urgency)} produk):\n"
        for product in high_urgency[:3]:
            response += f"   • {product['product_name']} - Restock {product['restock_quantity']:.0f} unit\n"
        
        response += f"\n🟡 **MEDIUM** ({len(medium_urgency)} produk):\n"
        for product in medium_urgency[:3]:
            response += f"   • {product['product_name']} - Restock {product['restock_quantity']:.0f} unit\n"
        
        response += f"\n🟢 **LOW** ({len(low_urgency)} produk):\n"
        for product in low_urgency[:2]:
            response += f"   • {product['product_name']} - Restock {product['restock_quantity']:.0f} unit\n"
        
        # Totals
        response += f"\n **TOTAL:** {len(restock_needs)} produk butuh restock\n"
        response += f"   • Urgent: {len(high_urgency)} produk\n"
        response += f"   • Medium: {len(medium_urgency)} produk\n"
        response += f"   • Low: {len(low_urgency)} produk\n"
        
        return response
    
    def get_top_products(self, by='quantity'):
        """Dapatkan produk terlaris"""
        if not self.is_ready:
            return "❌ System not ready. Please wait..."
        
        top_products = self.predictor.get_top_products_analysis(self.product_analysis)
        
        if by == 'revenue':
            products = top_products.get('top_by_revenue', [])[:5]
            title = "💰 PRODUK TERLARIS BY REVENUE"
        else:
            products = top_products.get('top_by_quantity', [])[:5]
            title = "🏆 PRODUK TERLARIS BY QUANTITY"
        
        response = f"{title}\n\n"
        
        for i, product in enumerate(products, 1):
            response += f"{i}. **{product['Nama Produk']}**\n"
            response += f"    Terjual: {product['Total_Quantity']:.0f} unit\n"
            response += f"   💰 Revenue: Rp {product['Total_Revenue']:,.0f}\n"
            response += f"   🛒 Transaksi: {product['Transaction_Count']}x\n\n"
        
        return response
    
    def predict_demand(self, days=30):
        """Prediksi demand ke depan - FIXED"""
        if not self.is_ready:
            return "❌ System not ready. Please wait..."
        
        results = self.predictor.predict_and_recommend(days=days)
        
        if not results:
            return "❌ Prediction failed."
        
        response = f"  **PREDIKSI {days} HARI KE DEPAN**\n\n"
        
        # Key metrics
        predictions = results.get('predictions', [])
        if predictions:
            avg_demand = sum(p['predicted_demand'] for p in predictions) / len(predictions)
            
            response += f" **Key Metrics:**\n"
            response += f"   • Rata-rata demand: {avg_demand:.1f} unit/hari\n"
            response += f"   • Total prediksi demand: {sum(p['predicted_demand'] for p in predictions):.0f} unit\n"
            
            # Check for urgent days
            recommendations = results.get('restock_recommendations', [])
            if recommendations:
                urgent_days = sum(1 for r in recommendations if r['urgency'] == 'HIGH')
                response += f"   • Hari butuh restock urgent: {urgent_days} hari\n"
            
            response += f"\n"
            
            # 5 days preview
            response += f"📅 **5 Hari Pertama:**\n"
            for i, pred in enumerate(predictions[:5], 1):
                urgency_info = ""
                if recommendations:
                    for rec in recommendations:
                        if rec['date'] == pred['date']:
                            urgency_info = f" {rec['urgency_color']}"
                            break
                response += f"   {i}. {pred['date']}{urgency_info} {pred['predicted_demand']:.0f} unit\n"
        else:
            response += "❌ Tidak ada data prediksi\n"
        
        return response
    
    def get_sales_trend(self):
        """Analisis tren penjualan"""
        if not self.is_ready:
            return "❌ System not ready. Please wait..."
        
        trend_analysis = self.predictor.generate_sales_trend_analysis(self.time_series_data)
        
        response = " **ANALISIS TREN PENJUALAN**\n\n"
        
        response += f" **Status Tren:** {trend_analysis['trend_icon']} {trend_analysis['trend_direction']}\n"
        response += f" **Rata-rata Penjualan:** {trend_analysis['recent_avg_daily_sales']:.1f} unit/hari\n"
        response += f"🎯 **Volatilitas:** {trend_analysis['volatility']:.1f}\n"
        response += f"📅 **Periode Analisis:** {trend_analysis['analysis_period_days']} hari\n"
        
        # Interpretation
        if trend_analysis['trend_direction'] == 'INCREASING':
            response += "\n **INTERPRETASI:** Penjualan sedang naik! Pertahankan performa."
        elif trend_analysis['trend_direction'] == 'DECREASING':
            response += "\n⚠️ **INTERPRETASI:** Penjualan sedang turun. Perlu evaluasi strategi."
        else:
            response += "\n➡️ **INTERPRETASI:** Penjualan stabil. Good job!"
        
        return response
    
    def setup(self, dataset_path=None):
        """Setup system dengan atau tanpa dataset - FIXED"""
        print("  Auto-Magic Predictor Initializing...")
        
        try:
            # Cari dataset otomatis jika tidak ada path
            if dataset_path is None:
                dataset_path = self.find_dataset()
            
            if dataset_path and os.path.exists(dataset_path):
                print(f" Loading real dataset: {dataset_path}")
                self.dataset = self.data_loader.load_from_file(dataset_path)
                self.using_real_data = True
            else:
                print(" Creating sample data...")
                self.create_auto_sample_data()
                self.using_real_data = False
            
            if self.dataset is None:
                print("❌ Failed to load data")
                return False
            
            print(" Auto-processing data...")
            self.time_series_data, self.daily_sales_data, self.product_analysis = \
                self.preprocessor.preprocess_data(self.dataset)
            
            if self.time_series_data.empty:
                print("❌ Processing failed - no time series data")
                return False
            
            print(" Auto-training model...")
            train_data, test_data = self.preprocessor.prepare_training_data(self.time_series_data)
            success = self.model_trainer.train_sarima(train_data)
            
            if success:
                self.is_ready = True
                data_type = "REAL DATASET" if self.using_real_data else "SAMPLE DATA"
                print(f" Auto-setup COMPLETED with {data_type}! Ready for questions.")
                
                # ⬇️ SIMPAN REFERENCES KE MODEL TRAINER
                self.model_trainer.time_series_data = self.time_series_data
                self.model_trainer.product_analysis = self.product_analysis
                
                self.show_dataset_info()
                return True
            else:
                print("❌ Auto-setup failed at model training")
                return False
                
        except Exception as e:
            print(f"❌ Auto-setup error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def find_dataset(self):
        """Cari dataset CSV secara otomatis di folder data/"""
        possible_locations = [
            'data/raw/*.csv',
            'data/*.csv', 
            '*.csv',
            '../*.csv',
            'dataset/*.csv'
        ]
        
        for location in possible_locations:
            files = glob.glob(location)
            for file in files:
                if file.endswith('.csv') and os.path.getsize(file) > 1024:  # Minimal 1KB
                    print(f" Found dataset: {file}")
                    return file
        
        print("❌ No dataset found. Using sample data.")
        return None
    
    def auto_setup(self, dataset_path=None):
        """Setup otomatis dengan dataset real atau sample"""
        try:
            # Cari dataset otomatis jika tidak ada path
            if dataset_path is None:
                dataset_path = self.find_dataset()
            
            if dataset_path and os.path.exists(dataset_path):
                print(f" Loading real dataset: {dataset_path}")
                self.dataset = self.data_loader.load_from_file(dataset_path)
                self.using_real_data = True
            else:
                print(" Creating sample data...")
                self.create_auto_sample_data()
                self.using_real_data = False
            
            if self.dataset is None:
                print("❌ Failed to load data")
                return
            
            print(" Auto-processing data...")
            self.time_series_data, self.daily_sales_data, self.product_analysis = \
                self.preprocessor.preprocess_data(self.dataset)
            
            if self.time_series_data.empty:
                print("❌ Processing failed - no time series data")
                return
            
            print(" Auto-training model...")
            train_data, test_data = self.preprocessor.prepare_training_data(self.time_series_data)
            success = self.model_trainer.train_sarima(train_data)
            
            if success:
                self.is_ready = True
                data_type = "REAL DATASET" if self.using_real_data else "SAMPLE DATA"
                print(f" Auto-setup COMPLETED with {data_type}! Ready for questions.")
                
                # Show dataset info
                self.show_dataset_info()
            else:
                print("❌ Auto-setup failed at model training")
                
        except Exception as e:
            print(f"❌ Auto-setup error: {str(e)}")
    
    def show_dataset_info(self):
        """Tampilkan info dataset yang digunakan"""
        if self.dataset is not None:
            print(f"\n DATASET INFO:")
            print(f"   • Shape: {self.dataset.shape}")
            print(f"   • Columns: {len(self.dataset.columns)}")
            print(f"   • Memory usage: {self.dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Check important columns
            important_cols = ['Waktu Pesanan Dibuat', 'Jumlah', 'Nama Produk', 'SKU Induk']
            missing_cols = [col for col in important_cols if col not in self.dataset.columns]
            if missing_cols:
                print(f"    Missing columns: {missing_cols}")
            else:
                print(f"    All important columns present")
            
            if 'Waktu Pesanan Dibuat' in self.dataset.columns:
                dates = pd.to_datetime(self.dataset['Waktu Pesanan Dibuat'], errors='coerce')
                valid_dates = dates.dropna()
                if not valid_dates.empty:
                    print(f"   • Date range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}")
            
            if 'Jumlah' in self.dataset.columns:
                total_sales = self.dataset['Jumlah'].sum()
                print(f"   • Total sales quantity: {total_sales:,.0f}")


    def get_top_products(self, by='quantity'):
        """Dapatkan produk terlaris - FIXED"""
        if not self.is_ready:
            return "❌ System not ready. Please wait..."
        
        top_products = self.predictor.get_top_products_analysis(self.product_analysis)
        
        if by == 'revenue':
            products = top_products.get('top_by_revenue', [])
            title = "💰 PRODUK TERLARIS BY REVENUE"
        else:
            products = top_products.get('top_by_quantity', [])
            title = "🏆 PRODUK TERLARIS BY QUANTITY"
        
        if not products:
            return f"{title}\n\n❌ Tidak ada data produk terlaris"
        
        response = f"{title}\n\n"
        
        for i, product in enumerate(products[:5], 1):
            product_name = product.get('Nama Produk', 'Unknown Product')
            quantity = product.get('Total_Quantity', 0)
            revenue = product.get('Total_Revenue', 0)
            transactions = product.get('Transaction_Count', 0)
            
            response += f"{i}. **{product_name}**\n"
            response += f"    Terjual: {quantity:.0f} unit\n"
            response += f"   💰 Revenue: Rp {revenue:,.0f}\n"
            response += f"   🛒 Transaksi: {transactions}x\n\n"
        
        return response

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Shopee Auto-Magic Predictor')
    parser.add_argument('--dataset', '-d', type=str, help='Path to your dataset CSV file')
    args = parser.parse_args()
    
    print("**SHOPEE AUTO-MAGIC PREDICTOR**")
    print("System akan setup otomatis...")
    
    if args.dataset:
        print(f" Using dataset: {args.dataset}")
    else:
        print(" Auto-searching for dataset...")
    
    print("-" * 60)
    
    predictor = AutoMagicPredictor() 
    
    if args.dataset:
        predictor.auto_setup(args.dataset)  
    
    if not predictor.is_ready:
        print("System gagal setup. Coba lagi.")
        return
    
    print("\n**SYSTEM READY!** Silakan tanyakan apa saja:")
    print("• 'restock urgent' - Produk butuh restock segera")
    print("• 'analisis restock' - Semua produk butuh restock") 
    print("• 'produk terlaris' - Top produk by quantity")
    print("• 'top revenue' - Top produk by revenue")
    print("• 'prediksi' - Prediksi demand 30 hari")
    print("• 'tren penjualan' - Analisis tren")
    print("• 'semua' - Semua analisis sekaligus")
    print("• 'exit' - Keluar")
    print("-" * 60)
    
    while True:
        try:
            prompt = input("\n Tanya: ").strip().lower()
            
            if prompt in ['exit', 'quit', 'keluar']:
                print("👋 Terima kasih!")
                break
            
            if not prompt:
                continue
            
            # Process prompt
            if 'urgent' in prompt:
                response = predictor.analyze_restock_urgent()
            elif 'restock' in prompt:
                response = predictor.analyze_all_restock()
            elif 'revenue' in prompt:
                response = predictor.get_top_products(by='revenue')
            elif 'terlaris' in prompt or 'top' in prompt:
                response = predictor.get_top_products(by='quantity')
            elif 'prediksi' in prompt or 'ramal' in prompt:
                # Extract days if any
                days_match = re.search(r'(\d+)', prompt)
                days = int(days_match.group(1)) if days_match else 30
                response = predictor.predict_demand(days=days)
            elif 'tren' in prompt or 'trend' in prompt:
                response = predictor.get_sales_trend()
            elif 'semua' in prompt or 'all' in prompt:
                # Combine all analyses
                response = " **LAPORAN LENGKAP**\n\n"
                response += predictor.analyze_restock_urgent() + "\n" + "="*50 + "\n\n"
                response += predictor.get_top_products() + "\n" + "="*50 + "\n\n"
                response += predictor.predict_demand() + "\n" + "="*50 + "\n\n"
                response += predictor.get_sales_trend()
            else:
                response = "❌ Tidak mengerti. Coba: 'restock urgent', 'produk terlaris', 'prediksi', dll."
            
            print(f"\n {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Sampai jumpa!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()