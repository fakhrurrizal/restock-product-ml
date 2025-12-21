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
            
    def process_natural_language(self, prompt):
        """Process natural language prompts dengan pattern matching yang lebih spesifik"""
        prompt_lower = prompt.lower().strip()
        
        print(f"Memahami: '{prompt}'")
        
        # Pattern matching untuk berbagai jenis pertanyaan
        patterns = {
            'restock_urgent': [
                'restock urgent', 'stok urgent', 'butuh restock segera', 'stok hampir habis',
                'produk yang harus direstock', 'yang harus dibeli lagi', 'stok menipis',
                'urgent', 'segera', 'cepat', 'prioritas', 'stok paling dikit', 'stok terkecil',
                'stok sedikit', 'hampir habis', 'nyaris habis'
            ],
            'restock_all': [
                'restock', 'stok', 'butuh restock', 'perlu restock', 'analisis restock',
                'semua restock', 'semua stok', 'semua produk restock'
            ],
            'top_products': [
                'produk terlaris', 'top produk', 'best seller', 'penjualan tertinggi',
                'produk paling laris', 'terlaris', 'laku', 'best seller'
            ],
            'top_revenue': [
                'pendapatan tertinggi', 'revenue tertinggi', 'penghasilan tertinggi',
                'produk revenue', 'uang terbanyak', 'profit tertinggi'
            ],
            'prediction': [
                'prediksi', 'ramalan', 'forecast', 'perkiraan', 'prediksi penjualan',
                'ramalan penjualan', 'perkiraan kedepan', 'berapa hari kedepan',
                'prediksi demand', 'perkiraan permintaan'
            ],
            'trend': [
                'tren', 'trend', 'analisis tren', 'perkembangan penjualan',
                'penjualan naik turun', 'grafik penjualan', 'kecenderungan'
            ],
            'summary': [
                'semua', 'ringkasan', 'summary', 'overview', 'laporan lengkap',
                'analisis lengkap', 'semua analisis', 'dashboard'
            ],
            'help': [
                'bantuan', 'help', 'menu', 'perintah', 'cara pakai', 'fitur',
                'bisa apa', 'tanya apa', 'command'
            ],
            'stock_most': [
                'stok paling banyak', 'stok terbanyak', 'stok besar', 'stok tinggi',
                'stok melimpah', 'stok berlebih'
            ],
            'stock_least': [
                'stok paling dikit', 'stok paling sedikit', 'stok terkecil', 'stok rendah',
                'stok kecil', 'stok minim'
            ]
        }
        
        # Cek pattern mana yang match
        matched_patterns = []
        for pattern_type, keywords in patterns.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    matched_patterns.append(pattern_type)
                    break
        
        # Handle berdasarkan pattern yang match
        if not matched_patterns:
            return self.handle_unknown_prompt(prompt)
        
        # Ambil pattern dengan priority
        priority_order = ['stock_least', 'stock_most', 'restock_urgent', 'restock_all', 'prediction', 'top_products', 'trend', 'summary', 'help']
        for pattern in priority_order:
            if pattern in matched_patterns:
                return self.execute_command(pattern, prompt)
        
        return self.execute_command(matched_patterns[0], prompt)
    
    def analyze_stock_most(self):
        if not self.is_ready:
            return "System not ready. Please wait..."
        
        restock_needs = self.predictor.analyze_product_restock_needs(self.daily_sales_data)
        
        if not restock_needs:
            return "Tidak ada data stok untuk dianalisis"
        
        sorted_products = sorted(restock_needs, key=lambda x: x['current_stock'], reverse=True)
        
        response = "PRODUK DENGAN STOK PALING BANYAK\n\n"
        
        for i, product in enumerate(sorted_products[:10], 1):  
            response += f"{i}. {product['product_name'][:50]}...\n"
            response += f"   Stok: {product['current_stock']:.0f} unit\n"
            response += f"   Penjualan/hari: {product['avg_daily_sales']:.1f} unit\n"
            response += f"   Stok cukup untuk: {product['stock_cover_days']:.1f} hari\n"
            response += f"   SKU: {product['sku']}\n\n"
        
        # Summary
        total_stock = sum(p['current_stock'] for p in sorted_products[:10])
        avg_cover_days = sum(p['stock_cover_days'] for p in sorted_products[:10]) / len(sorted_products[:10])
        
        response += f"SUMMARY:\n"
        response += f"   Total stok 10 produk: {total_stock:.0f} unit\n"
        response += f"   Rata-rata cover days: {avg_cover_days:.1f} hari\n"
        response += f"   Produk dengan stok terbanyak: {sorted_products[0]['product_name'][:30]}... ({sorted_products[0]['current_stock']:.0f} unit)\n"
        
        return response

    def analyze_stock_least(self):
        """Analisis produk dengan stok paling sedikit"""
        if not self.is_ready:
            return "System not ready. Please wait..."
        
        restock_needs = self.predictor.analyze_product_restock_needs(self.daily_sales_data)
        
        if not restock_needs:
            return "Tidak ada data stok untuk dianalisis"
        
        low_stock_threshold = 20  
        low_stock_products = [p for p in restock_needs if p['current_stock'] <= low_stock_threshold]
        
        if not low_stock_products:
            low_stock_products = sorted(restock_needs, key=lambda x: x['current_stock'])[:10]  
        
        sorted_products = sorted(low_stock_products, key=lambda x: x['current_stock'])
        
        response = "PRODUK DENGAN STOK PALING SEDIKIT\n\n"
        
        for i, product in enumerate(sorted_products[:10], 1):  # Top 10 saja
            response += f"{i}. {product['product_name'][:50]}...\n"
            response += f"   Stok: {product['current_stock']:.0f} unit\n"
            response += f"   Butuh restock: {product['restock_quantity']:.0f} unit\n"
            response += f"   Stok habis dalam: {product['stock_cover_days']:.1f} hari\n"
            response += f"   Penjualan/hari: {product['avg_daily_sales']:.1f} unit\n"
            response += f"   SKU: {product['sku']}\n\n"
        
        # Summary
        critical_products = [p for p in sorted_products if p['stock_cover_days'] < 5]
        
        response += f"SUMMARY:\n"
        response += f"   Total produk stok rendah: {len(sorted_products)} produk\n"
        response += f"   Produk kritis (stok < 5 hari): {len(critical_products)} produk\n"
        response += f"   Produk dengan stok tersedikit: {sorted_products[0]['product_name'][:30]}... ({sorted_products[0]['current_stock']:.0f} unit)\n"
        
        if critical_products:
            response += f"\nPERINGATAN:\n"
            response += f"   {len(critical_products)} produk akan habis dalam 5 hari!\n"
            response += f"   Prioritaskan: {critical_products[0]['product_name'][:30]}...\n"
        
        return response

    def execute_command(self, command_type, original_prompt):
        try:
            if command_type == 'restock_urgent':
                return self.analyze_restock_urgent()
            
            elif command_type == 'restock_all':
                return self.analyze_all_restock()
            
            elif command_type == 'top_products':
                return self.get_top_products(by='quantity')
            
            elif command_type == 'top_revenue':
                return self.get_top_products(by='revenue')
            
            elif command_type == 'prediction':
                days = self.extract_days_from_prompt(original_prompt)
                return self.predict_demand(days=days)
            
            elif command_type == 'trend':
                return self.get_sales_trend()
            
            elif command_type == 'summary':
                return self.get_comprehensive_summary()
            
            elif command_type == 'help':
                return self.show_natural_help()
            
            elif command_type == 'stock_most':
                return self.analyze_stock_most()
            
            elif command_type == 'stock_least':
                return self.analyze_stock_least()
            
        except Exception as e:
            return f"Error memproses permintaan: {str(e)}"
        
        def show_natural_help(self):
            """Tampilkan help yang natural"""
            help_text = """
            CARA BERINTERAKSI DENGAN SAYA:

            Anda bisa tanya dengan bahasa natural seperti:

            Tentang Stok & Restock:
            - Produk apa yang butuh restock segera?
            - Stok mana yang hampir habis?
            - Stok paling sedikit / paling dikit
            - Stok paling banyak / stok terbanyak
            - Butuh restock apa aja?
            - Analisis stok toko saya

            Tentang Penjualan:
            - Produk apa yang paling laris?
            - Produk dengan pendapatan tertinggi
            - Yang paling banyak duitnya

            Tentang Prediksi:
            - Prediksi penjualan 30 hari kedepan
            - Ramalan 2 minggu
            - Perkiraan permintaan bulan depan

            Tentang Analisis:
            - Bagaimana tren penjualan?
            - Penjualan naik atau turun?
            - Analisis perkembangan toko

            Laporan Lengkap:
            - Kasih laporan lengkap
            - Summary toko saya
            - Semua analisis

            Bantuan:
            - Bisa ngapain aja?
            - Fitur apa yang ada?
            - Help
            - Menu

            Contoh: 
            - Stok paling dikit
            - Produk yang stoknya banyak
            - Yang harus direstock segera
            """
            return help_text

    def extract_days_from_prompt(self, prompt):
        """Extract jumlah hari dari prompt natural"""
        import re
        numbers = re.findall(r'\b(\d+)\b', prompt)
        if numbers:
            return min(int(numbers[0]), 90)  # Maksimal 90 hari
        return 30  # Default 30 hari

    def handle_unknown_prompt(self, prompt):
        """Handle prompt yang tidak dikenali"""
        responses = [
            f"Saya tidak yakin memahami '{prompt}'. Coba tanyakan tentang: Restock produk, Produk terlaris, Prediksi penjualan, atau Tren penjualan",
            f"Maaf, saya belum paham maksud '{prompt}'. Bisa tanya tentang analisis stok atau penjualan?",
            f"Untuk '{prompt}', coba gunakan kata kunci: restock, produk terlaris, prediksi, atau tren",
        ]
        import random
        return random.choice(responses)

    def get_comprehensive_summary(self):
        """Ringkasan lengkap semua analisis"""
        if not self.is_ready:
            return "System not ready. Please wait..."
        
        response = "LAPORAN LENGKAP TOKO\n" + "="*50 + "\n\n"
        
        # 1. Restock urgent
        response += "RESTOCK URGENT\n"
        restock_needs = self.predictor.analyze_product_restock_needs(self.daily_sales_data)
        urgent_products = [p for p in restock_needs if p['urgency'] == 'HIGH']
        if urgent_products:
            response += f"   {len(urgent_products)} produk butuh restock segera\n"
            response += f"   Prioritas: {urgent_products[0]['product_name'][:30]}...\n"
        else:
            response += "   Semua produk aman\n"
        
        # 2. Top products
        response += "\nPRODUK TERLARIS\n"
        top_products = self.predictor.get_top_products_analysis(self.product_analysis)
        if top_products.get('top_by_quantity'):
            top_product = top_products['top_by_quantity'][0]
            response += f"   {top_product.get('Nama Produk', 'Unknown')[:30]}...\n"
            response += f"   Terjual: {top_product.get('Total_Quantity', 0):.0f} unit\n"
        
        # 3. Sales trend
        response += "\nTREN PENJUALAN\n"
        trend = self.predictor.generate_sales_trend_analysis(self.time_series_data)
        if trend:
            response += f"   {trend['trend_direction']}\n"
            response += f"   Rata-rata: {trend['recent_avg_daily_sales']:.1f} unit/hari\n"
        
        # 4. Prediction
        response += "\nPREDIKSI\n"
        predictions = self.predictor.predict_demand(days=7)
        if predictions:
            avg_demand = sum(p['predicted_demand'] for p in predictions) / len(predictions)
            response += f"   7 hari: {avg_demand:.1f} unit/hari\n"
        
        response += "\nREKOMENDASI\n"
        if urgent_products:
            response += f"   Restock {len(urgent_products)} produk urgent\n"
        if trend and trend['trend_direction'] == 'INCREASING':
            response += "   Penjualan naik, pertahankan!\n"
        elif trend and trend['trend_direction'] == 'DECREASING':
            response += "   Penjualan turun, evaluasi strategi\n"
        
        return response
    
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
            print(f"Auto-setup error: {str(e)}")
    
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
            return "System not ready. Please wait..."
        
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
        if not self.is_ready:
            return "System not ready. Please wait..."
        
        top_products = self.predictor.get_top_products_analysis(self.product_analysis)
        
        if by == 'revenue':
            products = top_products.get('top_by_revenue', [])
            title = "PRODUK TERLARIS BY REVENUE"
        else:
            products = top_products.get('top_by_quantity', [])
            title = "PRODUK TERLARIS BY QUANTITY"
        
        if not products:
            return f"{title}\nTidak ada data produk terlaris"
        
        response = f"{title}\n\n"
        
        for i, product in enumerate(products[:5], 1):
            product_name = product.get('Nama Produk', 'Unknown Product')
            quantity = product.get('Total_Quantity', 0)
            revenue = product.get('Total_Revenue', 0)
            transactions = product.get('Transaction_Count', 0)
            
            response += f"{i}. {product_name}\n"
            response += f"   Terjual: {quantity:.0f} unit\n"
            response += f"   Revenue: Rp {revenue:,.0f}\n"
            response += f"   Transaksi: {transactions}x\n\n"
        
        return response
    
    def predict_demand(self, days=30):
        if not self.is_ready:
            return "System not ready. Please wait..."
        
        results = self.predictor.predict_and_recommend(days=days)
        
        if not results:
            return "Prediction failed."
        
        response = f"PREDIKSI {days} HARI KE DEPAN\n\n"
        
        # Key metrics
        predictions = results.get('predictions', [])
        if predictions:
            avg_demand = sum(p['predicted_demand'] for p in predictions) / len(predictions)
            
            response += f"Key Metrics:\n"
            response += f"   Rata-rata demand: {avg_demand:.1f} unit/hari\n"
            response += f"   Total prediksi demand: {sum(p['predicted_demand'] for p in predictions):.0f} unit\n"
            
            # Check for urgent days
            recommendations = results.get('restock_recommendations', [])
            if recommendations:
                urgent_days = sum(1 for r in recommendations if r['urgency'] == 'HIGH')
                response += f"   Hari butuh restock urgent: {urgent_days} hari\n"
            
            response += f"\n"
            
            # 5 days preview
            response += f"5 Hari Pertama:\n"
            for i, pred in enumerate(predictions[:5], 1):
                urgency_info = ""
                if recommendations:
                    for rec in recommendations:
                        if rec['date'] == pred['date']:
                            urgency_info = f" {rec['urgency']}"
                            break
                response += f"   {i}. {pred['date']}{urgency_info} {pred['predicted_demand']:.0f} unit\n"
        else:
            response += "Tidak ada data prediksi\n"
        
        return response
    
    def get_sales_trend(self):
        """Analisis tren penjualan - tanpa emot"""
        if not self.is_ready:
            return "System not ready. Please wait..."
        
        trend_analysis = self.predictor.generate_sales_trend_analysis(self.time_series_data)
        
        response = "ANALISIS TREN PENJUALAN\n\n"
        
        response += f"Status Tren: {trend_analysis['trend_direction']}\n"
        response += f"Rata-rata Penjualan: {trend_analysis['recent_avg_daily_sales']:.1f} unit/hari\n"
        response += f"Volatilitas: {trend_analysis['volatility']:.1f}\n"
        response += f"Periode Analisis: {trend_analysis['analysis_period_days']} hari\n"
        
        # Interpretation
        if trend_analysis['trend_direction'] == 'INCREASING':
            response += "\nINTERPRETASI: Penjualan sedang naik! Pertahankan performa."
        elif trend_analysis['trend_direction'] == 'DECREASING':
            response += "\nINTERPRETASI: Penjualan sedang turun. Perlu evaluasi strategi."
        else:
            response += "\nINTERPRETASI: Penjualan stabil. Good job!"
        
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
                print("Failed to load data")
                return False
            
            print(" Auto-processing data...")
            self.time_series_data, self.daily_sales_data, self.product_analysis = \
                self.preprocessor.preprocess_data(self.dataset)
            
            if self.time_series_data.empty:
                print("Processing failed - no time series data")
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
                print("Auto-setup failed at model training")
                return False
                
        except Exception as e:
            print(f"Auto-setup error: {str(e)}")
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
        
        print("No dataset found. Using sample data.")
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
                print("Failed to load data")
                return
            
            print(" Auto-processing data...")
            self.time_series_data, self.daily_sales_data, self.product_analysis = \
                self.preprocessor.preprocess_data(self.dataset)
            
            if self.time_series_data.empty:
                print("Processing failed - no time series data")
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
                print("Auto-setup failed at model training")
                
        except Exception as e:
            print(f"Auto-setup error: {str(e)}")
    
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
            return "System not ready. Please wait..."
        
        top_products = self.predictor.get_top_products_analysis(self.product_analysis)
        
        if by == 'revenue':
            products = top_products.get('top_by_revenue', [])
            title = "💰 PRODUK TERLARIS BY REVENUE"
        else:
            products = top_products.get('top_by_quantity', [])
            title = "🏆 PRODUK TERLARIS BY QUANTITY"
        
        if not products:
            return f"{title}\n\nTidak ada data produk terlaris"
        
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
    """Main function dengan natural chat interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Shopee Auto-Magic Predictor')
    parser.add_argument('--dataset', '-d', type=str, help='Path to your dataset CSV file')
    args = parser.parse_args()
    
    print("SHOPEE AI ASSISTANT")
    print("Saya adalah asisten AI untuk analisis toko Shopee Anda!")
    print("Silakan tanyakan apa saja tentang stok, penjualan, dan prediksi toko Anda.")
    print("-" * 60)
    
    predictor = AutoMagicPredictor()
    success = predictor.setup(args.dataset)
    
    if not success:
        print("System gagal setup. Coba lagi.")
        return
    
    while True:
        try:
            prompt = input("\nApa yang bisa saya bantu : ").strip()
            
            if prompt.lower() in ['exit', 'quit', 'keluar', 'bye', 'selesai']:
                print("\nTerima kasih! Semoga toko Anda semakin sukses!")
                break
            
            if not prompt:
                continue
            
            response = predictor.process_natural_language(prompt)
            print(f"\nAsisten: {response}")
            
        except KeyboardInterrupt:
            print("\nSampai jumpa! Terima kasih telah menggunakan layanan kami.")
            break
        except Exception as e:
            print(f"\nMaaf, ada error: {str(e)}")
            print("Coba tanya lagi dengan pertanyaan yang berbeda.")

if __name__ == "__main__":
    main()