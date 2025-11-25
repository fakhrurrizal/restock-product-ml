#!/usr/bin/env python3
"""
Main Application untuk Shopee Restock Prediction System
"""

import os
import sys
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.predictor import RestockPredictor
from src.visualizer import DataVisualizer
from src.config import Config

class ShopeeRestockApp:
    def __init__(self):
        self.config = Config()
        self.config.ensure_directories()
        
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        self.predictor = RestockPredictor(self.model_trainer)
        self.visualizer = DataVisualizer()
        
        self.dataset = None
        self.time_series_data = None
        self.daily_sales_data = None
        self.product_analysis = None
    
    def load_data(self, source):
        """
        Load data dari URL atau file lokal
        
        Args:
            source (str): URL atau path file lokal
        """
        print("📥 Memuat data...")
        
        if source.startswith(('http://', 'https://')):
            self.dataset = self.data_loader.load_from_url(source)
        else:
            self.dataset = self.data_loader.load_from_file(source)
        
        if self.dataset is None:
            print("❌ Gagal memuat data")
            return False
        
        # Validasi dataset
        validation = self.data_loader.validate_dataset(self.dataset)
        if not validation['is_valid']:
            print(f"❌ Dataset tidak valid. Kolom yang hilang: {validation['missing_columns']}")
            return False
        
        # Tampilkan info dataset
        info = self.data_loader.get_dataset_info(self.dataset)
        print(f" Info Dataset:")
        print(f"   Shape: {info['shape']}")
        print(f"   Periode: {info['date_range']['start']} hingga {info['date_range']['end']}")
        print(f"   Total Penjualan: {info['total_sales']} unit")
        print(f"   Total Produk: {info['total_products']}")
        
        return True
    
    def preprocess(self):
        """Preprocessing data"""
        if self.dataset is None:
            print("❌ Data belum dimuat")
            return False
        
        self.time_series_data, self.daily_sales_data, self.product_analysis = \
            self.preprocessor.preprocess_data(self.dataset)
        
        if self.time_series_data.empty:
            print("❌ Gagal memproses data time series")
            return False
        
        print(f" Preprocessing selesai. Data time series: {len(self.time_series_data)} hari")
        return True
    
    def train_model(self, model_type='SARIMA'):
        """
        Training model
        
        Args:
            model_type (str): 'ARIMA' atau 'SARIMA'
        """
        if self.time_series_data is None:
            print("❌ Data belum diproses")
            return False
        
        # Siapkan data training
        train_data, test_data = self.preprocessor.prepare_training_data(
            self.time_series_data, 
            test_size=self.config.TEST_SIZE
        )
        
        # Training model
        if model_type.upper() == 'ARIMA':
            success = self.model_trainer.train_arima(train_data)
        else:
            success = self.model_trainer.train_sarima(train_data)
        
        if not success:
            return False
        
        # Evaluate model
        evaluation = self.model_trainer.evaluate_model(test_data)
        
        if evaluation:
            print(" Training dan evaluasi selesai")
            return True
        else:
            print("❌ Evaluasi model gagal")
            return False
    
    def predict_and_recommend(self, days=30):
        """
        Prediksi dan generate rekomendasi
        
        Args:
            days (int): Jumlah hari prediksi
        """
        if not self.model_trainer.is_trained:
            print("❌ Model belum ditraining")
            return None
        
        # Prediksi
        predictions = self.predictor.predict_demand(days=days)
        if not predictions:
            return None
        
        # Rekomendasi restock
        recommendations = self.predictor.generate_restock_recommendations()
        
        # Analisis produk terlaris
        top_products = self.predictor.get_top_products_analysis(self.product_analysis)
        
        # Analisis tren
        trend_analysis = self.predictor.generate_sales_trend_analysis(self.time_series_data)
        
        result = {
            'predictions': predictions,
            'restock_recommendations': recommendations,
            'top_products_analysis': top_products,
            'trend_analysis': trend_analysis,
            'model_info': self.model_trainer.training_history
        }
        
        return result
    
    def visualize_results(self, results, save_plots=False):
        """
        Visualisasi hasil
        
        Args:
            results (dict): Hasil prediksi dan analisis
            save_plots (bool): Simpan plot ke file
        """
        if not results:
            print("❌ Tidak ada hasil untuk divisualisasikan")
            return
        
        # Plot time series
        if not self.time_series_data.empty:
            ts_plot = self.visualizer.plot_time_series(self.time_series_data, "Data Penjualan Historis")
            if save_plots:
                self.visualizer.save_plot(ts_plot, "time_series")
        
        # Plot predictions
        if results.get('predictions'):
            pred_plot = self.visualizer.plot_predictions(results['predictions'])
            if save_plots:
                self.visualizer.save_plot(pred_plot, "predictions")
        
        # Plot restock recommendations
        if results.get('restock_recommendations'):
            restock_plot = self.visualizer.plot_restock_recommendations(results['restock_recommendations'])
            if save_plots:
                self.visualizer.save_plot(restock_plot, "restock_recommendations")
        
        # Plot top products
        if results.get('top_products_analysis'):
            top_prods_plot = self.visualizer.plot_top_products(results['top_products_analysis'])
            if save_plots:
                self.visualizer.save_plot(top_prods_plot, "top_products")
        
        # Dashboard comprehensive
        dashboard = self.visualizer.create_dashboard(
            self.time_series_data,
            results.get('predictions', []),
            results.get('restock_recommendations', []),
            results.get('top_products_analysis', {}),
            results.get('trend_analysis', {})
        )
        if save_plots:
            self.visualizer.save_plot(dashboard, "dashboard")
        
        print(" Visualisasi selesai")
    
    def save_model(self, filename=None):
        """Save trained model"""
        return self.model_trainer.save_model(filename)
    
    def load_saved_model(self, model_path):
        """Load model yang sudah disimpan"""
        return self.model_trainer.load_model(model_path)

def main():
    """Main function untuk testing"""
    app = ShopeeRestockApp()
    
    print("=" * 60)
    print("🛍️  SHOPEE RESTOCK PREDICTION SYSTEM")
    print("=" * 60)
    
    # Contoh penggunaan
    # Ganti dengan URL atau path file dataset Anda
    data_source = input("Masukkan URL dataset atau path file lokal: ").strip()
    
    if not data_source:
        print("❌ Sumber data tidak boleh kosong")
        return
    
    # Step 1: Load data
    if not app.load_data(data_source):
        return
    
    # Step 2: Preprocess
    if not app.preprocess():
        return
    
    # Step 3: Train model
    model_type = input("Pilih model (ARIMA/SARIMA) [SARIMA]: ").strip() or "SARIMA"
    if not app.train_model(model_type):
        return
    
    # Step 4: Predict
    try:
        days = int(input("Jumlah hari prediksi [30]: ").strip() or "30")
    except:
        days = 30
    
    results = app.predict_and_recommend(days=days)
    
    if not results:
        print("❌ Prediksi gagal")
        return
    
    # Step 5: Visualize
    save_viz = input("Simpan visualisasi? (y/n) [n]: ").strip().lower() == 'y'
    app.visualize_results(results, save_plots=save_viz)
    
    # Step 6: Save model
    save_model = input("Simpan model? (y/n) [y]: ").strip().lower() != 'n'
    if save_model:
        model_path = app.save_model()
        if model_path:
            print(f"💾 Model disimpan: {model_path}")
    
    # Tampilkan hasil
    print("\n" + "=" * 60)
    print(" HASIL PREDIKSI & REKOMENDASI")
    print("=" * 60)
    
    # Tampilkan rekomendasi restock
    if results.get('restock_recommendations'):
        print("\n REKOMENDASI RESTOCK (5 hari pertama):")
        print("-" * 80)
        for i, rec in enumerate(results['restock_recommendations'][:5]):
            print(f"{i+1}. {rec['date']}:")
            print(f"   Prediksi Demand: {rec['predicted_demand']:.0f} unit")
            print(f"   Rekomendasi Stok: {rec['recommended_stock_level']:.0f} unit")
            print(f"   Quantity Restock: {rec['restock_quantity']:.0f} unit")
            print(f"   Urgency: {rec['urgency_color']} {rec['urgency']}")
            print()
    
    # Tampilkan produk terlaris
    if results.get('top_products_analysis'):
        top_prods = results['top_products_analysis'].get('top_by_quantity', [])[:3]
        if top_prods:
            print("🏆 PRODUK TERLARIS:")
            print("-" * 80)
            for i, prod in enumerate(top_prods):
                print(f"{i+1}. {prod['Nama Produk']}")
                print(f"   Terjual: {prod['Total_Quantity']:.0f} unit")
                print(f"   Revenue: Rp {prod['Total_Revenue']:,.0f}")
                print()
    
    # Tampilkan analisis tren
    if results.get('trend_analysis'):
        trend = results['trend_analysis']
        print(" ANALISIS TREN:")
        print("-" * 80)
        print(f"Tren: {trend['trend_icon']} {trend['trend_direction']}")
        print(f"Rata-rata Penjualan Harian: {trend['recent_avg_daily_sales']:.1f} unit")
        print(f"Volatilitas: {trend['volatility']:.1f}")

if __name__ == "__main__":
    main()