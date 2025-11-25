import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from config import Config

class RestockPredictor:
    def __init__(self, model_trainer=None):
        self.config = Config()
        self.model_trainer = model_trainer
        self.predictions = None
    
    def analyze_product_restock_needs(self, daily_sales_data, days_forecast=7, min_restock_threshold=0.3):
        """
        Analisis produk yang butuh restock - FIXED ESTIMASI STOK
        """
        if daily_sales_data.empty or 'SKU Induk' not in daily_sales_data.columns:
            return []
        
        print("   Estimating current stock using improved algorithm...")
        
        # Analisis 90 hari terakhir untuk pattern yang lebih baik
        recent_date = daily_sales_data['Tanggal'].max()
        cutoff_date = recent_date - pd.Timedelta(days=90)
        recent_sales = daily_sales_data[daily_sales_data['Tanggal'] >= cutoff_date]
        
        if recent_sales.empty:
            return []
        
        # Hitung metrics per produk
        product_metrics = recent_sales.groupby(['SKU Induk', 'Nama Produk']).agg({
            'Jumlah': ['sum', 'mean', 'max', 'min', 'count'],
            'Total Pembayaran': 'sum'
        }).round(2)
        
        product_metrics.columns = ['Total_Sold', 'Avg_Daily_Sales', 'Max_Daily_Sales', 'Min_Daily_Sales', 'Transaction_Count', 'Total_Revenue']
        product_metrics = product_metrics.reset_index()
        
        products_need_restock = []
        
        for _, product in product_metrics.iterrows():
            sku = product['SKU Induk']
            product_name = product['Nama Produk']
            avg_daily_sales = product['Avg_Daily_Sales']
            max_daily_sales = product['Max_Daily_Sales']
            transaction_count = product['Transaction_Count']
            
            if pd.isna(avg_daily_sales) or avg_daily_sales <= 0:
                continue
                
            # **PERBAIKAN 1: Prediksi yang lebih realistis**
            # Gunakan rata-rata penjualan, bukan maksimum
            predicted_demand_7days = avg_daily_sales * days_forecast * 1.1  # Buffer 10% saja
            
            # **PERBAIKAN 2: Estimasi stok yang lebih cerdas**
            # Analisis pattern penjualan 30 hari terakhir
            recent_30days = daily_sales_data[
                (daily_sales_data['SKU Induk'] == sku) & 
                (daily_sales_data['Tanggal'] >= recent_date - pd.Timedelta(days=30))
            ]
            
            if recent_30days.empty:
                continue
                
            # Hitung berbagai metrics untuk estimasi stok
            last_30days_sales = recent_30days['Jumlah'].sum()
            avg_monthly_sales = last_30days_sales
            sales_velocity = avg_daily_sales
            
            # **PERBAIKAN 3: Estimasi stok berdasarkan frekuensi penjualan**
            if transaction_count < 10:  # Produk slow-moving
                current_stock = max_daily_sales * 14  # Stok 2 minggu
            elif sales_velocity > 20:  # Produk fast-moving  
                current_stock = avg_monthly_sales * 0.7  # 70% dari penjualan bulanan
            else:  # Produk normal
                current_stock = avg_monthly_sales * 0.5  # 50% dari penjualan bulanan
            
            # **PERBAIKAN 4: Pastikan stok minimal ada untuk 3 hari**
            min_required_stock = avg_daily_sales * 3
            current_stock = max(current_stock, min_required_stock)
            
            # **PERBAIKAN 5: Safety stock yang wajar**
            safety_stock = avg_daily_sales * 5  # Safety stock untuk 5 hari
            recommended_stock = predicted_demand_7days + safety_stock
            restock_need = max(0, recommended_stock - current_stock)
            
            # **PERBAIKAN 6: Hitung days of stock yang realistis**
            days_of_stock = current_stock / avg_daily_sales if avg_daily_sales > 0 else 0
            
            # Tentukan urgency berdasarkan business rules yang lebih baik
            if days_of_stock < 5:
                urgency = 'HIGH'
                urgency_icon = ''
            elif days_of_stock < 10:
                urgency = 'MEDIUM' 
                urgency_icon = '🟡'
            else:
                urgency = 'LOW'
                urgency_icon = '🟢'
            
            # **PERBAIKAN 7: Filter yang lebih selektif**
            # Hanya tampilkan produk yang benar-benar butuh perhatian
            needs_attention = (
                (urgency == 'HIGH') or  # Stok < 5 hari
                (restock_need > avg_daily_sales * 2) or  # Butuh restock > 2x penjualan harian
                (days_of_stock < 7 and transaction_count > 5)  # Produk populer dengan stok rendah
            )
            
            if needs_attention and restock_need >= (avg_daily_sales * 0.5):
                products_need_restock.append({
                    'sku': sku,
                    'product_name': product_name,
                    'current_stock': round(current_stock, 2),
                    'predicted_demand_7days': round(predicted_demand_7days, 2),
                    'recommended_stock': round(recommended_stock, 2),
                    'restock_quantity': round(restock_need, 2),
                    'urgency': urgency,
                    'urgency_icon': urgency_icon,
                    'avg_daily_sales': round(avg_daily_sales, 2),
                    'max_daily_sales': round(max_daily_sales, 2),
                    'total_revenue': product['Total_Revenue'],
                    'stock_cover_days': round(days_of_stock, 1),
                    'transaction_count': transaction_count,
                    'sales_velocity': round(sales_velocity, 2)
                })
        
        # **PERBAIKAN 8: Sort by business priority**
        products_need_restock.sort(key=lambda x: (
            x['urgency'] != 'HIGH',  # HIGH urgency first
            -x['sales_velocity'],     # High sales velocity next
            -x['restock_quantity']    # Large restock quantity next
        ))
        
        print(f"   Found {len(products_need_restock)} products needing attention")
        return products_need_restock
    
    def predict_demand(self, days=None, confidence_level=0.95):
        """
        Prediksi permintaan untuk hari ke depan
        """
        if not self.model_trainer or not self.model_trainer.is_trained:
            print("❌ Model belum ditraining")
            return None
        
        if days is None:
            days = self.config.DEFAULT_PREDICTION_DAYS
        
        try:
            # Generate forecast
            forecast = self.model_trainer.model_fit.forecast(steps=days)
            confidence_int = self.model_trainer.model_fit.get_forecast(steps=days).conf_int(alpha=1-confidence_level)
            
            # Generate dates
            last_date = pd.Timestamp.now()
            dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
            
            predictions = []
            for i, date in enumerate(dates):
                predictions.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_demand': float(forecast.iloc[i]),
                    'lower_ci': float(confidence_int.iloc[i, 0]),
                    'upper_ci': float(confidence_int.iloc[i, 1]),
                    'confidence_level': confidence_level
                })
            
            self.predictions = predictions
            return predictions
            
        except Exception as e:
            print(f"❌ Error dalam prediksi: {str(e)}")
            return None
    def predict_and_recommend(self, days=30):
        """
        Prediksi dan generate rekomendasi - METHOD YANG DIPANGGIL OLEH auto_predictor.py
        """
        predictions = self.predict_demand(days=days)
        if not predictions:
            return None
        
        recommendations = self.generate_restock_recommendations()
        
        # Analisis produk terlaris
        top_products = self.get_top_products_analysis(self.model_trainer.product_analysis if hasattr(self.model_trainer, 'product_analysis') else {})
        
        # Analisis tren
        trend_analysis = self.generate_sales_trend_analysis(self.model_trainer.time_series_data if hasattr(self.model_trainer, 'time_series_data') else pd.DataFrame())
        
        result = {
            'predictions': predictions,
            'restock_recommendations': recommendations,
            'top_products_analysis': top_products,
            'trend_analysis': trend_analysis
        }
        
        return result
    
    def generate_restock_recommendations(self, current_stock_levels=None, lead_time=1, safety_stock_ratio=None):
        """
        Generate rekomendasi restock berdasarkan prediksi
        """
        if self.predictions is None:
            print("❌ Jalankan prediksi terlebih dahulu")
            return None
        
        if safety_stock_ratio is None:
            safety_stock_ratio = self.config.SAFETY_STOCK_RATIO
        
        recommendations = []
        
        for pred in self.predictions:
            predicted_demand = pred['predicted_demand']
            
            # Calculate safety stock
            safety_stock = predicted_demand * safety_stock_ratio
            
            # Calculate recommended stock level
            recommended_stock = predicted_demand + safety_stock
            
            # Determine urgency level
            if predicted_demand > recommended_stock * 0.8:
                urgency = 'HIGH'
                urgency_color = ''
            elif predicted_demand > recommended_stock * 0.5:
                urgency = 'MEDIUM'
                urgency_color = '🟡'
            else:
                urgency = 'LOW'
                urgency_color = '🟢'
            
            recommendation = {
                'date': pred['date'],
                'predicted_demand': round(predicted_demand, 2),
                'safety_stock': round(safety_stock, 2),
                'recommended_stock_level': round(recommended_stock, 2),
                'restock_quantity': max(0, round(recommended_stock - (current_stock_levels or {}).get('default', 0), 2)),
                'urgency': urgency,
                'urgency_color': urgency_color,
                'lead_time_days': lead_time
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def get_top_products_analysis(self, product_analysis, top_n=10):
        """
        Analisis produk terlaris
        """
        if not product_analysis or 'top_products' not in product_analysis:
            return {'top_by_quantity': [], 'top_by_revenue': []}
        
        top_products = product_analysis['top_products']
        
        # Sort by quantity
        top_by_quantity = top_products.nlargest(top_n, 'Total_Quantity')[
            ['SKU Induk', 'Nama Produk', 'Total_Quantity', 'Total_Revenue', 'Transaction_Count']
        ].to_dict('records')
        
        # Sort by revenue
        top_by_revenue = top_products.nlargest(top_n, 'Total_Revenue')[
            ['SKU Induk', 'Nama Produk', 'Total_Revenue', 'Total_Quantity', 'Transaction_Count']
        ].to_dict('records')
        
        return {
            'top_by_quantity': top_by_quantity,
            'top_by_revenue': top_by_revenue
        }
    
    def generate_sales_trend_analysis(self, time_series_data):
        """
        Analisis tren penjualan
        """
        if time_series_data.empty:
            return {}
        
        # Calculate trends
        recent_data = time_series_data.tail(30)  # 30 hari terakhir
        
        if len(recent_data) < 2:
            return {}
        
        # Simple linear trend
        x = np.arange(len(recent_data))
        y = recent_data['Total_Quantity'].values
        trend_coef = np.polyfit(x, y, 1)[0]
        
        # Determine trend direction
        if trend_coef > 0:
            trend_direction = 'INCREASING'
            trend_icon = ''
        elif trend_coef < 0:
            trend_direction = 'DECREASING'
            trend_icon = '📉'
        else:
            trend_direction = 'STABLE'
            trend_icon = '➡️'
        
        # Seasonal analysis (weekly pattern)
        if len(time_series_data) >= 7:
            weekly_avg = time_series_data['Total_Quantity'].tail(28).mean()  # 4 minggu
        else:
            weekly_avg = time_series_data['Total_Quantity'].mean()
        
        trend_analysis = {
            'trend_direction': trend_direction,
            'trend_icon': trend_icon,
            'trend_strength': abs(trend_coef),
            'recent_avg_daily_sales': recent_data['Total_Quantity'].mean(),
            'overall_avg_daily_sales': time_series_data['Total_Quantity'].mean(),
            'weekly_avg_sales': weekly_avg,
            'volatility': recent_data['Total_Quantity'].std(),
            'analysis_period_days': len(recent_data)
        }
        
        return trend_analysis
