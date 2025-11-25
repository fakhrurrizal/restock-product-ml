#!/usr/bin/env python3
"""
Script untuk prediction dengan model yang sudah ditraining
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ShopeeRestockApp

def predict_with_model(model_path, data_source, days=30):
    """
    Prediction dengan model yang sudah ditraining
    
    Args:
        model_path (str): Path ke model yang sudah ditraining
        data_source (str): URL atau path file dataset baru
        days (int): Jumlah hari prediksi
    """
    app = ShopeeRestockApp()
    
    print(f"  Melakukan prediksi untuk {days} hari...")
    
    # Load model
    if not app.load_saved_model(model_path):
        return None
    
    # Load data baru untuk analisis
    if data_source and not app.load_data(data_source):
        return None
    
    if data_source and not app.preprocess():
        return None
    
    # Predict
    results = app.predict_and_recommend(days=days)
    
    if not results:
        return None
    
    # Output results as JSON
    output = {
        'predictions': results.get('predictions', []),
        'restock_recommendations': results.get('restock_recommendations', []),
        'top_products': results.get('top_products_analysis', {}).get('top_by_quantity', [])[:5],
        'trend_analysis': results.get('trend_analysis', {})
    }
    
    return output

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_path> <data_source> [days]")
        print("Example: python predict.py models/sarima_model.joblib https://example.com/new_data.csv 30")
        sys.exit(1)
    
    model_path = sys.argv[1]
    data_source = sys.argv[2]
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    results = predict_with_model(model_path, data_source, days)
    
    if results:
        print(json.dumps(results, indent=2))
    else:
        print("❌ Prediction gagal")