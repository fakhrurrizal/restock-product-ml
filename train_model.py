#!/usr/bin/env python3
"""
Script untuk training model saja
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ShopeeRestockApp

def train_model(data_source, model_type='SARIMA', save_model=True):
    """
    Training model dengan data source tertentu
    
    Args:
        data_source (str): URL atau path file dataset
        model_type (str): 'ARIMA' atau 'SARIMA'
        save_model (bool): Simpan model setelah training
    """
    app = ShopeeRestockApp()
    
    print(f" Training model {model_type}...")
    
    # Load data
    if not app.load_data(data_source):
        return None
    
    # Preprocess
    if not app.preprocess():
        return None
    
    # Train model
    if not app.train_model(model_type):
        return None
    
    # Save model
    if save_model:
        model_path = app.save_model()
        print(f"💾 Model disimpan: {model_path}")
        return model_path
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <data_source> [model_type]")
        print("Example: python train_model.py https://example.com/shopee_data.csv SARIMA")
        sys.exit(1)
    
    data_source = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'SARIMA'
    
    train_model(data_source, model_type)