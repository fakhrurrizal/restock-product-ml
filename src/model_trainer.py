import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')
from config import Config

class ModelTrainer:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.model_fit = None
        self.model_type = None
        self.is_trained = False
        self.training_history = {}
        self.time_series_data = None
        self.product_analysis = None
    
    def train_arima(self, train_data, order=None):
        """
        Train model ARIMA
        
        Args:
            train_data (pd.Series): Data training
            order (tuple): Parameter order ARIMA (p,d,q)
            
        Returns:
            bool: Status training
        """
        try:
            if order is None:
                order = self.config.ARIMA_ORDER
            
            print(f" Training ARIMA model dengan order {order}...")
            self.model = ARIMA(train_data, order=order)
            self.model_fit = self.model.fit()
            self.model_type = 'ARIMA'
            self.is_trained = True
            
            self.training_history = {
                'model_type': 'ARIMA',
                'order': order,
                'aic': self.model_fit.aic,
                'bic': self.model_fit.bic,
                'training_data_points': len(train_data)
            }
            
            print(" ARIMA model training berhasil")
            return True
            
        except Exception as e:
            print(f"Error training ARIMA: {str(e)}")
            return False
    
    def train_sarima(self, train_data, order=None, seasonal_order=None):
        """Train model SARIMA"""
        try:
            if order is None:
                order = self.config.SARIMA_ORDER
            if seasonal_order is None:
                seasonal_order = self.config.SARIMA_SEASONAL_ORDER
            
            print(f" Training SARIMA model dengan order {order} dan seasonal order {seasonal_order}...")
            self.model = SARIMAX(
                train_data, 
                order=order, 
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_fit = self.model.fit(disp=False)
            self.model_type = 'SARIMA'
            self.is_trained = True
            
            self.training_history = {
                'model_type': 'SARIMA',
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': self.model_fit.aic,
                'bic': self.model_fit.bic,
                'training_data_points': len(train_data)
            }
            
            # ⬇️ SIMPAN DATA TRAINING UNTUK REFERENSI
            self.time_series_data = train_data
            
            print(" SARIMA model training berhasil")
            return True
            
        except Exception as e:
            print(f"❌ Error training SARIMA: {str(e)}")
            return False
    
    def evaluate_model(self, test_data):
        """
        Evaluate model performance
        
        Args:
            test_data (pd.Series): Data testing
            
        Returns:
            dict: Metrics evaluasi
        """
        if not self.is_trained:
            return None
        
        try:
            # Forecast untuk periode testing
            forecast = self.model_fit.forecast(steps=len(test_data))
            
            # Calculate metrics
            mae = mean_absolute_error(test_data, forecast)
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
            
            evaluation_metrics = {
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'MAPE': round(mape, 2),
                'predictions': forecast.tolist(),
                'actual': test_data.tolist(),
                'test_periods': len(test_data)
            }
            
            print(f" Model Evaluation:")
            print(f"   MAE: {evaluation_metrics['MAE']}")
            print(f"   RMSE: {evaluation_metrics['RMSE']}")
            print(f"   MAPE: {evaluation_metrics['MAPE']}%")
            
            return evaluation_metrics
            
        except Exception as e:
            print(f"❌ Error evaluating model: {str(e)}")
            return None
    
    def save_model(self, filename=None):
        """
        Save trained model
        
        Args:
            filename (str): Nama file model
            
        Returns:
            str: Path model yang disimpan
        """
        if not self.is_trained:
            print("❌ Model belum ditraining")
            return None
        
        try:
            if filename is None:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{self.model_type}_model_{timestamp}.joblib"
            
            model_path = os.path.join(self.config.MODELS_DIR, filename)
            
            # Save model dan metadata
            model_data = {
                'model_fit': self.model_fit,
                'model_type': self.model_type,
                'training_history': self.training_history,
                'config': {
                    'model_type': self.model_type,
                    'order': self.training_history.get('order'),
                    'seasonal_order': self.training_history.get('seasonal_order')
                }
            }
            
            joblib.dump(model_data, model_path)
            print(f"💾 Model disimpan di: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return None
    
    def load_model(self, filepath):
        """
        Load trained model
        
        Args:
            filepath (str): Path menuju file model
            
        Returns:
            bool: Status loading
        """
        try:
            if not os.path.exists(filepath):
                print(f"❌ File model tidak ditemukan: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            self.model_fit = model_data['model_fit']
            self.model_type = model_data['model_type']
            self.training_history = model_data['training_history']
            self.is_trained = True
            
            print(f" Model {self.model_type} berhasil diload")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def get_model_summary(self):
        """Mendapatkan summary model"""
        if not self.is_trained:
            return "Model belum ditraining"
        
        return self.model_fit.summary()