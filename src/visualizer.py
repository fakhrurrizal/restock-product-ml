import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from config import Config

class DataVisualizer:
    def __init__(self):
        self.config = Config()
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_time_series(self, time_series_data, title="Time Series Data"):
        """Plot data time series"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Total Quantity
        axes[0, 0].plot(time_series_data.index, time_series_data['Total_Quantity'])
        axes[0, 0].set_title('Total Quantity per Hari')
        axes[0, 0].set_ylabel('Quantity')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Total Revenue
        axes[0, 1].plot(time_series_data.index, time_series_data['Total_Revenue'])
        axes[0, 1].set_title('Total Revenue per Hari')
        axes[0, 1].set_ylabel('Revenue')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Transaction Count
        axes[1, 0].plot(time_series_data.index, time_series_data['Transaction_Count'])
        axes[1, 0].set_title('Jumlah Transaksi per Hari')
        axes[1, 0].set_ylabel('Transactions')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Moving Average
        if len(time_series_data) > 7:
            ma_7 = time_series_data['Total_Quantity'].rolling(window=7).mean()
            axes[1, 1].plot(time_series_data.index, time_series_data['Total_Quantity'], alpha=0.5, label='Actual')
            axes[1, 1].plot(time_series_data.index, ma_7, label='7-Day MA', linewidth=2)
            axes[1, 1].set_title('Trend dengan Moving Average (7 hari)')
            axes[1, 1].set_ylabel('Quantity')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_predictions(self, predictions, actual_data=None, title="Demand Predictions"):
        """Plot hasil prediksi"""
        fig = go.Figure()
        
        dates = [pd.to_datetime(pred['date']) for pred in predictions]
        predicted_values = [pred['predicted_demand'] for pred in predictions]
        lower_ci = [pred['lower_ci'] for pred in predictions]
        upper_ci = [pred['upper_ci'] for pred in predictions]
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=dates, y=predicted_values,
            mode='lines+markers',
            name='Predicted Demand',
            line=dict(color='blue', width=3)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=upper_ci + lower_ci[::-1],
            fill='toself',
            fillcolor='rgba(0,100,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        # Add actual data if provided
        if actual_data is not None:
            actual_dates = [pd.to_datetime(act['date']) for act in actual_data]
            actual_values = [act['actual'] for act in actual_data]
            
            fig.add_trace(go.Scatter(
                x=actual_dates, y=actual_values,
                mode='lines+markers',
                name='Actual Demand',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Demand",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def plot_restock_recommendations(self, recommendations, title="Restock Recommendations"):
        """Plot rekomendasi restock"""
        if not recommendations:
            return None
        
        dates = [rec['date'] for rec in recommendations]
        predicted_demand = [rec['predicted_demand'] for rec in recommendations]
        recommended_stock = [rec['recommended_stock_level'] for rec in recommendations]
        urgency_colors = ['red' if rec['urgency'] == 'HIGH' else 'orange' if rec['urgency'] == 'MEDIUM' else 'green' for rec in recommendations]
        
        fig = go.Figure()
        
        # Add predicted demand
        fig.add_trace(go.Bar(
            x=dates, y=predicted_demand,
            name='Predicted Demand',
            marker_color='lightblue'
        ))
        
        # Add recommended stock
        fig.add_trace(go.Scatter(
            x=dates, y=recommended_stock,
            mode='lines+markers',
            name='Recommended Stock',
            line=dict(color='darkblue', width=3)
        ))
        
        # Add urgency markers
        for i, (date, stock, color) in enumerate(zip(dates, recommended_stock, urgency_colors)):
            fig.add_trace(go.Scatter(
                x=[date], y=[stock],
                mode='markers',
                marker=dict(size=12, color=color, symbol='diamond'),
                name=f'Urgency: {recommendations[i]["urgency"]}',
                showlegend=False
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Quantity",
            barmode='group',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_top_products(self, top_products_analysis, by='quantity'):
        """Plot produk terlaris"""
        if by == 'quantity':
            data = top_products_analysis['top_by_quantity']
            title = 'Top Products by Quantity Sold'
            y_column = 'Total_Quantity'
        else:
            data = top_products_analysis['top_by_revenue']
            title = 'Top Products by Revenue'
            y_column = 'Total_Revenue'
        
        product_names = [f"{p['Nama Produk'][:30]}..." if len(p['Nama Produk']) > 30 else p['Nama Produk'] for p in data]
        values = [p[y_column] for p in data]
        
        fig = px.bar(
            x=values,
            y=product_names,
            orientation='h',
            title=title,
            labels={'x': y_column.replace('_', ' ').title(), 'y': 'Product'}
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        return fig
    
    def create_dashboard(self, time_series_data, predictions, recommendations, top_products_analysis, trend_analysis):
        """Create comprehensive dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Sales Trend & Predictions',
                'Restock Recommendations', 
                'Top Products by Quantity',
                'Sales Statistics'
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ]
        )
        
        # Plot 1: Sales trend and predictions
        if not time_series_data.empty:
            recent_data = time_series_data.tail(60)  # 60 days recent data
            fig.add_trace(
                go.Scatter(x=recent_data.index, y=recent_data['Total_Quantity'], 
                          name='Actual Sales', line=dict(color='blue')),
                row=1, col=1
            )
        
        if predictions:
            pred_dates = [pd.to_datetime(p['date']) for p in predictions]
            pred_values = [p['predicted_demand'] for p in predictions]
            fig.add_trace(
                go.Scatter(x=pred_dates, y=pred_values, 
                          name='Predicted Sales', line=dict(color='red', dash='dash')),
                row=1, col=1
            )
        
        # Plot 2: Restock recommendations
        if recommendations:
            rec_dates = [rec['date'] for rec in recommendations]
            rec_stock = [rec['recommended_stock_level'] for rec in recommendations]
            fig.add_trace(
                go.Bar(x=rec_dates, y=rec_stock, name='Recommended Stock'),
                row=1, col=2
            )
        
        # Plot 3: Top products
        if top_products_analysis and top_products_analysis.get('top_by_quantity'):
            top_prods = top_products_analysis['top_by_quantity'][:5]  # Top 5
            prod_names = [p['Nama Produk'][:20] + '...' for p in top_prods]
            quantities = [p['Total_Quantity'] for p in top_prods]
            
            fig.add_trace(
                go.Bar(x=prod_names, y=quantities, name='Top Products'),
                row=2, col=1
            )
        
        # Plot 4: Trend indicator
        if trend_analysis:
            fig.add_trace(
                go.Indicator(
                    mode="delta",
                    value=trend_analysis['recent_avg_daily_sales'],
                    delta={'reference': trend_analysis['overall_avg_daily_sales'], 'relative': True},
                    title={"text": "Sales Trend"},
                    domain={'row': 1, 'col': 1}
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Shopee Restock Prediction Dashboard")
        return fig
    
    def save_plot(self, fig, filename, format='png'):
        """Save plot to file"""
        plots_dir = os.path.join(self.config.BASE_DIR, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        filepath = os.path.join(plots_dir, f"{filename}.{format}")
        
        if isinstance(fig, go.Figure):
            fig.write_image(filepath)
        else:
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        print(f" Plot disimpan: {filepath}")
        return filepath