import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Model imports
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pickle
import io
import base64

# Configure page
st.set_page_config(
    page_title="Advanced Time Series Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e1e1e1;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stAlert > div {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}

# Utility functions
def detect_date_format(date_series):
    """Auto-detect date format from a pandas Series"""
    sample_dates = date_series.dropna().head(10)
    common_formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
        '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%d-%m-%Y', '%d.%m.%Y', '%Y.%m.%d'
    ]
    
    for fmt in common_formats:
        try:
            pd.to_datetime(sample_dates, format=fmt)
            return fmt
        except:
            continue
    return None

def handle_missing_values(df, method='interpolate'):
    """Handle missing values in the dataset"""
    if method == 'interpolate':
        return df.interpolate(method='linear')
    elif method == 'forward_fill':
        return df.fillna(method='ffill')
    elif method == 'backward_fill':
        return df.fillna(method='bfill')
    elif method == 'drop':
        return df.dropna()
    return df

def calculate_metrics(y_true, y_pred):
    """Calculate forecasting metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def create_sequences(data, n_steps):
    """Create sequences for LSTM model"""
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Model classes
class ARIMAForecaster:
    def __init__(self, order=None, seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
    
    def fit(self, train_data, auto_arima=True):
        if auto_arima:
            self.fitted_model = pm.auto_arima(
                train_data, 
                seasonal=True, 
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
        else:
            if self.seasonal_order:
                self.model = SARIMAX(train_data, order=self.order, seasonal_order=self.seasonal_order)
            else:
                self.model = ARIMA(train_data, order=self.order)
            self.fitted_model = self.model.fit()
    
    def forecast(self, steps, confidence_level=0.95):
        forecast = self.fitted_model.forecast(steps=steps)
        conf_int = self.fitted_model.get_forecast(steps=steps).conf_int(alpha=1-confidence_level)
        return forecast, conf_int

class XGBoostForecaster:
    def __init__(self, lag_features=10):
        self.lag_features = lag_features
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = MinMaxScaler()
    
    def create_features(self, data):
        df = pd.DataFrame({'value': data})
        for i in range(1, self.lag_features + 1):
            df[f'lag_{i}'] = df['value'].shift(i)
        return df.dropna()
    
    def fit(self, train_data):
        feature_df = self.create_features(train_data)
        X = feature_df.drop('value', axis=1)
        y = feature_df['value']
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.last_values = train_data[-self.lag_features:].values
    
    def forecast(self, steps):
        forecasts = []
        current_values = self.last_values.copy()
        
        for _ in range(steps):
            X_pred = current_values[-self.lag_features:].reshape(1, -1)
            X_pred_scaled = self.scaler.transform(X_pred)
            pred = self.model.predict(X_pred_scaled)[0]
            forecasts.append(pred)
            current_values = np.append(current_values, pred)
        
        return np.array(forecasts)

class LSTMForecaster:
    def __init__(self, sequence_length=10, units=50):
        self.sequence_length = sequence_length
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler()
    
    def fit(self, train_data, epochs=50, batch_size=32):
        # Scale data
        train_scaled = self.scaler.fit_transform(train_data.values.reshape(-1, 1))
        
        # Create sequences
        X, y = create_sequences(train_scaled, self.sequence_length)
        
        # Build model
        self.model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(self.units, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        # Train model
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Store last sequence for forecasting
        self.last_sequence = train_scaled[-self.sequence_length:]
    
    def forecast(self, steps):
        forecasts = []
        current_sequence = self.last_sequence.copy()
        
        for _ in range(steps):
            pred_scaled = self.model.predict(current_sequence.reshape(1, self.sequence_length, 1), verbose=0)
            pred = self.scaler.inverse_transform(pred_scaled)[0, 0]
            forecasts.append(pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], pred_scaled[0, 0])
        
        return np.array(forecasts)

# Streamlit App
def main():
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📈 Advanced Time Series Forecasting Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("*Built for Data Scientists - Supporting ARIMA, SARIMA, Prophet, LSTM, and XGBoost models*")
    
    # Sidebar
    st.sidebar.title("🎛️ Configuration Panel")
    
    # Data Input Section
    st.sidebar.header("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file with datetime and target variable columns"
    )
    
    if uploaded_file is not None:
        # Load data
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            st.sidebar.success(f"✅ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Column selection
            date_column = st.sidebar.selectbox("Select Date Column", data.columns)
            target_column = st.sidebar.selectbox("Select Target Variable", data.columns)
            
            # Handle missing values
            missing_method = st.sidebar.selectbox(
                "Handle Missing Values",
                ['interpolate', 'forward_fill', 'backward_fill', 'drop']
            )
            
            if date_column and target_column:
                # Process data
                processed_data = data[[date_column, target_column]].copy()
                
                # Auto-detect and convert date format
                try:
                    processed_data[date_column] = pd.to_datetime(processed_data[date_column])
                except:
                    st.error("❌ Could not parse date column. Please check the format.")
                    return
                
                # Handle missing values
                processed_data = handle_missing_values(processed_data, missing_method)
                
                # Set date as index
                processed_data.set_index(date_column, inplace=True)
                processed_data.sort_index(inplace=True)
                
                # Store processed data
                st.session_state.processed_data = processed_data
                st.session_state.target_column = target_column
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return
    
    # Main content
    if st.session_state.data is not None and 'processed_data' in st.session_state:
        data = st.session_state.processed_data
        target_col = st.session_state.target_column
        
        # Display data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", len(data))
        with col2:
            st.metric("📅 Date Range", f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("📈 Mean Value", f"{data[target_col].mean():.2f}")
        with col4:
            st.metric("📉 Std Dev", f"{data[target_col].std():.2f}")
        
        # Data visualization
        st.subheader("📊 Data Overview")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[target_col],
            mode='lines',
            name='Historical Data',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            title="Time Series Data",
            xaxis_title="Date",
            yaxis_title=target_col,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model Configuration
        st.sidebar.header("🤖 Model Configuration")
        
        # Train-test split
        test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20) / 100
        split_index = int(len(data) * (1 - test_size))
        
        train_data = data.iloc[:split_index][target_col]
        test_data = data.iloc[split_index:][target_col]
        
        st.sidebar.info(f"Train: {len(train_data)} samples | Test: {len(test_data)} samples")
        
        # Model selection
        selected_models = st.sidebar.multiselect(
            "Select Models to Train",
            ['Auto-ARIMA', 'SARIMA', 'XGBoost', 'LSTM'],
            default=['Auto-ARIMA']
        )
        
        # Model parameters
        if 'SARIMA' in selected_models:
            st.sidebar.subheader("SARIMA Parameters")
            p = st.sidebar.slider("AR order (p)", 0, 5, 1)
            d = st.sidebar.slider("Differencing (d)", 0, 2, 1)
            q = st.sidebar.slider("MA order (q)", 0, 5, 1)
            seasonal_period = st.sidebar.slider("Seasonal Period", 1, 52, 12)
        
        if 'XGBoost' in selected_models:
            st.sidebar.subheader("XGBoost Parameters")
            lag_features = st.sidebar.slider("Lag Features", 5, 20, 10)
        
        if 'LSTM' in selected_models:
            st.sidebar.subheader("LSTM Parameters")
            sequence_length = st.sidebar.slider("Sequence Length", 5, 30, 10)
            lstm_units = st.sidebar.slider("LSTM Units", 32, 128, 50)
            epochs = st.sidebar.slider("Training Epochs", 20, 100, 50)
        
        # Training section
        if st.sidebar.button("🚀 Train Models", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_models = len(selected_models)
            
            for i, model_name in enumerate(selected_models):
                status_text.text(f"Training {model_name}...")
                
                try:
                    if model_name == 'Auto-ARIMA':
                        model = ARIMAForecaster()
                        model.fit(train_data, auto_arima=True)
                        forecast_values, conf_int = model.forecast(len(test_data))
                        
                        st.session_state.trained_models[model_name] = model
                        st.session_state.forecasts[model_name] = {
                            'values': forecast_values,
                            'conf_int': conf_int,
                            'dates': test_data.index
                        }
                    
                    elif model_name == 'SARIMA':
                        model = ARIMAForecaster(
                            order=(p, d, q),
                            seasonal_order=(1, 1, 1, seasonal_period)
                        )
                        model.fit(train_data, auto_arima=False)
                        forecast_values, conf_int = model.forecast(len(test_data))
                        
                        st.session_state.trained_models[model_name] = model
                        st.session_state.forecasts[model_name] = {
                            'values': forecast_values,
                            'conf_int': conf_int,
                            'dates': test_data.index
                        }
                    
                    elif model_name == 'XGBoost':
                        model = XGBoostForecaster(lag_features=lag_features)
                        model.fit(train_data)
                        forecast_values = model.forecast(len(test_data))
                        
                        st.session_state.trained_models[model_name] = model
                        st.session_state.forecasts[model_name] = {
                            'values': forecast_values,
                            'dates': test_data.index
                        }
                    
                    elif model_name == 'LSTM':
                        model = LSTMForecaster(sequence_length=sequence_length, units=lstm_units)
                        model.fit(train_data, epochs=epochs)
                        forecast_values = model.forecast(len(test_data))
                        
                        st.session_state.trained_models[model_name] = model
                        st.session_state.forecasts[model_name] = {
                            'values': forecast_values,
                            'dates': test_data.index
                        }
                    
                    # Calculate metrics
                    metrics = calculate_metrics(test_data.values, forecast_values)
                    st.session_state.model_metrics[model_name] = metrics
                    
                except Exception as e:
                    st.error(f"❌ Error training {model_name}: {str(e)}")
                
                progress_bar.progress((i + 1) / total_models)
            
            status_text.text("✅ Training completed!")
            st.success("🎉 All models trained successfully!")
        
        # Results section
        if st.session_state.forecasts:
            st.header("📈 Forecasting Results")
            
            # Visualization
            st.subheader("📊 Forecast vs Actual")
            
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=["Forecasting Results"]
            )
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=train_data.index,
                y=train_data.values,
                mode='lines',
                name='Training Data',
                line=dict(color='blue', width=2)
            ))
            
            # Add actual test data
            fig.add_trace(go.Scatter(
                x=test_data.index,
                y=test_data.values,
                mode='lines',
                name='Actual',
                line=dict(color='green', width=3)
            ))
            
            # Add forecasts
            colors = ['red', 'orange', 'purple', 'brown']
            for i, (model_name, forecast_data) in enumerate(st.session_state.forecasts.items()):
                fig.add_trace(go.Scatter(
                    x=forecast_data['dates'],
                    y=forecast_data['values'],
                    mode='lines',
                    name=f'{model_name} Forecast',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
                
                # Add confidence intervals for models that support it
                if 'conf_int' in forecast_data:
                    conf_int = forecast_data['conf_int']
                    fig.add_trace(go.Scatter(
                        x=forecast_data['dates'],
                        y=conf_int.iloc[:, 0],
                        mode='lines',
                        line=dict(color=colors[i % len(colors)], width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast_data['dates'],
                        y=conf_int.iloc[:, 1],
                        mode='lines',
                        line=dict(color=colors[i % len(colors)], width=0),
                        fill='tonexty',
                        fillcolor=f'rgba({colors[i % len(colors)]}, 0.2)',
                        name=f'{model_name} CI',
                        hoverinfo='skip'
                    ))
            
            fig.update_layout(
                title="Time Series Forecasting Results",
                xaxis_title="Date",
                yaxis_title=target_col,
                hovermode='x unified',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics comparison
            st.subheader("📊 Model Performance Metrics")
            
            if st.session_state.model_metrics:
                metrics_df = pd.DataFrame(st.session_state.model_metrics).T
                metrics_df = metrics_df.round(4)
                
                # Display metrics table
                st.dataframe(metrics_df, use_container_width=True)
                
                # Metrics visualization
                fig_metrics = go.Figure()
                
                for metric in ['MAE', 'RMSE', 'MAPE']:
                    fig_metrics.add_trace(go.Bar(
                        name=metric,
                        x=metrics_df.index,
                        y=metrics_df[metric],
                        text=metrics_df[metric].round(4),
                        textposition='auto'
                    ))
                
                fig_metrics.update_layout(
                    title="Model Performance Comparison",
                    xaxis_title="Models",
                    yaxis_title="Metric Value",
                    barmode='group'
                )
                
                st.plotly_chart(fig_metrics, use_container_width=True)
                
                # Best model recommendation
                best_model_mae = metrics_df['MAE'].idxmin()
                best_model_rmse = metrics_df['RMSE'].idxmin()
                best_model_mape = metrics_df['MAPE'].idxmin()
                
                st.success(f"🏆 **Best Models**: MAE: {best_model_mae}, RMSE: {best_model_rmse}, MAPE: {best_model_mape}")
            
            # Export functionality
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Export Forecasts as CSV"):
                    # Combine all forecasts
                    export_df = pd.DataFrame({'Date': test_data.index, 'Actual': test_data.values})
                    
                    for model_name, forecast_data in st.session_state.forecasts.items():
                        export_df[f'{model_name}_Forecast'] = forecast_data['values']
                    
                    csv = export_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="forecasts.csv">📥 Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            with col2:
                if st.button("📊 Export Metrics as CSV"):
                    if st.session_state.model_metrics:
                        metrics_df = pd.DataFrame(st.session_state.model_metrics).T
                        csv = metrics_df.to_csv()
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="model_metrics.csv">📥 Download Metrics CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
    
    else:
        # Welcome screen
        st.info("👋 **Welcome to the Advanced Time Series Forecasting Dashboard!**")
        st.markdown("""
        ### 🚀 Getting Started
        1. **Upload your CSV file** using the sidebar
        2. **Select date and target columns** from your data
        3. **Choose preprocessing options** for missing values
        4. **Configure and train models** (ARIMA, SARIMA, XGBoost, LSTM)
        5. **Analyze results** with interactive visualizations
        6. **Export forecasts** for further analysis
        
        ### 📋 Supported Features
        - **📁 Data Input**: CSV upload with auto-detection of date formats
        - **🔧 Preprocessing**: Multiple options for handling missing values
        - **🤖 Models**: Auto-ARIMA, SARIMA, XGBoost, LSTM with hyperparameter tuning
        - **📊 Visualization**: Interactive plots with confidence intervals
        - **📈 Metrics**: MAE, RMSE, MAPE evaluation with model comparison
        - **💾 Export**: Download forecasts and metrics as CSV
        
        ### 💡 Tips for Best Results
        - Ensure your date column is properly formatted
        - Choose appropriate train/test split ratios
        - Consider seasonal patterns when selecting models
        - Use multiple models for comparison and ensemble approaches
        """)

if __name__ == "__main__":
    main()