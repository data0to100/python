import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Model imports (basic implementations)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import base64
import io

# Configure page
st.set_page_config(
    page_title="Time Series Forecasting Dashboard",
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

# Simple forecasting models
class SimpleMovingAverage:
    def __init__(self, window=10):
        self.window = window
        self.model_name = f"Simple Moving Average (window={window})"
    
    def fit(self, train_data):
        self.train_data = train_data
        self.last_values = train_data[-self.window:]
    
    def forecast(self, steps):
        forecasts = []
        current_values = self.last_values.values.copy()
        
        for _ in range(steps):
            forecast_value = np.mean(current_values[-self.window:])
            forecasts.append(forecast_value)
            current_values = np.append(current_values, forecast_value)
        
        return np.array(forecasts)

class LinearTrendForecaster:
    def __init__(self):
        self.model = LinearRegression()
        self.model_name = "Linear Trend"
    
    def fit(self, train_data):
        X = np.arange(len(train_data)).reshape(-1, 1)
        y = train_data.values
        self.model.fit(X, y)
        self.train_length = len(train_data)
    
    def forecast(self, steps):
        X_future = np.arange(self.train_length, self.train_length + steps).reshape(-1, 1)
        forecasts = self.model.predict(X_future)
        return forecasts

class ExponentialSmoothingForecaster:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.model_name = f"Exponential Smoothing (α={alpha})"
    
    def fit(self, train_data):
        self.train_data = train_data.values
        # Calculate smoothed values
        self.smoothed = [self.train_data[0]]
        for i in range(1, len(self.train_data)):
            smoothed_value = self.alpha * self.train_data[i] + (1 - self.alpha) * self.smoothed[-1]
            self.smoothed.append(smoothed_value)
        self.last_smoothed = self.smoothed[-1]
    
    def forecast(self, steps):
        # For simple exponential smoothing, forecast is constant
        forecasts = [self.last_smoothed] * steps
        return np.array(forecasts)

class SeasonalNaiveForecaster:
    def __init__(self, seasonal_period=7):
        self.seasonal_period = seasonal_period
        self.model_name = f"Seasonal Naive (period={seasonal_period})"
    
    def fit(self, train_data):
        self.train_data = train_data.values
        self.seasonal_period = min(self.seasonal_period, len(train_data))
    
    def forecast(self, steps):
        forecasts = []
        for i in range(steps):
            # Use the value from seasonal_period steps back
            seasonal_index = len(self.train_data) - self.seasonal_period + (i % self.seasonal_period)
            forecasts.append(self.train_data[seasonal_index])
        return np.array(forecasts)

# Streamlit App
def main():
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📈 Time Series Forecasting Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("*Professional Time Series Forecasting with Multiple Algorithms*")
    
    # Sidebar
    st.sidebar.title("🎛️ Configuration Panel")
    
    # Data Input Section
    st.sidebar.header("📁 Data Input")
    
    # Option to use sample data or upload
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV", "Use Sample Data"]
    )
    
    if data_source == "Use Sample Data":
        sample_choice = st.sidebar.selectbox(
            "Select sample dataset:",
            ["Sales Data", "Stock Data", "Energy Data"]
        )
        
        if st.sidebar.button("Load Sample Data"):
            try:
                if sample_choice == "Sales Data":
                    data = pd.read_csv('sample_sales_data.csv')
                    date_column = 'date'
                    target_column = 'sales'
                elif sample_choice == "Stock Data":
                    data = pd.read_csv('sample_stock_data.csv')
                    date_column = 'date'
                    target_column = 'close_price'
                elif sample_choice == "Energy Data":
                    data = pd.read_csv('sample_energy_data.csv')
                    date_column = 'datetime'
                    target_column = 'energy_consumption'
                
                st.session_state.data = data
                st.session_state.date_column = date_column
                st.session_state.target_column = target_column
                st.sidebar.success(f"✅ {sample_choice} loaded successfully!")
                
            except FileNotFoundError:
                st.sidebar.error("❌ Sample data files not found. Please generate them first by running `python sample_data.py`")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading sample data: {str(e)}")
    
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with datetime and target variable columns"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                
                st.sidebar.success(f"✅ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
                
                # Column selection
                date_column = st.sidebar.selectbox("Select Date Column", data.columns)
                target_column = st.sidebar.selectbox("Select Target Variable", data.columns)
                
                st.session_state.date_column = date_column
                st.session_state.target_column = target_column
                
            except Exception as e:
                st.sidebar.error(f"❌ Error loading data: {str(e)}")
                return
    
    # Process data if available
    if st.session_state.data is not None and hasattr(st.session_state, 'date_column'):
        data = st.session_state.data
        date_column = st.session_state.date_column
        target_column = st.session_state.target_column
        
        # Handle missing values
        missing_method = st.sidebar.selectbox(
            "Handle Missing Values",
            ['interpolate', 'forward_fill', 'backward_fill', 'drop']
        )
        
        try:
            # Process data
            processed_data = data[[date_column, target_column]].copy()
            
            # Convert date column
            processed_data[date_column] = pd.to_datetime(processed_data[date_column])
            
            # Handle missing values
            processed_data = handle_missing_values(processed_data, missing_method)
            
            # Set date as index
            processed_data.set_index(date_column, inplace=True)
            processed_data.sort_index(inplace=True)
            
            # Store processed data
            st.session_state.processed_data = processed_data
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            return
    
    # Main content
    if hasattr(st.session_state, 'processed_data'):
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
            hovermode='x unified',
            height=500
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
            ['Simple Moving Average', 'Linear Trend', 'Exponential Smoothing', 'Seasonal Naive'],
            default=['Simple Moving Average', 'Linear Trend']
        )
        
        # Model parameters
        if 'Simple Moving Average' in selected_models:
            st.sidebar.subheader("Moving Average Parameters")
            ma_window = st.sidebar.slider("Window Size", 3, 30, 10)
        
        if 'Exponential Smoothing' in selected_models:
            st.sidebar.subheader("Exponential Smoothing Parameters")
            alpha = st.sidebar.slider("Smoothing Parameter (α)", 0.1, 0.9, 0.3)
        
        if 'Seasonal Naive' in selected_models:
            st.sidebar.subheader("Seasonal Naive Parameters")
            seasonal_period = st.sidebar.slider("Seasonal Period", 2, 365, 7)
        
        # Training section
        if st.sidebar.button("🚀 Train Models", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_models = len(selected_models)
            st.session_state.forecasts.clear()
            st.session_state.model_metrics.clear()
            
            for i, model_name in enumerate(selected_models):
                status_text.text(f"Training {model_name}...")
                
                try:
                    if model_name == 'Simple Moving Average':
                        model = SimpleMovingAverage(window=ma_window)
                    elif model_name == 'Linear Trend':
                        model = LinearTrendForecaster()
                    elif model_name == 'Exponential Smoothing':
                        model = ExponentialSmoothingForecaster(alpha=alpha)
                    elif model_name == 'Seasonal Naive':
                        model = SeasonalNaiveForecaster(seasonal_period=seasonal_period)
                    
                    # Fit model and forecast
                    model.fit(train_data)
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
            
            fig = go.Figure()
            
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
                    barmode='group',
                    height=400
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
        st.info("👋 **Welcome to the Time Series Forecasting Dashboard!**")
        st.markdown("""
        ### 🚀 Getting Started
        1. **Choose data source** - Upload CSV or use sample data
        2. **Select columns** - Choose date and target variable
        3. **Configure preprocessing** - Handle missing values
        4. **Select models** - Choose from multiple forecasting algorithms
        5. **Train models** - Click the train button to start forecasting
        6. **Analyze results** - Review forecasts and metrics
        7. **Export data** - Download results for further analysis
        
        ### 📋 Available Models
        - **Simple Moving Average**: Uses recent values to predict future
        - **Linear Trend**: Fits a linear trend to the data
        - **Exponential Smoothing**: Weighted average giving more importance to recent data
        - **Seasonal Naive**: Uses seasonal patterns for forecasting
        
        ### 💡 Tips
        - Use the sample data to get started quickly
        - Try multiple models and compare their performance
        - Adjust parameters to optimize forecasting accuracy
        - Export results for further analysis or reporting
        """)

if __name__ == "__main__":
    main()