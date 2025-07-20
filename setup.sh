#!/bin/bash

echo "🚀 Setting up Time Series Forecasting Dashboard..."
echo "=================================================="

# Add local bin to PATH
export PATH=$PATH:/home/ubuntu/.local/bin

# Install core dependencies
echo "📦 Installing core dependencies..."
pip3 install --break-system-packages --user pandas numpy streamlit plotly scikit-learn altair

# Generate sample data
echo "📊 Generating sample datasets..."
python3 sample_data.py

# Create directories for models and results
mkdir -p models
mkdir -p results

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 To run the forecasting dashboard:"
echo "   streamlit run forecasting_app_basic.py"
echo ""
echo "📂 Available sample datasets:"
echo "   - sample_sales_data.csv (Daily sales with seasonality)"
echo "   - sample_stock_data.csv (Stock price data)"
echo "   - sample_energy_data.csv (Hourly energy consumption)"
echo ""
echo "🌐 The app will be available at: http://localhost:8501"
echo ""
echo "📋 Features:"
echo "   ✓ CSV data upload and sample data"
echo "   ✓ Auto date format detection"
echo "   ✓ Missing value handling"
echo "   ✓ Multiple forecasting algorithms"
echo "   ✓ Interactive visualizations"
echo "   ✓ Performance metrics (MAE, RMSE, MAPE)"
echo "   ✓ Model comparison"
echo "   ✓ Results export (CSV)"
echo ""
echo "Happy forecasting! 📈"