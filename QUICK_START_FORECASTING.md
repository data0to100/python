# 🚀 Quick Start Guide - Time Series Forecasting Dashboard

## ⚡ 30-Second Setup

```bash
# 1. Run the setup script
bash setup.sh

# 2. Start the dashboard
streamlit run forecasting_app_basic.py
```

## 📱 Using the App

### Step 1: Load Data
- **Option A**: Upload your CSV file with date and target columns
- **Option B**: Use sample data (Sales, Stock, or Energy)

### Step 2: Configure Data
- Select date column and target variable
- Choose missing value handling method
- Set train/test split ratio

### Step 3: Select Models
Choose from:
- 📊 **Simple Moving Average** - Good for stable trends
- 📈 **Linear Trend** - Captures linear growth/decline  
- 🔄 **Exponential Smoothing** - Recent data weighted more
- 🗓️ **Seasonal Naive** - Uses seasonal patterns

### Step 4: Train & Analyze
- Click "🚀 Train Models" 
- Review interactive forecasts vs actual
- Compare model performance (MAE, RMSE, MAPE)
- Export results as CSV

## 📊 Sample Data Overview

### Sales Data (Daily)
- **Period**: 2020-2023 (4 years)
- **Features**: Trend + yearly/weekly seasonality
- **Use case**: Retail sales forecasting

### Stock Data (Business Days)
- **Period**: 2020-2023
- **Features**: Realistic price movements with volatility
- **Use case**: Financial time series

### Energy Data (Hourly)
- **Period**: 2022-2023 (2 years)
- **Features**: Daily and seasonal consumption patterns
- **Use case**: Utility demand forecasting

## 🎯 Best Practices

### For Business Users
1. Start with **sample data** to learn the interface
2. Use **Moving Average** for stable metrics
3. Try **Seasonal Naive** for cyclical data
4. Compare multiple models before deciding

### For Data Scientists
1. Upload your own CSV with proper date formatting
2. Experiment with different train/test splits
3. Fine-tune model parameters
4. Export results for further analysis

## 📋 Data Format Requirements

Your CSV should have:
```csv
date,target_variable,other_columns...
2023-01-01,100.5,extra_data
2023-01-02,102.1,extra_data
...
```

**Supported date formats:**
- `YYYY-MM-DD`
- `MM/DD/YYYY` 
- `DD/MM/YYYY`
- `YYYY-MM-DD HH:MM:SS`
- And many more!

## 🔧 Model Selection Guide

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| Moving Average | Stable data | Simple, reliable | No trend capture |
| Linear Trend | Growing/declining data | Captures trends | Assumes linearity |
| Exponential Smoothing | Recent pattern changes | Adaptive | Flat forecasts |
| Seasonal Naive | Cyclical patterns | Handles seasonality | Needs sufficient history |

## 📈 Metrics Explained

- **MAE (Mean Absolute Error)**: Average prediction error in original units
- **RMSE (Root Mean Square Error)**: Emphasizes larger errors
- **MAPE (Mean Absolute Percentage Error)**: Error as percentage (scale-independent)

**Lower values = Better performance**

## 🚨 Troubleshooting

### "Sample data not found"
```bash
python3 sample_data.py
```

### "Module not found" errors
```bash
pip3 install --break-system-packages pandas numpy streamlit plotly scikit-learn
```

### Date parsing issues
- Ensure consistent date format in your CSV
- Check for missing/invalid dates
- Use one of the supported formats listed above

## 💡 Pro Tips

1. **Model Comparison**: Always train multiple models and compare
2. **Parameter Tuning**: Adjust window sizes and periods for your data
3. **Validation**: Use appropriate train/test splits (70-80% train)
4. **Export Results**: Download forecasts for presentation/reporting
5. **Domain Knowledge**: Choose models based on your data characteristics

## 🔄 Next Steps

After mastering the basic app, consider:
- Implementing ARIMA/SARIMA models
- Adding Prophet for complex seasonality
- Building ensemble forecasts
- Creating automated retraining pipelines

---

**Happy Forecasting!** 📈✨

*Built with ❤️ using Streamlit, Plotly, and scikit-learn*