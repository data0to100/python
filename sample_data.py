import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data():
    """Generate sample time series data for testing the forecasting app"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate date range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate synthetic time series with trend, seasonality, and noise
    n_days = len(dates)
    
    # Trend component
    trend = np.linspace(100, 200, n_days)
    
    # Seasonal component (yearly and weekly)
    yearly_season = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    weekly_season = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    # Random noise
    noise = np.random.normal(0, 15, n_days)
    
    # Combine components
    sales = trend + yearly_season + weekly_season + noise
    
    # Add some random missing values (5% of data)
    missing_indices = np.random.choice(n_days, size=int(0.05 * n_days), replace=False)
    sales[missing_indices] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + np.random.normal(0, 3, n_days),
        'promotion': np.random.choice([0, 1], size=n_days, p=[0.8, 0.2])
    })
    
    # Save to CSV
    df.to_csv('sample_sales_data.csv', index=False)
    print("✅ Sample data generated: sample_sales_data.csv")
    print(f"📊 Data shape: {df.shape}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"📈 Sales range: {df['sales'].min():.2f} to {df['sales'].max():.2f}")
    print(f"❌ Missing values: {df['sales'].isna().sum()} ({df['sales'].isna().sum()/len(df)*100:.1f}%)")
    
    return df

def generate_stock_data():
    """Generate sample stock price data"""
    np.random.seed(123)
    
    # Generate date range (business days only)
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.bdate_range(start=start_date, end=end_date)
    
    n_days = len(dates)
    
    # Generate stock price using geometric Brownian motion
    initial_price = 100
    mu = 0.0008  # daily return
    sigma = 0.02  # volatility
    
    returns = np.random.normal(mu, sigma, n_days)
    price_ratios = np.exp(returns)
    prices = initial_price * np.cumprod(price_ratios)
    
    # Add volume data
    volume = np.random.lognormal(mean=12, sigma=0.5, size=n_days)
    
    df = pd.DataFrame({
        'date': dates,
        'close_price': prices,
        'volume': volume.astype(int),
        'high_price': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'low_price': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    })
    
    df.to_csv('sample_stock_data.csv', index=False)
    print("✅ Sample stock data generated: sample_stock_data.csv")
    print(f"📊 Data shape: {df.shape}")
    
    return df

def generate_energy_consumption_data():
    """Generate sample energy consumption data"""
    np.random.seed(456)
    
    # Hourly data for 2 years
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    n_hours = len(dates)
    
    # Base consumption with daily and seasonal patterns
    hour_of_day = dates.hour
    day_of_year = dates.dayofyear
    
    # Daily pattern (higher during day, lower at night)
    daily_pattern = 50 + 30 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    
    # Seasonal pattern (higher in summer for AC, winter for heating)
    seasonal_pattern = 20 * (np.sin(2 * np.pi * day_of_year / 365) + 
                            0.5 * np.sin(4 * np.pi * day_of_year / 365))
    
    # Weekend effect
    is_weekend = dates.weekday >= 5
    weekend_effect = -10 * is_weekend
    
    # Random noise
    noise = np.random.normal(0, 8, n_hours)
    
    consumption = daily_pattern + seasonal_pattern + weekend_effect + noise
    consumption = np.maximum(consumption, 10)  # Ensure positive values
    
    df = pd.DataFrame({
        'datetime': dates,
        'energy_consumption': consumption,
        'temperature': 15 + 15 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5, n_hours),
        'is_weekend': is_weekend.astype(int)
    })
    
    df.to_csv('sample_energy_data.csv', index=False)
    print("✅ Sample energy data generated: sample_energy_data.csv")
    print(f"📊 Data shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    print("🔄 Generating sample datasets for forecasting app testing...")
    print()
    
    # Generate different types of sample data
    sales_df = generate_sample_data()
    print()
    
    stock_df = generate_stock_data()
    print()
    
    energy_df = generate_energy_consumption_data()
    print()
    
    print("🎉 All sample datasets generated successfully!")
    print("\nYou can now use these files to test the forecasting app:")
    print("1. sample_sales_data.csv - Daily sales data with trend and seasonality")
    print("2. sample_stock_data.csv - Stock price data (business days)")
    print("3. sample_energy_data.csv - Hourly energy consumption data")