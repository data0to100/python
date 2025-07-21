#!/usr/bin/env python3
"""
Test script for the Enterprise AI/ML Platform

This script tests the core functionality of the ML platform to ensure
all components are working correctly.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all ML platform modules can be imported."""
    print("Testing imports...")
    
    try:
        from ml_platform.config import MLConfig
        from ml_platform.data_loader import DataLoader
        from ml_platform.models import XGBoostModel, LightGBMModel, AutoML
        from ml_platform.visualization import VisualizationEngine, InsightsGenerator
        from ml_platform.exceptions import MLPlatformError
        from ml_platform.logger import get_logger
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("Testing configuration...")
    
    try:
        from ml_platform.config import MLConfig
        
        config = MLConfig()
        print(f"✅ Configuration loaded successfully")
        print(f"   - Debug: {config.debug}")
        print(f"   - Log Level: {config.log_level}")
        print(f"   - Max Workers: {config.max_workers}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_data_loader():
    """Test data loading functionality."""
    print("Testing data loader...")
    
    try:
        from ml_platform.data_loader import DataLoader
        from ml_platform.config import MLConfig
        
        config = MLConfig()
        loader = DataLoader(config)
        
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Save to CSV for testing
        test_file = "test_data.csv"
        sample_data.to_csv(test_file, index=False)
        
        # Test loading
        df = loader.load_data(test_file, source_type="csv")
        print(f"✅ Data loaded successfully: {df.shape}")
        
        # Test data info
        info = loader.get_data_info(df)
        print(f"   - Columns: {len(info['columns'])}")
        print(f"   - Missing values: {sum(info['missing_values'].values())}")
        
        # Clean up
        os.remove(test_file)
        
        return True
    except Exception as e:
        print(f"❌ Data loader error: {e}")
        return False

def test_models():
    """Test model functionality."""
    print("Testing models...")
    
    try:
        from ml_platform.models import XGBoostModel, AutoML
        from ml_platform.config import MLConfig
        from sklearn.datasets import load_iris
        
        # Load sample data
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target)
        
        # Test XGBoost
        config = MLConfig()
        model = XGBoostModel(config, task_type="classification")
        
        # Quick training with small dataset
        metrics = model.fit(X, y, test_size=0.3)
        print(f"✅ XGBoost training successful")
        print(f"   - Test Accuracy: {metrics.get('test_accuracy', 0):.4f}")
        
        # Test predictions
        predictions = model.predict(X.head(5))
        print(f"   - Predictions shape: {predictions.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False

def test_visualization():
    """Test visualization functionality."""
    print("Testing visualization...")
    
    try:
        from ml_platform.visualization import VisualizationEngine, InsightsGenerator
        from ml_platform.config import MLConfig
        
        config = MLConfig()
        viz_engine = VisualizationEngine(config)
        insights_gen = InsightsGenerator(config)
        
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'numeric1': np.random.normal(0, 1, 100),
            'numeric2': np.random.normal(5, 2, 100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Test EDA dashboard
        dashboard = viz_engine.create_eda_dashboard(sample_data)
        print(f"✅ EDA dashboard created successfully")
        print(f"   - Overview keys: {list(dashboard['overview'].keys())}")
        print(f"   - Insights count: {len(dashboard['insights'])}")
        
        # Test insights generation
        summary = insights_gen.generate_statistical_summary(sample_data)
        print(f"✅ Statistical summary generated")
        print(f"   - Summary keys: {list(summary.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def test_logging():
    """Test logging functionality."""
    print("Testing logging...")
    
    try:
        from ml_platform.logger import get_logger, log_performance
        
        logger = get_logger()
        logger.info("Test log message")
        
        # Test performance logging
        log_performance("test_operation", 0.1, {"test": "metadata"})
        
        print(f"✅ Logging system working")
        return True
    except Exception as e:
        print(f"❌ Logging error: {e}")
        return False

def main():
    """Run all tests."""
    print("🤖 Enterprise AI/ML Platform Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_logging,
        test_data_loader,
        test_visualization,
        test_models,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The ML platform is ready to use.")
        print("\nTo run the platform:")
        print("1. python main.py --interface web")
        print("2. Choose option 2 for Enterprise AI/ML Platform")
        print("3. Open http://localhost:8501 in your browser")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())