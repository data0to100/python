#!/usr/bin/env python3
"""
ML Pipeline Runner Script

A simple script to run the ML pipeline with command line arguments.
This script demonstrates basic usage and provides a convenient entry point.

Usage:
    python scripts/run_pipeline.py --data data.csv --target target_column
    python scripts/run_pipeline.py --config config.yaml --data data.csv --target target_column
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_pipeline import MLPipeline, Config


def main():
    """Main function to run the ML pipeline"""
    parser = argparse.ArgumentParser(
        description="Run the ML Pipeline on your data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py --data data.csv --target sales
  python scripts/run_pipeline.py --config custom_config.yaml --data data.csv --target price
  python scripts/run_pipeline.py --data data.csv --target target --output results/
        """
    )
    
    parser.add_argument(
        "--data", 
        required=True,
        help="Path to the data file (CSV, Parquet, etc.)"
    )
    
    parser.add_argument(
        "--target",
        required=True, 
        help="Name of the target column"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration YAML file (optional)"
    )
    
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory for results and artifacts (default: outputs)"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)"
    )
    
    parser.add_argument(
        "--validation",
        choices=["holdout", "cv", "timeseries"],
        default="holdout",
        help="Validation strategy (default: holdout)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        return 1
    
    if args.config and not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        print("="*80)
        print("ML Pipeline Runner")
        print("="*80)
        print(f"Data: {args.data}")
        print(f"Target: {args.target}")
        print(f"Config: {args.config or 'default'}")
        print(f"Output: {args.output}")
        print(f"Validation: {args.validation}")
        print("="*80)
        
        # Load configuration
        if args.config:
            config = Config.from_yaml(args.config)
            print(f"Loaded configuration from: {args.config}")
        else:
            config = Config()
            print("Using default configuration")
        
        # Override output directory
        config.output_dir = args.output
        
        # Set logging level
        if args.verbose:
            config.logging.level = "DEBUG"
        
        # Initialize pipeline
        print("\nInitializing ML Pipeline...")
        pipeline = MLPipeline(config)
        
        # Run pipeline
        print("Running complete ML pipeline...")
        results = pipeline.run_complete_pipeline(
            data_source=args.data,
            target_column=args.target,
            test_size=args.test_size,
            validation_strategy=args.validation
        )
        
        # Print summary
        print("\n" + "="*80)
        print("PIPELINE RESULTS SUMMARY")
        print("="*80)
        
        summary = results["summary"]
        print(f"Data Shape: {summary['data_shape']}")
        print(f"Final Features: {summary['final_features']}")
        print(f"Best Model Score: {summary.get('best_model_score', 'N/A')}")
        print(f"Data Type: {summary.get('data_type', 'N/A')}")
        print(f"Recommendations: {summary['recommendations_count']}")
        
        # Print top insights
        insights = results["results"]["insights"]
        print(f"\nData Quality: {insights['data_insights']['data_quality']['total_rows']} rows")
        print(f"Missing Data: {insights['data_insights']['data_quality'].get('missing_data_percentage', 0):.1f}%")
        print(f"Best Model: {insights.get('model_insights', {}).get('best_model', {}).get('name', 'N/A')}")
        
        # Print recommendations
        recommendations = results["results"]["recommendations"]
        if recommendations:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        print(f"\nResults saved to: {args.output}")
        print("="*80)
        print("Pipeline completed successfully!")
        
        # Cleanup
        pipeline.cleanup()
        
        return 0
        
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        print("Please check your data and configuration.")
        return 1


if __name__ == "__main__":
    exit(main())