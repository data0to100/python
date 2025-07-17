"""
Database Configuration for Data Analysis Toolkit
Centralized configuration for database connections.
"""

import os
from typing import Dict, Any

# Database connection configurations
DATABASE_CONFIGS = {
    'postgresql': {
        'engine': 'postgresql',
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', 5432),
        'database': os.getenv('DB_NAME', 'analytics'),
        'username': os.getenv('DB_USER', 'analyst'),
        'password': os.getenv('DB_PASSWORD', ''),
        'connection_string': 'postgresql://{username}:{password}@{host}:{port}/{database}'
    },
    
    'mysql': {
        'engine': 'mysql',
        'host': os.getenv('MYSQL_HOST', 'localhost'),
        'port': os.getenv('MYSQL_PORT', 3306),
        'database': os.getenv('MYSQL_DB', 'analytics'),
        'username': os.getenv('MYSQL_USER', 'root'),
        'password': os.getenv('MYSQL_PASSWORD', ''),
        'connection_string': 'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
    },
    
    'sqlite': {
        'engine': 'sqlite',
        'database': os.getenv('SQLITE_DB', 'data/analytics.db'),
        'connection_string': 'sqlite:///{database}'
    },
    
    'sqlserver': {
        'engine': 'mssql',
        'host': os.getenv('MSSQL_HOST', 'localhost'),
        'port': os.getenv('MSSQL_PORT', 1433),
        'database': os.getenv('MSSQL_DB', 'analytics'),
        'username': os.getenv('MSSQL_USER', 'sa'),
        'password': os.getenv('MSSQL_PASSWORD', ''),
        'connection_string': 'mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server'
    }
}

# API configurations
API_CONFIGS = {
    'default': {
        'base_url': os.getenv('API_BASE_URL', 'https://api.example.com'),
        'api_key': os.getenv('API_KEY', ''),
        'headers': {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {api_key}' if os.getenv('API_KEY') else None
        },
        'timeout': 30,
        'retry_attempts': 3
    }
}

# Cloud storage configurations
CLOUD_CONFIGS = {
    'aws_s3': {
        'access_key': os.getenv('AWS_ACCESS_KEY_ID', ''),
        'secret_key': os.getenv('AWS_SECRET_ACCESS_KEY', ''),
        'region': os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
        'bucket': os.getenv('S3_BUCKET', 'data-analytics')
    },
    
    'azure_blob': {
        'account_name': os.getenv('AZURE_STORAGE_ACCOUNT', ''),
        'account_key': os.getenv('AZURE_STORAGE_KEY', ''),
        'container': os.getenv('AZURE_CONTAINER', 'analytics')
    },
    
    'gcp_storage': {
        'project_id': os.getenv('GCP_PROJECT_ID', ''),
        'credentials_path': os.getenv('GOOGLE_APPLICATION_CREDENTIALS', ''),
        'bucket': os.getenv('GCS_BUCKET', 'data-analytics')
    }
}

def get_connection_string(db_type: str, custom_config: Dict[str, Any] = None) -> str:
    """
    Get formatted connection string for specified database type.
    
    Args:
        db_type: Database type ('postgresql', 'mysql', 'sqlite', 'sqlserver')
        custom_config: Custom configuration to override defaults
        
    Returns:
        Formatted connection string
    """
    if db_type not in DATABASE_CONFIGS:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    config = DATABASE_CONFIGS[db_type].copy()
    
    if custom_config:
        config.update(custom_config)
    
    return config['connection_string'].format(**config)

def get_api_config(api_name: str = 'default') -> Dict[str, Any]:
    """
    Get API configuration for specified API.
    
    Args:
        api_name: Name of the API configuration
        
    Returns:
        API configuration dictionary
    """
    if api_name not in API_CONFIGS:
        raise ValueError(f"Unknown API configuration: {api_name}")
    
    config = API_CONFIGS[api_name].copy()
    
    # Format headers with actual values
    if config['headers'].get('Authorization') and '{api_key}' in config['headers']['Authorization']:
        if config.get('api_key'):
            config['headers']['Authorization'] = config['headers']['Authorization'].format(
                api_key=config['api_key']
            )
        else:
            config['headers'].pop('Authorization')
    
    return config

def get_cloud_config(provider: str) -> Dict[str, Any]:
    """
    Get cloud storage configuration for specified provider.
    
    Args:
        provider: Cloud provider ('aws_s3', 'azure_blob', 'gcp_storage')
        
    Returns:
        Cloud configuration dictionary
    """
    if provider not in CLOUD_CONFIGS:
        raise ValueError(f"Unsupported cloud provider: {provider}")
    
    return CLOUD_CONFIGS[provider].copy()

# Sample queries for common analytics tasks
SAMPLE_QUERIES = {
    'customer_analysis': """
        SELECT 
            customer_id,
            COUNT(*) as total_orders,
            SUM(order_value) as total_spent,
            AVG(order_value) as avg_order_value,
            MAX(order_date) as last_order_date
        FROM orders 
        GROUP BY customer_id
        ORDER BY total_spent DESC
    """,
    
    'monthly_revenue': """
        SELECT 
            DATE_TRUNC('month', order_date) as month,
            COUNT(*) as total_orders,
            SUM(order_value) as revenue,
            AVG(order_value) as avg_order_value
        FROM orders 
        WHERE order_date >= '{start_date}'
        GROUP BY DATE_TRUNC('month', order_date)
        ORDER BY month
    """,
    
    'product_performance': """
        SELECT 
            product_id,
            product_name,
            SUM(quantity) as total_sold,
            SUM(quantity * price) as revenue,
            COUNT(DISTINCT order_id) as num_orders
        FROM order_items oi
        JOIN products p ON oi.product_id = p.id
        GROUP BY product_id, product_name
        ORDER BY revenue DESC
    """
}

def get_sample_query(query_name: str, **kwargs) -> str:
    """
    Get formatted sample query.
    
    Args:
        query_name: Name of the sample query
        **kwargs: Parameters to format into the query
        
    Returns:
        Formatted SQL query string
    """
    if query_name not in SAMPLE_QUERIES:
        raise ValueError(f"Unknown sample query: {query_name}")
    
    query = SAMPLE_QUERIES[query_name]
    
    if kwargs:
        query = query.format(**kwargs)
    
    return query