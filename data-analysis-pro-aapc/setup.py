from setuptools import setup, find_packages

setup(
    name="data-analysis-pro-aapc",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.110",
        "uvicorn[standard]>=0.24",
        "pydantic>=2.7",
        "pandas>=2.2",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "plotly>=5.22",
        "python-dotenv>=1.0",
        "joblib>=1.4",
        "requests>=2.31",
        "SQLAlchemy>=2.0",
        "Jinja2>=3.1",
    ],
)