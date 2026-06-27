from setuptools import setup, find_packages

setup(
    name="option-pricer",
    version="1.0.0",
    description="Quantitative Options Pricer & Risk Analyzer",
    author="ymistercap",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.14.0",
        "QuantLib>=1.31",
        "streamlit>=1.28.0",
        "yfinance>=0.2.28",
    ],
)
