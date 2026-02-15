"""Package setup for Financial Risk Analyzer."""

from setuptools import setup, find_packages

setup(
    name="financial-risk-analyzer",
    version="1.0.0",
    author="Taofik Bishi",
    description="Portfolio risk analysis tool with Monte Carlo simulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/taofikbishi/financial-risk-analyzer",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "risk-analyzer=risk_analyzer.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
