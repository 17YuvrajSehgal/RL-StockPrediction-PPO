"""Setup configuration for swing_trading package."""

from setuptools import setup, find_packages

with open("swing_trading/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="swing_trading",
    version="0.1.0",
    author="Yuvraj Sehgal",
    author_email="ys19rk@brocku.ca",
    description="Professional RL environment for swing trading with continuous actions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RL-StockPrediction-PPO",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
            "ruff>=0.0.250",
        ],
        "rl": [
            "stable-baselines3>=2.0.0",
            "torch>=2.0.0",
        ],
    },
)
