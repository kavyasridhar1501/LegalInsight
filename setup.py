"""
Setup script for LegalInsight
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="legalinsight-selfrag",
    version="1.0.0",
    author="LegalInsight Team",
    description="AI-Powered Legal Contract Analysis with Self-RAG + EigenScore",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kavyasridhar1501/LegalInsight",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Industry",
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
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "datasets>=2.14.0",
        "faiss-cpu>=1.7.4",
        "langchain>=0.1.0",
        "langgraph>=0.2.0",
        "requests>=2.28.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "jupyter>=1.0.0", "notebook>=7.0.0"],
        "local_model": ["llama-cpp-python>=0.2.0"],
    },
)
