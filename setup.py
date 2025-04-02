"""
Setup script for the Limnos - Testing Environment for Agentic RAG Systems project.
"""

from setuptools import setup, find_packages

setup(
    name="limnos",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "python-arango>=7.5.0",
        "neo4j>=5.8.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "openai>=1.1.1",
        "requests>=2.28.0",
        "rouge>=1.0.1",
        "nltk>=3.8.0",
        "bert-score>=0.3.13",
        "py-rouge>=1.1",
        "sacrebleu>=2.3.0",
        "fastapi>=0.104.1",
        "uvicorn>=0.23.2",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
    ],
    python_requires=">=3.10",
)
