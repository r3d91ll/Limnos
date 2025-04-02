from setuptools import setup, find_packages

setup(
    name="limnos",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "redis>=4.5.1",
        "pypdf>=3.15.1",
        "python-docx>=0.8.11",
        "markdown>=3.4.3",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "sentence-transformers>=2.2.2",
    ],
    extras_require={
        "openai": ["openai>=1.3.0"],
        "vllm": ["vllm>=0.8.2", "torch>=2.0.0", "transformers>=4.30.0"],
    },
    entry_points={
        "console_scripts": [
            "limnos=limnos_cli:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Testing Environment for Agentic RAG Systems",
    keywords="rag, embeddings, retrieval, llm",
    url="https://github.com/yourusername/limnos",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
