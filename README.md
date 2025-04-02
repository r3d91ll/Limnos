# Limnos - Testing Environment for Agentic RAG Systems

Limnos is a testing environment for Agentic RAG (Retrieval-Augmented Generation) systems. It provides a modular architecture for building and experimenting with different RAG components, including document processing, chunking, embedding, retrieval, and inference.

## Features

- **Modular Architecture**: Plug-and-play components for different parts of the RAG pipeline
- **Multiple Embedding Models**: Support for local models (vLLM with ModernBERT), sentence-transformers, and OpenAI
- **Flexible Storage**: Redis-based storage with support for documents, chunks, and embeddings
- **Command-Line Interface**: Easy-to-use CLI for ingesting documents and testing retrieval

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/limnos.git
cd limnos

# Install dependencies
pip install -r requirements.txt

# For vLLM support (optional)
bash scripts/setup_vllm_env.sh
source .venv/bin/activate
```

### Usage

```bash
# Ingest a document
python limnos_cli.py ingest --file /path/to/document.pdf

# Ingest a directory of documents
python limnos_cli.py ingest --directory /path/to/documents --recursive

# Use vLLM for embeddings
python limnos_cli.py --embedding-model vllm ingest --file /path/to/document.pdf
```

## Architecture

Limnos follows a modular architecture based on Actor-Network Theory principles:

- **Pipeline**: Core orchestration and component registration
- **Storage**: Document, chunk, and embedding storage backends
- **Models**: Embedding and inference model implementations
- **Ingest**: Document processing, chunking, and embedding generation
- **Retrieve**: Query processing and vector search
- **Infer**: Context assembly and response generation

## Configuration

Configuration is handled through YAML files in the `config` directory. You can override configuration values using command-line arguments.

## License

[MIT License](LICENSE)
