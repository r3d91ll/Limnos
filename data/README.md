# Data Directories

This directory contains data used by the Limnos project:

- `datasets/`: Datasets for testing and benchmarking
- `results/`: Results from experiments
- `sample_documents/`: Sample documents for testing
- `source_documents/`: Source documents for ingestion
- `databases/`: Generated vector databases and indexes

## Important Note

Large data files should not be committed to the git repository. The directory structure is preserved in git (using .gitkeep files), but the actual content is excluded via .gitignore.

## Usage

### Source Documents

Place your source documents in the `source_documents/` directory for ingestion. This can include PDFs, text files, Markdown files, etc.

```bash
# Example: Copy documents to the source_documents directory
cp /path/to/your/documents/*.pdf /home/todd/ML-Lab/Olympus/limnos/data/source_documents/
```

### Databases

The `databases/` directory is where generated vector databases and indexes will be stored. When running the ingestion pipeline, you can specify this directory as the output location.

```bash
# Example: Run ingestion with output to the databases directory
python limnos_cli.py ingest --directory data/source_documents --output data/databases
```
