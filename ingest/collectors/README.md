# Limnos Document Collectors

This directory contains document collector components for the Limnos RAG system. These collectors are responsible for acquiring documents from various sources, processing them to extract content and metadata, and storing them in a standardized format.

## Universal Document Collector

The `UniversalDocumentCollector` is the primary component for collecting documents in Limnos. It handles:

1. Processing documents to extract content and metadata
2. Storing original documents alongside universal metadata
3. Organizing documents in a consistent directory structure

### Directory Structure

Documents are stored in the following structure:

```
/limnos/data/source_documents/
  ├── <document_id>/
  │   ├── original_document.pdf  # Original document
  │   └── original_document.json # Universal metadata
  ├── <document_id>/
  │   ├── another_document.pdf
  │   └── another_document.json
  └── ...
```

### Universal Metadata Schema

All documents use a standardized metadata schema defined in `metadata_schema.py`. This ensures consistency across different document types and enables interoperability between RAG frameworks.

Key metadata fields include:

- **Identification**: `doc_id`, `doc_type`
- **Basic Information**: `title`, `authors`, `date_created`, `date_modified`
- **Content Information**: `language`, `content_length`, `summary`
- **Source Information**: `source_path`, `source_url`, `storage_path`
- **Semantic Information**: `keywords`, `categories`
- **Document-Specific Fields**: 
  - Academic papers: `abstract`, `doi`, `journal`, `references`
  - Documentation: `version`, `api_version`, `framework`

### Usage

The collector can be used programmatically or via the command-line script:

#### Command-line Usage

```bash
# Collect a single file
python limnos/scripts/collect_documents.py collect-file --file_path /path/to/document.pdf

# Collect all files in a directory
python limnos/scripts/collect_documents.py collect-dir --dir_path /path/to/directory --recursive

# List all collected documents
python limnos/scripts/collect_documents.py list

# Get details of a specific document
python limnos/scripts/collect_documents.py get --doc_id <document_id>
```

#### Programmatic Usage

```python
from limnos.ingest.collectors.universal_collector import UniversalDocumentCollector

# Initialize the collector
collector = UniversalDocumentCollector()
collector.initialize({
    'processor_type': 'academic',  # Use academic paper processor
    'source_dir': '/path/to/source_documents'
})

# Collect a single file
document, paths = collector.collect_file('/path/to/paper.pdf')

# Collect all files in a directory
results = collector.collect_directory('/path/to/papers', recursive=True)

# Get a document by ID
document, paths = collector.get_document('document_id')
```

## Specialized Document Processors

The Universal Document Collector uses specialized document processors to handle different document types:

- **AcademicPaperProcessor**: Extracts rich metadata from academic papers (PDF)
- **BasicDocumentProcessor**: Handles common document formats (text, markdown, PDF, docx)

## Future Extensions

The collector architecture is designed to be extensible for future document sources:

1. **Web Scraping**: Collecting documentation from websites
2. **GitHub Repositories**: Importing code and documentation from GitHub
3. **Academic APIs**: Fetching papers from arXiv, Semantic Scholar, etc.
