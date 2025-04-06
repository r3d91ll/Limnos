# Limnos RAG Implementations

This directory contains the various RAG framework implementations for the Limnos project.

## Architecture

The Limnos architecture has been refactored to use Pydantic models for metadata handling and ensure clear separation between:

1. **Universal document processing**: Common to all RAG frameworks
2. **Framework-specific processing**: Unique to each implementation

For comprehensive documentation, see the [architecture documentation](/docs/architecture/rag_framework_metadata.md).

## Available Frameworks

### GraphRAG

Graph-based RAG that extracts entities and relationships from documents to build knowledge graphs for retrieval.

Key components:
- Document collector integration
- Entity extraction
- Graph construction

### PathRAG

Path-based RAG that identifies meaningful paths through document sections for more coherent retrieval.

Key components:
- Document collector integration
- Path extraction
- Vector storage

## Data Storage

**IMPORTANT**: Framework-specific data is stored separately from code:

```
/limnos/data/implementations/graphrag/  - GraphRAG data
/limnos/data/implementations/pathrag/   - PathRAG data
```

This separation allows for using high-performance storage for data while keeping code on system drives.

## Integration Testing

Integration tests for both frameworks can be found in:
```
/tests/integration/test_rag_integrations.py
```
