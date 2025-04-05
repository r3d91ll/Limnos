# Metadata Processing Architecture

## 1. Overview

This document outlines the metadata processing architecture for the Limnos RAG system, focusing on how universal document metadata is collected, transformed, and utilized by different RAG framework implementations. The architecture ensures proper separation between universal and framework-specific metadata while enabling each framework to work with optimized data structures.

## 2. Architectural Principles

1. **Separation of Concerns**: Clear separation between universal document collection and framework-specific processing
2. **Metadata Independence**: Universal metadata remains unmodified by framework-specific operations
3. **Transformation over Modification**: Frameworks transform universal metadata rather than modifying it
4. **Namespace Isolation**: Each framework maintains its data in a separate namespace
5. **Extension Points**: Well-defined interfaces for metadata transformation

## 3. Architecture Components

### 3.1 System Components

```
                     │ Universal Document Collector │
                     └──────────────┬──────────────┘
                                    │ Universal Metadata
                          ┌─────────┴─────────┐
                          │                   │
┌─────────────────────────▼───┐     ┌─────────▼───────────────┐
│ PathRAG Metadata Preprocessor│     │GraphRAG Metadata Preprocessor│
└─────────────────┬───────────┘     └───────────┬──────────────┘
                  │                              │
                  │ Optimized                    │ Optimized
                  │ Metadata                     │ Metadata
                  │                              │
┌─────────────────▼───────────┐     ┌───────────▼──────────────┐
│  PathRAG Processing Pipeline │     │GraphRAG Processing Pipeline│
└───────────────────────────┬─┘     └┬──────────────────────────┘
                            │        │
                            ▼        ▼
                     Framework-specific storage
                    (independent namespaces)
```

### 3.2 Component Interactions

1. **Universal Document Collector**:
   - Collects documents from various sources
   - Generates universal metadata (format, title, content, sections, etc.)
   - Stores original documents and universal metadata
   - Provides a consistent interface for accessing universal metadata

2. **Metadata Preprocessors**:
   - Framework-specific components
   - Transform universal metadata into formats optimized for each framework
   - Implement preprocessing hooks in their respective document collector integrations
   - Add framework-specific fields while maintaining universal metadata integrity

3. **Framework Processing Pipelines**:
   - Consume optimized metadata from their respective preprocessors
   - Perform framework-specific operations (e.g., path or graph construction)
   - Generate framework-specific artifacts
   - Store results in framework-specific namespaces

## 4. Metadata Workflow

### 4.1 Document Collection Phase

1. Document is submitted to the Universal Document Collector
2. Universal Document Collector processes the document:
   - Extracts text content
   - Identifies document structure (sections, headings)
   - Extracts basic metadata (title, authors, date)
   - Generates universal metadata JSON file
3. Original document and universal metadata are stored in the source documents directory

### 4.2 Preprocessing Phase

1. Framework-specific document collector integration retrieves the universal metadata
2. Metadata preprocessor transforms universal metadata:
   - Restructures content according to framework needs
   - Adds framework-specific fields
   - Optimizes data structures for framework operations
3. Transformed metadata is passed to the framework processing pipeline

### 4.3 Processing Phase

1. Framework processing pipeline processes the optimized metadata:
   - For PathRAG: Extracts entities, relationships, and constructs paths
   - For GraphRAG: Extracts entities, relationships, and constructs knowledge graphs
2. Framework-specific artifacts are generated:
   - Paths, vector embeddings, and search indexes for PathRAG
   - Knowledge graphs, graph indexes, and subgraphs for GraphRAG
3. Results are stored in framework-specific directories

## 5. Metadata Schema

### 5.1 Universal Metadata Schema

Universal metadata includes:
- Document ID and title
- Original file path and format
- Creation and modification timestamps
- Content extraction method
- Extracted full text content
- Section structure (headings, content blocks)
- Basic entities (authors, publication info)

### 5.2 Framework-Specific Extensions

#### 5.2.1 PathRAG Extensions
- Path-optimized content blocks
- Path extraction configuration
- Path-specific entity annotations
- Vector embedding parameters

#### 5.2.2 GraphRAG Extensions
- Graph-optimized content blocks
- Entity and relationship extraction parameters
- Graph construction configuration
- Subgraph caching parameters

## 6. Implementation Strategy

### 6.1 Universal Document Collector Enhancement

1. Update Universal Document Collector to expose a standardized metadata interface
2. Implement well-defined schema for universal metadata
3. Create extension points for framework-specific adaptations
4. Maintain strict separation of concerns

### 6.2 Metadata Preprocessors

1. Implement PathRAG Metadata Preprocessor:
   - Create preprocessing hook in PathRAG document collector integration
   - Optimize metadata for path extraction and processing
   - Maintain separation between universal and framework-specific metadata

2. Implement GraphRAG Metadata Preprocessor:
   - Leverage existing preprocessing hook mechanism in GraphRAG DocumentCollectorIntegration
   - Optimize metadata for entity extraction, relationship identification, and graph construction
   - Transform document sections into graph-friendly format

### 6.3 Testing and Validation

1. Create Metadata Transformation Testing Suite:
   - Verify correct transformation between Universal Document Collector and each RAG framework
   - Test edge cases and different document types
   - Ensure compatibility across frameworks

## 7. Benefits

This architecture provides several benefits:

1. **Clear boundaries** between universal and framework-specific data
2. **Optimized transformations** tailored to each framework's needs
3. **Consistent interface** for all frameworks to access universal metadata
4. **Testable components** with well-defined responsibilities
5. **Flexibility** for future framework additions
6. **Performance optimizations** specific to each framework's requirements
7. **Empirical comparison** of different document processing approaches
8. **Knowledge building** through parallel implementations

## 8. Future Extensions

1. **Universal Preprocessing Options**: Common preprocessing options that can be shared across frameworks
2. **Metadata Versioning**: Track changes to metadata over time
3. **Cross-Framework Integration**: Methods for frameworks to share and combine insights
4. **Hybrid Processing**: Capability to blend processing approaches for optimal results
