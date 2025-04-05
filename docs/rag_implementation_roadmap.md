# Limnos RAG Implementation Roadmap

This document outlines the roadmap for implementing multiple RAG frameworks within Limnos and establishing a comprehensive testing environment.

## Phase 1: Complete PathRAG Implementation

### Core PathRAG Components
- [x] Entity Extractor
- [x] Relationship Extractor
- [x] Path Constructor
- [x] Path Structures
- [x] Path Vector Store
- [x] Path Storage Manager
- [x] Query Processor (Hermes prototype)
- [x] Document Ranker
- [x] Universal Document Collector Integration

## Phase 2: Implement GraphRAG

### Core GraphRAG Components
- [ ] Graph entity extraction and relationship identification
- [ ] Graph construction and representation using NetworkX
- [ ] Redis integration for graph caching and persistence
- [ ] Graph traversal and search algorithms
- [ ] Semantic embedding and similarity calculation
- [ ] Subgraph extraction and pruning
- [ ] Document retrieval and ranking

### GraphRAG Integration
- [ ] Connect GraphRAG to Hermes prototype
- [ ] Create GraphRAG-specific adapter for Hermes
- [ ] Setup GraphRAG-specific data storage using Limnos architecture
- [ ] Implement GraphRAG query processor and parser
- [ ] Build GraphRAG organizer for content refinement
- [ ] Develop retrieval connector with Redis caching layer
- [ ] Create document collector integration for GraphRAG

## Phase 3: Prepare for Multi-Framework Support

### Extract Hermes to Limnos Level
- [ ] Move query processing interfaces to `/limnos/hermes/`
- [ ] Create framework registration system
- [ ] Implement query routing mechanisms
- [ ] Add framework selection capabilities (sequential/parallel execution)
- [ ] Develop unified result format

### Refactor PathRAG
- [ ] Update PathRAG to use the Hermes interfaces
- [ ] Create PathRAG-specific adapter for Hermes
- [ ] Ensure backward compatibility

### Refactor GraphRAG
- [ ] Update GraphRAG to use the Hermes interfaces
- [ ] Create GraphRAG-specific adapter for Hermes
- [ ] Ensure backward compatibility

## Phase 4: Testing Environment

### Framework-Agnostic Testing Infrastructure
- [ ] Design standard evaluation metrics
- [ ] Build tools to compare results across frameworks
- [ ] Implement result visualization
- [ ] Create performance dashboards

### Benchmarking System
- [ ] Establish standard benchmark datasets
- [ ] Develop automated testing pipelines
- [ ] Create framework performance leaderboards
- [ ] Implement continuous performance monitoring

## Implementation Guidelines

### Architectural Principles
1. **Separation of Concerns**: Each RAG implementation should be independent and self-contained
2. **Universal Metadata**: Maintain clear separation between universal metadata and framework-specific data
3. **Framework Independence**: Framework-specific components should be isolated in their own namespaces
4. **Common Interfaces**: All frameworks should implement common interfaces defined by Hermes

### Directory Structure
```
/limnos/
├── data/
│   ├── source_documents/        # Original documents and universal metadata
│   └── implementations/
│       ├── pathrag/             # PathRAG-specific data
│       └── graphrag/            # GraphRAG-specific data
├── hermes/                      # Query processing and framework coordination
├── implementations/
│   ├── pathrag/                 # PathRAG implementation
│   │   ├── core/                # Core components
│   │   └── adapters/            # Hermes adapters
│   └── graphrag/                # GraphRAG implementation
│       ├── core/                # Core components
│       └── adapters/            # Hermes adapters
└── testing/                     # Framework-agnostic testing infrastructure
```

### Development Priorities
1. Complete PathRAG core components
2. Implement Hermes at Limnos level
3. Develop GraphRAG core components
4. Implement testing infrastructure when both frameworks are operational
5. Add additional frameworks (e.g., LiteRAG) as needed

### Notes
- Testing infrastructure development should be delayed until multiple frameworks are available to ensure fairness
- Each framework should have equal input into the design of testing methodologies
- Hermes will serve as both a testing tool and production query router
