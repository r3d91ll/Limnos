# GraphRAG Design Document

## 1. Overview

GraphRAG is a graph-based Retrieval Augmented Generation framework that leverages graph structures for more effective knowledge retrieval and generation. This document outlines the design and implementation strategy for GraphRAG within the Olympus project's Limnos component.

The implementation will utilize NetworkX for graph representation and Redis for caching, ensuring compatibility with our existing PathRAG implementation while maintaining the established Limnos architecture principles.

## 2. Architecture

### 2.1 System Components

GraphRAG consists of the following core components:

1. **Entity Extractor**: Identifies and extracts entities from documents
2. **Relationship Extractor**: Identifies relationships between entities
3. **Graph Constructor**: Builds the document graph from entities and relationships
4. **Graph Storage Manager**: Manages storage and retrieval of graph data
5. **Query Processor**: Processes queries for graph-based retrieval
6. **Graph Search Engine**: Retrieves relevant subgraphs based on queries
7. **Subgraph Extractor**: Extracts and prunes relevant subgraphs
8. **Document Ranker**: Ranks documents based on graph relevance
9. **Organizer**: Refines and structures the retrieved content
10. **Redis Cache Manager**: Manages graph caching and persistence

### 2.2 Component Interactions

```
┌─────────────────┐                ┌─────────────────┐
│  Document       │                │ Query           │
│  Collector      │                │ Input           │
└─────────┬───────┘                └─────────┬───────┘
          │                                  │
          ▼                                  ▼
┌───────────────────┐              ┌───────────────────┐
│ Entity Extractor  │              │ Query Processor   │
└─────────┬─────────┘              └─────────┬─────────┘
          │                                  │
          ▼                                  │
┌───────────────────┐                        │
│ Relationship      │                        │
│ Extractor         │                        │
└─────────┬─────────┘                        │
          │                                  │
          ▼                                  ▼
┌───────────────────┐              ┌───────────────────┐
│ Graph             │              │ Graph Search      │
│ Constructor       │◄────────────►│ Engine            │
└─────────┬─────────┘              └─────────┬─────────┘
          │                                  │
          ▼                                  ▼
┌───────────────────┐              ┌───────────────────┐
│ Graph Storage     │              │ Subgraph          │
│ Manager           │◄────────────►│ Extractor         │
└─────────┬─────────┘              └─────────┬─────────┘
          │                                  │
          │                                  ▼
          │                          ┌───────────────────┐
          │                          │ Document          │
          │                          │ Ranker            │
          │                          └───────┬───────────┘
          │                                  │
          │                                  ▼
          │                          ┌───────────────────┐
          │                          │ Organizer         │
          │                          └───────────────────┘
          │                                  │
          ▼                                  ▼
┌───────────────────────────────────────────────────────┐
│                    Redis Cache                        │
└───────────────────────────────────────────────────────┘
```

## 3. Component Specifications

### 3.1 Entity Extractor

**Purpose**: Extract named entities and concepts from documents.

**Key Functions**:
- `extract_entities(text: str) -> List[Entity]`
- `process_document(document: Dict[str, Any]) -> List[Entity]`
- `batch_process(documents: List[Dict[str, Any]]) -> Dict[str, List[Entity]]`

**Implementation Details**:
- Uses spaCy for NER and concept extraction
- Employs entity linking to connect mentions to canonical entities
- Captures entity metadata (type, position, confidence)
- Implements batching for efficient processing

### 3.2 Relationship Extractor

**Purpose**: Identify relationships between entities in documents.

**Key Functions**:
- `extract_relationships(text: str, entities: List[Entity]) -> List[Relationship]`
- `detect_explicit_relationships(entities: List[Entity], document: Dict[str, Any]) -> List[Relationship]`
- `infer_implicit_relationships(entities: List[Entity], text: str) -> List[Relationship]`

**Implementation Details**:
- Identifies explicit relationships (e.g., citations, references)
- Infers implicit relationships (co-occurrence, semantic similarity)
- Assigns relationship types and weights
- Supports domain-specific relationship extraction

### 3.3 Graph Constructor

**Purpose**: Build a document graph from entities and relationships.

**Key Functions**:
- `build_graph(entities: List[Entity], relationships: List[Relationship]) -> nx.Graph`
- `add_document_to_graph(document_id: str, entities: List[Entity], relationships: List[Relationship], graph: nx.Graph) -> nx.Graph`
- `merge_graphs(graphs: List[nx.Graph]) -> nx.Graph`

**Implementation Details**:
- Uses NetworkX for graph representation
- Creates nodes for entities with attributes
- Creates edges for relationships with attributes
- Implements incremental graph construction
- Supports both directed and undirected relationships

### 3.4 Graph Storage Manager

**Purpose**: Manage storage and retrieval of graph data.

**Key Functions**:
- `save_graph(graph: nx.Graph, graph_id: str) -> bool`
- `load_graph(graph_id: str) -> nx.Graph`
- `update_graph(graph_id: str, updates: Dict[str, Any]) -> bool`
- `save_document_mapping(document_id: str, graph_elements: Dict[str, Any]) -> bool`

**Implementation Details**:
- Stores graph data in framework-specific directories
- Implements serialization/deserialization for NetworkX graphs
- Maintains index of documents to graph elements
- Follows Limnos metadata architecture principles

### 3.5 Query Processor

**Purpose**: Process natural language queries for graph-based retrieval.

**Key Functions**:
- `process_query(query: str) -> Dict[str, Any]`
- `extract_query_entities(query: str) -> List[Entity]`
- `extract_query_relationships(query: str, entities: List[Entity]) -> List[Relationship]`
- `formulate_graph_query(entities: List[Entity], relationships: List[Relationship]) -> Dict[str, Any]`

**Implementation Details**:
- Extracts query entities and relationships
- Expands queries with relevant context
- Converts natural language to structured graph queries
- Supports different query types (entity-centric, relationship-centric)

### 3.6 Graph Search Engine

**Purpose**: Retrieve relevant subgraphs based on queries.

**Key Functions**:
- `search(query: Dict[str, Any], graph: nx.Graph) -> List[Dict[str, Any]]`
- `entity_based_search(query_entities: List[Entity], graph: nx.Graph) -> List[Dict[str, Any]]`
- `relationship_based_search(query_relationships: List[Relationship], graph: nx.Graph) -> List[Dict[str, Any]]`
- `combined_search(query: Dict[str, Any], graph: nx.Graph) -> List[Dict[str, Any]]`

**Implementation Details**:
- Implements various graph traversal algorithms (BFS, DFS)
- Supports semantic similarity search for entities
- Uses both structural and semantic features for ranking
- Optimizes search based on query characteristics

### 3.7 Subgraph Extractor

**Purpose**: Extract and prune relevant subgraphs from search results.

**Key Functions**:
- `extract_subgraph(graph: nx.Graph, nodes: List[str], max_hops: int = 2) -> nx.Graph`
- `prune_subgraph(subgraph: nx.Graph, query: Dict[str, Any], relevance_threshold: float = 0.5) -> nx.Graph`
- `score_node_relevance(node: str, node_attrs: Dict[str, Any], query: Dict[str, Any]) -> float`
- `score_edge_relevance(edge: Tuple[str, str], edge_attrs: Dict[str, Any], query: Dict[str, Any]) -> float`

**Implementation Details**:
- Extracts connected subgraphs around relevant nodes
- Scores nodes and edges based on query relevance
- Prunes irrelevant paths while preserving connectivity
- Balances coverage with relevance for optimal retrieval

### 3.8 Document Ranker

**Purpose**: Rank documents based on their relevance to the query using graph features.

**Key Functions**:
- `rank_documents(subgraphs: List[nx.Graph], document_ids: List[str], query: Dict[str, Any]) -> List[Tuple[str, float]]`
- `compute_graph_relevance_score(subgraph: nx.Graph, query: Dict[str, Any]) -> float`
- `compute_path_relevance_score(paths: List[List[str]], query: Dict[str, Any]) -> float`
- `combine_scores(scores: Dict[str, float], weights: Dict[str, float] = None) -> float`

**Implementation Details**:
- Leverages both graph structure and semantic content for ranking
- Considers path length, node relevance, and relationship strength
- Computes separate scores for different relevance dimensions
- Combines scores using configurable weighting

### 3.9 Organizer

**Purpose**: Refine and structure retrieved content for optimal generation.

**Key Functions**:
- `organize_content(subgraphs: List[nx.Graph], ranked_documents: List[Tuple[str, float]], query: Dict[str, Any]) -> Dict[str, Any]`
- `extract_key_facts(subgraph: nx.Graph) -> List[str]`
- `generate_structured_content(facts: List[str], context: Dict[str, Any]) -> Dict[str, Any]`
- `verbalize_graph(subgraph: nx.Graph, detail_level: str = 'medium') -> str`

**Implementation Details**:
- Extracts key facts from subgraphs
- Eliminates redundancy in retrieved content
- Structures content for easy consumption by LLMs
- Implements structure-aware verbalization of graphs

### 3.10 Redis Cache Manager

**Purpose**: Manage graph caching and persistence using Redis.

**Key Functions**:
- `cache_graph(graph_id: str, graph: nx.Graph, ttl: int = None) -> bool`
- `get_cached_graph(graph_id: str) -> Optional[nx.Graph]`
- `cache_subgraph(query_hash: str, subgraph: nx.Graph, ttl: int = None) -> bool`
- `get_cached_subgraph(query_hash: str) -> Optional[nx.Graph]`
- `invalidate_cache(document_id: str) -> bool`

**Implementation Details**:
- Serializes NetworkX graphs for Redis storage
- Implements efficient cache key generation
- Supports time-to-live for cache entries
- Provides cache invalidation strategies
- Optimizes memory usage with selective caching

## 4. Data Flow

### 4.1 Document Processing Flow

1. Document Collector provides documents to Entity Extractor
2. Entity Extractor identifies entities and passes to Relationship Extractor
3. Relationship Extractor identifies relationships between entities
4. Graph Constructor builds or updates the document graph
5. Graph Storage Manager persists the updated graph
6. Redis Cache Manager updates cached graph elements

### 4.2 Query Processing Flow

1. Query Processor receives natural language query
2. Query Processor extracts query entities and relationships
3. Graph Search Engine retrieves relevant graph elements
4. Subgraph Extractor extracts and prunes subgraphs
5. Document Ranker scores and ranks documents using subgraphs
6. Organizer refines and structures the content for generation

## 5. Metadata Storage Strategy

Following the established Limnos architecture principles:

1. **Universal Metadata**: Stored alongside source documents (generated by Document Collector)
2. **Framework-Specific Metadata**: Stored in GraphRAG-specific directories
3. **Clear Separation**: Maintained between universal and framework-specific metadata

Directory structure:
```
/limnos/data/
├── source_documents/            # Original documents and universal metadata
│   ├── paper1.pdf
│   ├── paper1.json              # Universal metadata
│   └── ...
└── implementations/
    ├── pathrag/                 # PathRAG-specific data
    │   └── ...
    └── graphrag/                # GraphRAG-specific data
        ├── graphs/              # Serialized graph data
        ├── entities/            # Entity metadata
        ├── relationships/       # Relationship metadata
        ├── document_mappings/   # Document-to-graph mappings
        └── embeddings/          # Vector embeddings for graph elements
```

## 6. Integration with Existing Components

### 6.1 Integration with Document Collector

- Subscribes to document collection events
- Processes new documents as they are added
- Extracts entities and relationships
- Updates graphs when documents change
- Maintains consistent metadata structure

### 6.2 Integration with Hermes

- Implements Hermes adapter for GraphRAG
- Translates Hermes queries to graph queries
- Returns results in Hermes-compatible format
- Supports parallel execution with other frameworks

### 6.3 Redis Integration

- Reuses existing Redis infrastructure
- Implements GraphRAG-specific serialization
- Defines appropriate cache invalidation strategies
- Optimizes memory usage for graph operations

## 7. Performance Considerations

### 7.1 Scalability

- Implements batched processing for documents
- Uses incremental graph updates
- Supports selective graph loading
- Optimizes query execution for large graphs

### 7.2 Caching Strategy

- Caches frequently accessed subgraphs
- Implements tiered caching (memory and Redis)
- Uses LRU eviction for cache management
- Supports background prefetching for common queries

### 7.3 Query Optimization

- Implements query planning for complex graph queries
- Uses indexing for efficient node and edge lookup
- Employs early pruning of irrelevant paths
- Balances traversal depth with relevance

## 8. Testing Methodology

### 8.1 Component Testing

- Unit tests for each component
- Integration tests for component interactions
- Benchmarking tests for performance evaluation

### 8.2 Functional Testing

- End-to-end tests for document processing
- End-to-end tests for query processing
- Comparison tests against PathRAG

### 8.3 Performance Testing

- Load testing with large document sets
- Query performance with varying complexity
- Cache effectiveness measurements
- Memory usage and optimization tests

## 9. Implementation Plan

### 9.1 Phase 1: Core Infrastructure

1. Set up GraphRAG directory structure
2. Implement entity and relationship extractors
3. Create graph constructor and storage manager
4. Implement Redis integration for caching

### 9.2 Phase 2: Retrieval Components

1. Implement query processor
2. Build graph search engine
3. Create subgraph extractor
4. Develop document ranker

### 9.3 Phase 3: Integration and Refinement

1. Implement organizer component
2. Create Hermes adapter
3. Build document collector integration
4. Implement end-to-end workflows

## 10. Conclusion

This design outlines a comprehensive implementation plan for GraphRAG, leveraging the strengths of graph-based retrieval while maintaining compatibility with the existing Limnos architecture. The implementation uses NetworkX for graph representation and Redis for caching, ensuring consistency with PathRAG for fair comparison while exploiting the unique capabilities of graph structures for improved retrieval performance.
