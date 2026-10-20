# Project 02 — Smart Search Engine

## Objective

Build a practical retrieval engine that searches document chunks and returns the most relevant results using lexical, semantic, and hybrid search.

## Learning Modules

### Module 1: Retrieval Fundamentals
- [ ] Understand lexical vs. semantic retrieval
- [ ] Define the retrieval architecture

### Module 2: Document and Chunk Preparation
- [ ] Prepare chunks with stable IDs and metadata
- [ ] Inspect chunk quality

### Module 3: Baseline Lexical Search
- [ ] Implement TF-IDF retrieval
- [ ] Return ranked top-K chunks

### Module 4: Semantic Search and Embeddings
- [ ] Generate embeddings using OpenAI
- [ ] Rank chunks using cosine similarity
- [ ] Compare semantic and lexical results

### Module 5: Vector Index
- [ ] Build an initial local vector index
- [ ] Support indexing and querying

### Module 6: Hybrid Retrieval
- [ ] Combine lexical and semantic results
- [ ] Implement reciprocal rank fusion (RRF)

### Module 7: Retrieval Controls
- [ ] Support top-K and metadata filtering
- [ ] Return structured results

### Module 8: Evaluation Integration
- [ ] Connect real retrieval results to Project 01
- [ ] Compare retrieval configurations

### Module 9: Search API
- [ ] Expose retrieval through FastAPI
- [ ] Validate requests and responses with Pydantic

### Module 10: Documentation
- [ ] Document setup, usage, decisions, and evaluation results

## Engineering Decisions
- Use the existing repository and `uv` environment.
- Use OpenAI embeddings for semantic search.
- Start with local vector search before considering a vector database.
- Manually verify behavior; automated tests are out of scope for now.
- Commit and push each meaningful module.

