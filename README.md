# 🐞 Ladybug Memory MCP Server

[![PyPI version](https://badge.fury.io/py/ladybug-memory-mcp.svg)](https://pypi.org/project/ladybug-memory-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Persistent graph memory for AI agents using [LadybugDB](https://ladybugdb.com/) — an embedded graph database with native vector search and full-text search.

Give your AI agent memory that persists across sessions, deduplicates automatically, and models knowledge as a graph with typed relationships.

## Why Ladybug Memory?

- **Graph memory** — memories linked via Topic nodes and relationships (RELATED_TO, SUPERSEDES) with Cypher queries
- **Three-layer auto-dedup** — exact hash + semantic similarity + LLM-driven consolidation
- **Hybrid search** — HNSW vector search + full-text search combined
- **Topic auto-linking** — tags become graph nodes, enabling traversal queries
- **Embedded** — no Docker, no server process, single database directory
- **Zero config** — sensible defaults, just install and run
- **Importance & access tracking** — memories ranked by relevance and usage

## Quick Start

```bash
# Run directly with uvx (no install needed)
uvx ladybug-memory-mcp
```

Or install and run:

```bash
pip install ladybug-memory-mcp
ladybug-memory-mcp
```

## MCP Configuration

Add to your MCP client config (Kiro, Claude Desktop, Cursor, etc.):

```json
{
  "mcpServers": {
    "ladybug-memory": {
      "command": "uvx",
      "args": ["ladybug-memory-mcp@latest"],
      "env": {
        "FASTMCP_LOG_LEVEL": "ERROR"
      }
    }
  }
}
```

That's it — zero config required. All settings have sensible defaults.

## Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Store a memory with auto-dedup and auto-link to Topic nodes |
| `memory_search` | Hybrid semantic + keyword search, ranked by relevance |
| `memory_get` | Get full untruncated content of a memory by ID |
| `memory_update` | Update content, importance, or tags of a memory |
| `memory_delete` | Delete a memory and all its relationships |
| `memory_relate` | Create RELATED_TO or SUPERSEDES relationship between memories |
| `memory_traverse` | Run read-only Cypher queries against the memory graph |
| `memory_list` | List memories filtered by recency, category, topic, or importance |
| `memory_stats` | Database statistics: counts, categories, topics, relationships |
| `memory_consolidate` | Find clusters of similar memories for review and merging |

## Graph Data Model

```
(:Memory)  — content, embedding, category, tags, importance, access_count, timestamps
(:Topic)   — auto-created from tags

(:Memory)-[:ABOUT]->(:Topic)          # memory is about a topic
(:Memory)-[:RELATED_TO]->(:Memory)    # memories are related
(:Memory)-[:SUPERSEDES]->(:Memory)    # newer memory replaces older
```

### Example: Store and Search

```python
# Store a memory (via MCP tool call)
memory_store(
    content="User prefers Python over Node.js for backend tools",
    category="preference",
    tags=["python", "nodejs", "backend"],
    importance=4
)

# Search memories
memory_search(query="what language does the user prefer")

# Traverse the graph
memory_traverse(
    cypher_query="MATCH (m:Memory)-[:ABOUT]->(t:Topic {name: 'python'}) RETURN m.content"
)
```

### Example: Graph Relationships

```python
# Link related memories
memory_relate(from_id=5, to_id=3, relationship="RELATED_TO")

# Mark a decision as superseded
memory_relate(from_id=8, to_id=2, relationship="SUPERSEDES")

# Find all memories about a topic
memory_traverse(
    cypher_query="MATCH (m:Memory)-[:ABOUT]->(t:Topic) RETURN t.name, COUNT(m) ORDER BY COUNT(m) DESC"
)
```

## Three-Layer Deduplication

Every `memory_store` call runs through three dedup layers:

1. **Exact hash** — SHA256 of normalized content. Identical content is rejected, importance bumped.
2. **Semantic similarity** — If cosine similarity > 0.92 with an existing memory, merges into it (keeps longer content, merges tags, bumps importance).
3. **Consolidation** — Manual via `memory_consolidate`. Finds clusters of related memories for LLM-driven review and merging.

## Categories

| Category | Use for |
|----------|---------|
| `learning` | Technical knowledge, facts, how things work |
| `preference` | User preferences and choices |
| `decision` | Architecture decisions, tool choices |
| `pattern` | Recurring workflows, conventions |
| `general` | Everything else (default) |

## Configuration

All settings are optional — defaults work out of the box.

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MEMORY_DB_PATH` | `~/.agent-memory/memory.lbug` | LadybugDB database path |
| `MEMORY_DEDUP_THRESHOLD` | `0.92` | Semantic similarity threshold for auto-dedup |
| `MEMORY_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model for embeddings |
| `MEMORY_EMBEDDING_DIM` | `384` | Embedding dimension (must match model) |
| `MEMORY_SEARCH_LIMIT` | `10` | Max results from `memory_search` |
| `MEMORY_LIST_LIMIT` | `20` | Max results from `memory_list` |
| `MEMORY_MAX_CONTENT` | `500` | Content truncation length in search/list results |
| `MEMORY_LATENCY_WARN_MS` | `50` | Log warning when operation exceeds this (ms) |

### In-Memory Mode (Testing)

```json
"env": { "MEMORY_DB_PATH": ":memory:" }
```

All data is ephemeral — lost on restart. Useful for testing.

## Kiro Power

This server is also available as a [Kiro Power](https://github.com/arunkumars-mf/ladybug-memory-power) with:
- Pre-configured MCP server
- Three hooks for automatic memory persistence and recall
- Steering files with setup guide and Cypher query examples

## Architecture

```
AI Agent (Kiro, Claude, etc.)
    │
    ├─ memory_store ──→ embed content → dedup check → insert node → link topics
    ├─ memory_search ─→ embed query → HNSW vector search + FTS → rank & return
    ├─ memory_traverse → execute Cypher → return graph results
    │
    └─ LadybugDB (embedded, single directory)
        ├─ Memory nodes (content + FLOAT[384] embeddings)
        ├─ Topic nodes (auto-linked from tags)
        ├─ HNSW vector index (cosine similarity)
        ├─ FTS index (keyword search)
        └─ Graph relationships (ABOUT, RELATED_TO, SUPERSEDES)
```

## Requirements

- Python 3.10+
- Dependencies installed automatically: `real-ladybug`, `fastembed`, `mcp`
- ~130MB disk for the embedding model (downloaded on first run)

## Contributing

Issues and PRs welcome. See [LICENSE](LICENSE) for terms.

## License

[MIT](LICENSE)
