"""
Ladybug Memory MCP Server
===========================
AI agent memory powered by LadybugDB — graph + vector + FTS in one embedded database.

Graph data model:
  - Memory nodes (content, embeddings, metadata)
  - Topic nodes (auto-linked from tags)
  - Relationships: ABOUT, RELATED_TO, SUPERSEDES

Tools:
  - memory_store: Store with auto-dedup + auto-link topics
  - memory_search: Hybrid semantic + keyword + graph search
  - memory_get: Full content by ID
  - memory_update: Update content/metadata by ID
  - memory_delete: Delete node + relationships
  - memory_relate: Create relationship between nodes
  - memory_traverse: Run Cypher graph queries
  - memory_list: List by recency, type, topic, importance
  - memory_stats: Database statistics
  - memory_consolidate: Find merge candidates
"""

import hashlib
import inspect
import json
import logging
import os
import time
from typing import Optional

import real_ladybug as lb
from fastembed import TextEmbedding
from mcp.server.fastmcp import FastMCP

# --- Logging ---
logger = logging.getLogger("ladybug-memory")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- Configuration ---
DB_PATH = os.environ.get("MEMORY_DB_PATH", os.path.expanduser("~/.agent-memory/memory.lbug"))
EMBEDDING_MODEL = os.environ.get("MEMORY_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.environ.get("MEMORY_EMBEDDING_DIM", "384"))
DEDUP_THRESHOLD = float(os.environ.get("MEMORY_DEDUP_THRESHOLD", "0.92"))
LATENCY_WARN_MS = int(os.environ.get("MEMORY_LATENCY_WARN_MS", "50"))

MAX_CONTENT_LENGTH = int(os.environ.get("MEMORY_MAX_CONTENT", "500"))
MAX_SEARCH_RESULTS = int(os.environ.get("MEMORY_SEARCH_LIMIT", "10"))
MAX_LIST_RESULTS = int(os.environ.get("MEMORY_LIST_LIMIT", "20"))
MAX_CONSOLIDATE_CLUSTERS = int(os.environ.get("MEMORY_CONSOLIDATE_CLUSTERS", "10"))
MAX_CONSOLIDATE_SCAN = int(os.environ.get("MEMORY_CONSOLIDATE_SCAN", "200"))

# --- Globals ---
mcp = FastMCP("ladybug-memory")
_embed_model: Optional[TextEmbedding] = None
_conn: Optional[lb.Connection] = None
_db: Optional[lb.Database] = None


def _timed(operation: str):
    """Decorator: injects elapsed_ms into JSON response, logs slow operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result_json = func(*args, **kwargs)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            try:
                result = json.loads(result_json)
                if isinstance(result, dict):
                    result["elapsed_ms"] = elapsed_ms
                elif isinstance(result, list):
                    result = {"results": result, "elapsed_ms": elapsed_ms}
                result_json = json.dumps(result, indent=2)
            except (json.JSONDecodeError, TypeError):
                pass
            if elapsed_ms > LATENCY_WARN_MS:
                logger.warning(f"SLOW {operation}: {elapsed_ms}ms")
            else:
                logger.info(f"{operation}: {elapsed_ms}ms")
            return result_json
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__annotations__ = func.__annotations__
        wrapper.__signature__ = inspect.signature(func)
        wrapper.__wrapped__ = func
        return wrapper
    return decorator


def get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(EMBEDDING_MODEL)
    return _embed_model


def get_conn() -> lb.Connection:
    global _conn, _db
    if _conn is not None:
        return _conn

    from pathlib import Path
    if DB_PATH == ":memory:":
        _db = lb.Database(":memory:")
    else:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _db = lb.Database(DB_PATH)

    _conn = lb.Connection(_db)

    # Load extensions
    _conn.execute("INSTALL vector; LOAD EXTENSION vector;")
    _conn.execute("INSTALL fts; LOAD EXTENSION fts;")

    _init_schema(_conn)
    return _conn


def _init_schema(conn: lb.Connection):
    """Create graph schema if not exists."""
    # Memory node — the core entity
    _safe_execute(conn, f"""
        CREATE NODE TABLE Memory(
            id INT64 PRIMARY KEY,
            content STRING,
            content_hash STRING,
            category STRING DEFAULT 'general',
            tags STRING DEFAULT '[]',
            importance INT64 DEFAULT 3,
            access_count INT64 DEFAULT 0,
            created_at DOUBLE DEFAULT 0.0,
            updated_at DOUBLE DEFAULT 0.0,
            embedding FLOAT[{EMBEDDING_DIM}]
        );
    """)

    # Topic node — auto-created from tags
    _safe_execute(conn, """
        CREATE NODE TABLE Topic(
            name STRING PRIMARY KEY
        );
    """)

    # Relationships
    _safe_execute(conn, "CREATE REL TABLE ABOUT(FROM Memory TO Topic);")
    _safe_execute(conn, "CREATE REL TABLE RELATED_TO(FROM Memory TO Memory);")
    _safe_execute(conn, "CREATE REL TABLE SUPERSEDES(FROM Memory TO Memory);")

    # Vector index
    _safe_execute(conn, """
        CALL CREATE_VECTOR_INDEX(
            'Memory', 'memory_vec_idx', 'embedding',
            metric := 'cosine'
        );
    """)

    logger.info("Schema initialized")


def _safe_execute(conn, query, params=None):
    """Execute a query, ignoring errors (for CREATE IF NOT EXISTS patterns)."""
    try:
        if params:
            conn.execute(query, params)
        else:
            conn.execute(query)
    except Exception:
        pass


# --- Helpers ---

def _embed(text: str) -> list[float]:
    return list(get_embed_model().embed([text]))[0].tolist()


def _truncate(text: str, max_len: int = MAX_CONTENT_LENGTH) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


def _parse_tags(val) -> list:
    """Safely parse tags from LadybugDB — could be string, list, or None."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        # Handle comma-separated format: "tag1,tag2,tag3"
        if val and not val.startswith("["):
            return [t.strip() for t in val.split(",") if t.strip()]
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except (json.JSONDecodeError, TypeError):
            return [val] if val else []
    return []


def _format_tags(tags: list[str]) -> str:
    """Format tags for storage in LadybugDB as comma-separated string."""
    return ",".join(tags) if tags else ""


def _normalize(content: str) -> str:
    return " ".join(content.lower().split())


def _content_hash(content: str) -> str:
    return hashlib.sha256(_normalize(content).encode()).hexdigest()


def _next_id(conn: lb.Connection) -> int:
    result = conn.execute("MATCH (m:Memory) RETURN MAX(m.id);")
    if result.has_next():
        val = result.get_next()[0]
        if val is not None:
            return val + 1
    return 1


def _count_memories(conn: lb.Connection) -> int:
    result = conn.execute("MATCH (m:Memory) RETURN COUNT(*);")
    if result.has_next():
        return result.get_next()[0]
    return 0


def _ensure_topics(conn: lb.Connection, memory_id: int, tags: list[str]):
    """Create Topic nodes and ABOUT relationships for each tag."""
    for tag in tags:
        _safe_execute(conn, "MERGE (t:Topic {name: $name});", {"name": tag})
        _safe_execute(conn, """
            MATCH (m:Memory {id: $mid}), (t:Topic {name: $name})
            MERGE (m)-[:ABOUT]->(t);
        """, {"mid": memory_id, "name": tag})


def _collect_results(result) -> list:
    """Collect all rows from a query result."""
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    return rows


# --- MCP Tools ---

@mcp.tool()
@_timed("memory_store")
def memory_store(
    content: str,
    category: str = "general",
    tags: list[str] = [],
    importance: int = 3,
) -> str:
    """
    Store a new memory with automatic deduplication and topic linking.

    Dedup: exact hash match rejects duplicates, semantic similarity > threshold updates existing.
    Topics: each tag becomes a Topic node linked via ABOUT relationship.

    Args:
        content: The memory content (learning, preference, decision, pattern)
        category: One of: learning, preference, decision, pattern, general
        tags: List of topic tags (e.g. ["python", "architecture"])
        importance: 1-5 scale (1=trivial, 5=critical)

    Returns:
        JSON status (stored_new, updated_existing, or already_exists)
    """
    conn = get_conn()
    c_hash = _content_hash(content)

    # Layer 1: Exact hash dedup
    result = conn.execute(
        "MATCH (m:Memory {content_hash: $hash}) RETURN m.id;",
        {"hash": c_hash},
    )
    if result.has_next():
        existing_id = result.get_next()[0]
        conn.execute(
            """MATCH (m:Memory {id: $id})
               SET m.importance = CASE WHEN m.importance < 5 THEN m.importance + 1 ELSE 5 END,
                   m.updated_at = $now;""",
            {"id": existing_id, "now": time.time()},
        )
        return json.dumps({"status": "already_exists", "id": existing_id,
                           "message": "Exact match found. Importance bumped."})

    # Layer 2: Semantic dedup
    embedding = _embed(content)

    if _count_memories(conn) > 0:
        try:
            result = conn.execute(
                """CALL QUERY_VECTOR_INDEX('Memory', 'memory_vec_idx', $query, $k)
                   WITH node AS m, distance
                   RETURN m.id, m.content, m.category, m.tags, m.importance, distance
                   ORDER BY distance LIMIT 3;""",
                {"query": embedding, "k": 3},
            )
            for row in _collect_results(result):
                similarity = round(1.0 - row[5], 4)
                if similarity >= DEDUP_THRESHOLD:
                    match_id = row[0]
                    match_content = row[1]
                    match_tags = _parse_tags(row[3])
                    match_importance = row[4]

                    keep = content if len(content) > len(match_content) else match_content
                    merged_tags = list(set(match_tags + tags))
                    new_imp = min(5, max(match_importance, importance) + 1)
                    new_emb = _embed(keep)

                    conn.execute(
                        """MATCH (m:Memory {id: $id})
                           SET m.content = $content, m.content_hash = $hash,
                               m.tags = $tags, m.importance = $imp,
                               m.updated_at = $now, m.embedding = $emb;""",
                        {"id": match_id, "content": keep, "hash": _content_hash(keep),
                         "tags": _format_tags(merged_tags), "imp": new_imp,
                         "now": time.time(), "emb": new_emb},
                    )
                    _ensure_topics(conn, match_id, merged_tags)
                    return json.dumps({
                        "status": "updated_existing", "id": match_id,
                        "similarity": similarity,
                        "message": f"Merged (similarity: {similarity}). Importance: {new_imp}.",
                    })
        except Exception as e:
            logger.warning(f"Vector dedup search failed: {e}")

    # No match — insert new
    now = time.time()
    mem_id = _next_id(conn)
    conn.execute(
        """CREATE (m:Memory {
               id: $id, content: $content, content_hash: $hash,
               category: $cat, tags: $tags, importance: $imp,
               access_count: 0, created_at: $now, updated_at: $now,
               embedding: $emb
           });""",
        {"id": mem_id, "content": content, "hash": c_hash,
         "cat": category, "tags": _format_tags(tags), "imp": importance,
         "now": now, "emb": embedding},
    )
    _ensure_topics(conn, mem_id, tags)

    return json.dumps({"status": "stored_new", "id": mem_id, "message": "New memory stored."})


@mcp.tool()
@_timed("memory_search")
def memory_search(
    query: str,
    category: Optional[str] = None,
    tags: Optional[list[str]] = None,
    top_k: int = 5,
) -> str:
    """
    Search memories using hybrid semantic + keyword search.

    Args:
        query: Natural language search query
        category: Filter by category
        tags: Filter by tags
        top_k: Number of results (default 5, max 10)

    Returns:
        JSON list of matching memories ranked by relevance
    """
    conn = get_conn()
    top_k = min(top_k, MAX_SEARCH_RESULTS)
    scores: dict[int, float] = {}
    memory_data: dict[int, dict] = {}

    # Vector search
    embedding = _embed(query)
    if _count_memories(conn) > 0:
        try:
            result = conn.execute(
                """CALL QUERY_VECTOR_INDEX('Memory', 'memory_vec_idx', $query, $k)
                   WITH node AS m, distance
                   RETURN m.id, m.content, m.category, m.tags, m.importance, m.access_count, distance
                   ORDER BY distance;""",
                {"query": embedding, "k": top_k * 2},
            )
            for row in _collect_results(result):
                mid = row[0]
                similarity = 1.0 - row[6]
                scores[mid] = similarity
                memory_data[mid] = {
                    "id": mid, "content": row[1], "category": row[2],
                    "tags": _parse_tags(row[3]),
                    "importance": row[4], "access_count": row[5],
                }
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

    # Keyword search via graph traversal on topics
    if tags:
        for tag in tags:
            try:
                result = conn.execute(
                    """MATCH (m:Memory)-[:ABOUT]->(t:Topic {name: $tag})
                       RETURN m.id;""",
                    {"tag": tag},
                )
                for row in _collect_results(result):
                    scores[row[0]] = scores.get(row[0], 0) + 0.15
            except Exception:
                pass

    # Build results
    results = []
    for mid, score in sorted(scores.items(), key=lambda x: -x[1]):
        mem = memory_data.get(mid)
        if not mem:
            # Fetch from DB
            try:
                r = conn.execute(
                    """MATCH (m:Memory {id: $id})
                       RETURN m.content, m.category, m.tags, m.importance, m.access_count;""",
                    {"id": mid},
                )
                if r.has_next():
                    row = r.get_next()
                    mem = {"id": mid, "content": row[0], "category": row[1],
                           "tags": _parse_tags(row[2]),
                           "importance": row[3], "access_count": row[4]}
            except Exception:
                continue

        if not mem:
            continue
        if category and mem["category"] != category:
            continue

        boosted = score * (1 + (mem.get("importance", 3) - 3) * 0.1)
        results.append({
            "id": mid, "content": _truncate(mem["content"]),
            "category": mem["category"], "tags": mem["tags"],
            "importance": mem["importance"],
            "relevance_score": round(boosted, 4),
            "access_count": mem.get("access_count", 0),
        })
        if len(results) >= top_k:
            break

    # Bump access counts
    for r in results:
        _safe_execute(conn,
            "MATCH (m:Memory {id: $id}) SET m.access_count = m.access_count + 1;",
            {"id": r["id"]})

    return json.dumps(results, indent=2)


@mcp.tool()
def memory_get(memory_id: int) -> str:
    """
    Get full untruncated content of a memory by ID.

    Args:
        memory_id: The memory ID

    Returns:
        JSON with full memory content, metadata, and linked topics
    """
    conn = get_conn()
    result = conn.execute(
        """MATCH (m:Memory {id: $id})
           RETURN m.id, m.content, m.category, m.tags, m.importance,
                  m.access_count, m.created_at, m.updated_at;""",
        {"id": memory_id},
    )
    if not result.has_next():
        return json.dumps({"status": "not_found", "message": f"Memory {memory_id} not found."})

    r = result.get_next()

    # Get linked topics
    topics_result = conn.execute(
        "MATCH (m:Memory {id: $id})-[:ABOUT]->(t:Topic) RETURN t.name;",
        {"id": memory_id},
    )
    topics = [row[0] for row in _collect_results(topics_result)]

    _safe_execute(conn,
        "MATCH (m:Memory {id: $id}) SET m.access_count = m.access_count + 1;",
        {"id": memory_id})

    return json.dumps({
        "id": r[0], "content": r[1] or "", "category": r[2] or "general",
        "tags": _parse_tags(r[3]),
        "importance": r[4] or 3, "access_count": (r[5] or 0) + 1,
        "created_at": r[6] or 0, "updated_at": r[7] or 0,
        "topics": topics,
    }, indent=2)


@mcp.tool()
def memory_update(memory_id: int, content: Optional[str] = None,
                  importance: Optional[int] = None, tags: Optional[list[str]] = None) -> str:
    """
    Update a memory's content, importance, or tags.

    Args:
        memory_id: The memory ID to update
        content: New content (re-embeds if changed)
        importance: New importance (1-5)
        tags: New tags (re-links topics)

    Returns:
        JSON status
    """
    conn = get_conn()

    # Fetch existing memory
    result = conn.execute(
        """MATCH (m:Memory {id: $id})
           RETURN m.id, m.content, m.content_hash, m.category, m.tags,
                  m.importance, m.access_count, m.created_at, m.updated_at;""",
        {"id": memory_id},
    )
    if not result.has_next():
        return json.dumps({"status": "not_found", "message": f"Memory {memory_id} not found."})

    r = result.get_next()
    now = time.time()

    if content is not None:
        # Vector index prevents SET on embedding — must delete and re-create
        new_content = content
        new_hash = _content_hash(content)
        new_emb = _embed(content)
        new_tags = _format_tags(tags) if tags is not None else r[4]
        new_imp = min(5, max(1, importance)) if importance is not None else r[5]

        # Delete old node + relationships
        conn.execute("MATCH (m:Memory {id: $id}) DETACH DELETE m;", {"id": memory_id})

        # Re-create with same ID
        conn.execute(
            """CREATE (m:Memory {
                   id: $id, content: $content, content_hash: $hash,
                   category: $cat, tags: $tags, importance: $imp,
                   access_count: $ac, created_at: $ca, updated_at: $now,
                   embedding: $emb
               });""",
            {"id": memory_id, "content": new_content, "hash": new_hash,
             "cat": r[3], "tags": new_tags, "imp": new_imp,
             "ac": r[6] or 0, "ca": r[7] or now, "now": now, "emb": new_emb},
        )
        # Re-link topics
        parsed_tags = _parse_tags(new_tags)
        _ensure_topics(conn, memory_id, parsed_tags)
    else:
        # No content change — safe to SET without touching embedding
        if importance is not None:
            conn.execute(
                "MATCH (m:Memory {id: $id}) SET m.importance = $imp, m.updated_at = $now;",
                {"id": memory_id, "imp": min(5, max(1, importance)), "now": now},
            )

        if tags is not None:
            conn.execute(
                "MATCH (m:Memory {id: $id}) SET m.tags = $tags, m.updated_at = $now;",
                {"id": memory_id, "tags": _format_tags(tags), "now": now},
            )
            _safe_execute(conn, "MATCH (m:Memory {id: $id})-[r:ABOUT]->() DELETE r;", {"id": memory_id})
            _ensure_topics(conn, memory_id, tags)

    return json.dumps({"status": "updated", "id": memory_id, "message": "Memory updated."})


@mcp.tool()
def memory_delete(memory_id: int) -> str:
    """
    Delete a memory and all its relationships.

    Args:
        memory_id: The memory ID to delete

    Returns:
        JSON status
    """
    conn = get_conn()
    result = conn.execute("MATCH (m:Memory {id: $id}) RETURN m.id;", {"id": memory_id})
    if not result.has_next():
        return json.dumps({"status": "not_found", "message": f"Memory {memory_id} not found."})

    conn.execute("MATCH (m:Memory {id: $id}) DETACH DELETE m;", {"id": memory_id})
    return json.dumps({"status": "deleted", "id": memory_id, "message": "Memory and relationships deleted."})


@mcp.tool()
def memory_relate(from_id: int, to_id: int, relationship: str = "RELATED_TO") -> str:
    """
    Create a relationship between two memory nodes.

    Args:
        from_id: Source memory ID
        to_id: Target memory ID
        relationship: One of: RELATED_TO, SUPERSEDES

    Returns:
        JSON status
    """
    conn = get_conn()
    rel = relationship.upper()
    if rel not in ("RELATED_TO", "SUPERSEDES"):
        return json.dumps({"status": "error", "message": f"Unknown relationship: {rel}. Use RELATED_TO or SUPERSEDES."})

    try:
        conn.execute(
            f"MATCH (a:Memory {{id: $from}}), (b:Memory {{id: $to}}) CREATE (a)-[:{rel}]->(b);",
            {"from": from_id, "to": to_id},
        )
        return json.dumps({"status": "created", "from": from_id, "to": to_id,
                           "relationship": rel, "message": f"Relationship {rel} created."})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def memory_traverse(cypher_query: str) -> str:
    """
    Run a read-only Cypher query against the memory graph.
    Use for graph traversals like "find all memories about topic X"
    or "what superseded decision Y".

    Args:
        cypher_query: A Cypher MATCH/RETURN query (read-only)

    Returns:
        JSON list of result rows
    """
    conn = get_conn()
    query_upper = cypher_query.strip().upper()

    # Block write operations
    for keyword in ("CREATE", "DELETE", "SET ", "MERGE", "DROP", "REMOVE"):
        if keyword in query_upper:
            return json.dumps({"status": "error",
                               "message": f"Write operations not allowed in traverse. Use other tools for mutations."})

    try:
        result = conn.execute(cypher_query)
        rows = _collect_results(result)
        return json.dumps({"rows": rows, "count": len(rows)}, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def memory_list(count: int = 10, category: Optional[str] = None,
                topic: Optional[str] = None, min_importance: Optional[int] = None) -> str:
    """
    List memories with optional filters.

    Args:
        count: Number to return (default 10, max 20)
        category: Filter by category (learning, preference, decision, pattern, general)
        topic: Filter by topic name (memories linked via ABOUT)
        min_importance: Minimum importance level (1-5)

    Returns:
        JSON list of memories
    """
    conn = get_conn()
    count = min(count, MAX_LIST_RESULTS)

    if topic:
        # Graph traversal to find memories about a topic
        query = """
            MATCH (m:Memory)-[:ABOUT]->(t:Topic {name: $topic})
            RETURN m.id, m.content, m.category, m.tags, m.importance, m.access_count
            ORDER BY m.updated_at DESC LIMIT $limit;
        """
        result = conn.execute(query, {"topic": topic, "limit": count})
    elif category:
        result = conn.execute(
            """MATCH (m:Memory) WHERE m.category = $cat
               RETURN m.id, m.content, m.category, m.tags, m.importance, m.access_count
               ORDER BY m.updated_at DESC LIMIT $limit;""",
            {"cat": category, "limit": count},
        )
    elif min_importance:
        result = conn.execute(
            """MATCH (m:Memory) WHERE m.importance >= $imp
               RETURN m.id, m.content, m.category, m.tags, m.importance, m.access_count
               ORDER BY m.updated_at DESC LIMIT $limit;""",
            {"imp": min_importance, "limit": count},
        )
    else:
        result = conn.execute(
            """MATCH (m:Memory)
               RETURN m.id, m.content, m.category, m.tags, m.importance, m.access_count
               ORDER BY m.updated_at DESC LIMIT $limit;""",
            {"limit": count},
        )

    memories = []
    for row in _collect_results(result):
        memories.append({
            "id": row[0], "content": _truncate(row[1]),
            "category": row[2],
            "tags": _parse_tags(row[3]),
            "importance": row[4], "access_count": row[5],
        })
    return json.dumps(memories, indent=2)


@mcp.tool()
@_timed("memory_stats")
def memory_stats() -> str:
    """
    Get statistics about the memory graph.

    Returns:
        JSON with counts, categories, topics, importance distribution.
    """
    conn = get_conn()

    total = _count_memories(conn)

    # Categories
    cats = {}
    for row in _collect_results(conn.execute(
        "MATCH (m:Memory) RETURN m.category, COUNT(*) ORDER BY COUNT(*) DESC;"
    )):
        cats[row[0]] = row[1]

    # Importance
    imp_dist = {}
    for row in _collect_results(conn.execute(
        "MATCH (m:Memory) RETURN m.importance, COUNT(*) ORDER BY m.importance;"
    )):
        imp_dist[str(row[0])] = row[1]

    # Topics
    topics = {}
    for row in _collect_results(conn.execute(
        "MATCH (t:Topic)<-[:ABOUT]-(m:Memory) RETURN t.name, COUNT(m) ORDER BY COUNT(m) DESC LIMIT 10;"
    )):
        topics[row[0]] = row[1]

    # Relationships
    rel_count = 0
    try:
        r = conn.execute("MATCH ()-[r]->() RETURN COUNT(r);")
        if r.has_next():
            rel_count = r.get_next()[0]
    except Exception:
        pass

    # Top accessed
    top = []
    for row in _collect_results(conn.execute(
        "MATCH (m:Memory) RETURN m.id, m.content, m.access_count ORDER BY m.access_count DESC LIMIT 5;"
    )):
        top.append({"id": row[0], "content": row[1][:80] if row[1] else "", "access_count": row[2]})

    return json.dumps({
        "total_memories": total,
        "total_relationships": rel_count,
        "categories": cats,
        "importance_distribution": imp_dist,
        "top_topics": topics,
        "most_accessed": top,
        "db_path": DB_PATH,
    }, indent=2)


@mcp.tool()
@_timed("memory_consolidate")
def memory_consolidate(similarity_threshold: float = 0.88) -> str:
    """
    Find clusters of similar memories that could be merged.
    Returns candidates for review — does not auto-merge.

    Args:
        similarity_threshold: Similarity threshold (default 0.88)

    Returns:
        JSON list of memory clusters
    """
    conn = get_conn()
    total = _count_memories(conn)
    if total < 2:
        return json.dumps({"clusters": [], "message": "Not enough memories."})

    # Get memories to scan
    result = conn.execute(
        """MATCH (m:Memory)
           RETURN m.id, m.content, m.importance, m.embedding
           ORDER BY m.updated_at DESC LIMIT $limit;""",
        {"limit": MAX_CONSOLIDATE_SCAN},
    )
    all_mems = _collect_results(result)

    visited = set()
    clusters = []

    for mem in all_mems:
        mid, content, importance, embedding = mem[0], mem[1], mem[2], mem[3]
        if mid in visited or embedding is None:
            continue

        # Search for similar using stored embedding
        try:
            result = conn.execute(
                """CALL QUERY_VECTOR_INDEX('Memory', 'memory_vec_idx', $query, $k)
                   WITH node AS m, distance
                   RETURN m.id, m.content, distance;""",
                {"query": list(embedding), "k": 6},
            )
            cluster_members = []
            for row in _collect_results(result):
                sim = round(1.0 - row[2], 4)
                if row[0] != mid and row[0] not in visited and sim >= similarity_threshold:
                    cluster_members.append({
                        "id": row[0], "content": _truncate(row[1]), "similarity": sim,
                    })
                    visited.add(row[0])

            if cluster_members:
                visited.add(mid)
                clusters.append({
                    "anchor": {"id": mid, "content": _truncate(content), "importance": importance},
                    "similar": cluster_members,
                    "suggestion": "These memories could be merged into one.",
                })
                if len(clusters) >= MAX_CONSOLIDATE_CLUSTERS:
                    break
        except Exception:
            continue

    return json.dumps({
        "clusters": clusters, "total_clusters": len(clusters),
        "memories_scanned": len(all_mems),
        "capped": len(clusters) >= MAX_CONSOLIDATE_CLUSTERS,
    }, indent=2)


# --- Entry point ---

def main():
    """Entry point for the MCP server (used by pyproject.toml console_scripts)."""
    mcp.run()


if __name__ == "__main__":
    main()
