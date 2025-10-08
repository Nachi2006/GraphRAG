
# utils/graph_utils.py
import os
import asyncio
from neo4j import AsyncGraphDatabase

class Neo4jAsyncDriver:
    def __init__(self, driver):
        self._driver = driver

    @classmethod
    def from_env(cls):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        if not (uri and user and password):
            raise ValueError("Missing Neo4j connection details in .env")
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        return cls(driver)

    async def close(self):
        await self._driver.close()

    async def execute_write(self, query, parameters=None):
        async with self._driver.session() as session:
            return await session.execute_write(lambda tx: tx.run(query, parameters or {}).data())

    async def execute_read(self, query, parameters=None):
        async with self._driver.session() as session:
            return await session.execute_read(lambda tx: tx.run(query, parameters or {}).data())

# Graph operations
async def create_document_with_chunks(driver: Neo4jAsyncDriver, doc_props: dict, chunks: list):
    """
    Create a Document node and multiple Chunk nodes linked by HAS_CHUNK relationships.
    Each chunk is a dict with keys: 'text', 'chunk_index', 'source' (optional), 'embedding' (list of floats)
    """
    query = """
    UNWIND $chunks AS c
    MERGE (d:Document {title: $title, source: $source})
    CREATE (ch:Chunk {
        text: c.text,
        chunk_index: c.chunk_index,
        source: coalesce(c.source, $source),
        embedding: c.embedding
    })
    MERGE (d)-[:HAS_CHUNK]->(ch)
    RETURN d.title AS title, count(ch) AS created
    """
    params = {
        "title": doc_props.get("title"),
        "source": doc_props.get("source"),
        "chunks": chunks
    }
    async with driver._driver.session() as session:
        await session.run(query, params)

async def fetch_chunk_candidates(driver: Neo4jAsyncDriver, limit: int = 500):
    """
    Fetch up to `limit` chunk nodes and return a list of dicts with 'text', 'embedding', 'source', 'chunk_index'
    """
    query = "MATCH (c:Chunk) RETURN c.text AS text, c.embedding AS embedding, c.source AS source, c.chunk_index AS chunk_index LIMIT $limit"
    params = {"limit": limit}
    async with driver._driver.session() as session:
        result = await session.run(query, params)
        rows = []
        async for record in result:
            rows.append({
                "text": record["text"],
                "embedding": record["embedding"],
                "source": record["source"],
                "chunk_index": record["chunk_index"]
            })
        return rows
