
# app.py
import streamlit as st
from dotenv import load_dotenv
import os
import asyncio
import tempfile

from utils.graph_utils import Neo4jAsyncDriver
from utils.ingest_utils import extract_text_from_file, chunk_text
from utils.embeddings_utils import get_embeddings
from utils.llm_utils import get_llm
from utils.graph_utils import (
    create_document_with_chunks,
    fetch_chunk_candidates
)
from utils.helpers import cosine_similarity, top_k_similar

load_dotenv()

st.set_page_config(page_title="GraphRAG (Neo4j + Ollama)", layout="wide")
st.title("📚 GraphRAG — ingest docs, build graph, ask questions")

# --- Initialize async loop & driver ---
@st.cache_resource
def get_event_loop():
    import asyncio
    return asyncio.new_event_loop()

loop = get_event_loop()

@st.cache_resource
def init_driver():
    driver = Neo4jAsyncDriver.from_env()
    return driver

driver = init_driver()
embeddings = get_embeddings()
llm = get_llm()

# Sidebar settings
st.sidebar.header("Ingestion settings")
chunk_size = st.sidebar.number_input("Chunk size (chars)", min_value=200, max_value=5000, value=1000)
chunk_overlap = st.sidebar.number_input("Chunk overlap (chars)", min_value=0, max_value=500, value=200)
max_candidates = st.sidebar.number_input("Max candidates to fetch from DB", min_value=10, max_value=2000, value=500)
top_k = st.sidebar.number_input("Top K chunks for RAG", min_value=1, max_value=30, value=5)

# --- Tabs ---
tab1, tab2 = st.tabs(["Ingest documents", "Ask questions"])

with tab1:
    st.header("Upload documents to build the knowledge graph")
    uploaded = st.file_uploader("Upload PDF / TXT / DOCX (or multiple)", accept_multiple_files=True, type=["pdf", "txt", "docx"])
    source_name = st.text_input("Source name (e.g., 'HR Handbook' or filename prefix')", value="uploaded_source")
    if st.button("Ingest files") and uploaded:
        status_box = st.empty()
        total_chunks = 0
        for file in uploaded:
            status_box.info(f"Ingesting {file.name} ...")
            # Save to temp file and extract text
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name
            raw_text = extract_text_from_file(tmp_path)
            # chunk
            chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            total_chunks += len(chunks)
            # embed chunks in batches
            batch_size = 32
            embeddings_list = []
            for i in range(0, len(chunks), batch_size):
                batch_texts = [c["text"] for c in chunks[i:i+batch_size]]
                embs = embeddings.embed_documents(batch_texts)
                embeddings_list.extend(embs)
            # create Document node and Chunk nodes in Neo4j
            loop.run_until_complete(create_document_with_chunks(
                driver=driver,
                doc_props={"title": file.name, "source": source_name},
                chunks=[{**chunks[i], "embedding": embeddings_list[i]} for i in range(len(chunks))]
            ))
            status_box.success(f"Ingested {file.name} — {len(chunks)} chunks")
        st.success(f"Finished ingesting {len(uploaded)} file(s), created {total_chunks} chunks.")

with tab2:
    st.header("Ask questions (RAG)")
    question = st.text_area("Enter question about ingested docs")
    if st.button("Get answer") and question.strip():
        st.info("Computing embeddings and retrieving relevant chunks...")
        q_emb = embeddings.embed_query(question)
        # fetch candidate chunks from DB
        candidates = loop.run_until_complete(fetch_chunk_candidates(driver=driver, limit=max_candidates))
        # candidates: list of dicts with 'text' and 'embedding'
        # compute similarity
        ranked = top_k_similar(q_emb, candidates, k=top_k)
        st.write("### 🔎 Retrieved context chunks (top results)")
        for i, item in enumerate(ranked):
            st.write(f"**Rank {i+1} (score: {item['score']:.4f})** — source: {item.get('source','unknown')}")
            st.write(item["text"][:1000] + ("…" if len(item["text"])>1000 else ""))
            st.write("---")
        # Build prompt for LLM
        context_text = "\n\n---\n\n".join([c["text"] for c in ranked])
        prompt = (
            "You are a helpful assistant. Use the context below to answer the question. "
            "If the answer is not contained in the context, say you don't know or provide best-effort reasoning.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
        )
        st.info("Calling LLM...")
        llm_response = llm(prompt)
        st.write("### 🤖 LLM answer")
        st.write(llm_response)
