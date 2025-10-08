import streamlit as st
import asyncio
from neo4j import AsyncGraphDatabase
import os
import json
from dotenv import load_dotenv
import re
load_dotenv()

# Import Ollama-specific classes from the library
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import OllamaLLM

# --- App Configuration ---
VECTOR_INDEX_NAME = "entity_embeddings"
EMBEDDING_DIM = 768  # Dimensions for the default ollama embedding model (e.g., nomic-embed-text)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_LLM_MODEL = os.getenv("LLM")
OLLAMA_EMBEDDING_MODEL = os.getenv("EMBED")

# --- Core GraphRAG Functions (Adapted for Ollama & Streamlit) ---

async def build_graph_manually(driver_config, text, llm, embedder):
    """
    Manually builds a knowledge graph by performing the following steps:
    1.  Extracts entities and relationships from text using an LLM.
    2.  Generates embeddings for each entity.
    3.  Loads the graph data into Neo4j.
    """
    status_placeholder = st.empty()

    # == STEP 1: EXTRACT entities and relationships using the LLM ==
    status_placeholder.info("1/4 - Extracting entities & relationships... 🧠")

    # Define the schema and create a detailed prompt for the LLM
    schema = {
        "node_types": ["Person", "House", "Planet"],
        "relationship_types": ["PARENT_OF", "HEIR_OF", "RULES"]
    }
    
    extract_prompt = f"""
    You are an expert at extracting a knowledge graph from a given text.
    Your task is to identify entities and relationships that match the provided schema.
    Extract the information and format it as a JSON object with two keys: "nodes" and "relationships".

    Schema:
    Node Types: {schema['node_types']}
    Relationship Types: {schema['relationship_types']}

    Rules:
    - Each node must have an 'id' (a unique string) and a 'type'.
    - Each relationship must have a 'source' (the id of the source node), a 'target' (the id of the target node), and a 'type'.
    - The source and target of a relationship must be one of the provided node types.

    Example:
    Text: "The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of House Atreides, an aristocratic family that rules the planet Caladan."
    JSON Output:
    {{
      "nodes": [
        {{ "id": "Paul Atreides", "type": "Person" }},
        {{ "id": "Leto Atreides", "type": "Person" }},
        {{ "id": "Lady Jessica", "type": "Person" }},
        {{ "id": "House Atreides", "type": "House" }},
        {{ "id": "Caladan", "type": "Planet" }}
      ],
      "relationships": [
        {{ "source": "Leto Atreides", "target": "Paul Atreides", "type": "PARENT_OF" }},
        {{ "source": "Lady Jessica", "target": "Paul Atreides", "type": "PARENT_OF" }},
        {{ "source": "Paul Atreides", "target": "House Atreides", "type": "HEIR_OF" }},
        {{ "source": "House Atreides", "target": "Caladan", "type": "RULES" }}
      ]
    }}

    Now, extract the graph from the following text:
    Text: "{text}"
    """
    
    # Get the structured data from the LLM
    extraction_response = await llm.ainvoke(extract_prompt)
    llm_output = extraction_response.content

    # Use regex to find and extract the JSON content from the markdown block
    match = re.search(r'```json\s*(.*?)\s*```', llm_output, re.DOTALL)
    
    if match:
        json_string = match.group(1)
    else:
        # If no markdown block is found, assume the whole output is the JSON
        json_string = llm_output

    try:
        graph_data = json.loads(json_string)
    except json.JSONDecodeError:
        st.error("The LLM did not return valid JSON. Please try again or adjust the text.")
        st.code(llm_output) # Show the original, uncleaned output for debugging
        return

    nodes = graph_data.get("nodes", [])
    relationships = graph_data.get("relationships", [])

    if not nodes:
        st.warning("No nodes were extracted from the text.")
        return

    # == STEP 2: GENERATE embeddings for each node ==
    # == STEP 2: GENERATE embeddings for each node ==
    status_placeholder.info("2/4 - Generating entity embeddings... 💡")
    node_texts = [node['id'] for node in nodes]

    # Create a list of embedding tasks to run concurrently
    embedding_coroutines = [embedder.embed_query(text) for text in node_texts]
    # Run all embedding tasks and gather the results
    node_embeddings = await asyncio.gather(*embedding_coroutines)

    # Add embeddings to the node data
    for node, embedding in zip(nodes, node_embeddings):
        node['embedding'] = embedding

    # == STEP 3: LOAD the graph into Neo4j ==
    status_placeholder.info("3/4 - Loading graph into Neo4j... 🚀")
    driver = AsyncGraphDatabase.driver(**driver_config)
    async with driver.session() as session:
        # Use UNWIND to create all nodes in a single, efficient transaction
        await session.run(
            """
            UNWIND $nodes AS node_data
            // Use apoc.merge.node for dynamic labels
            CALL apoc.merge.node([node_data.type], {id: node_data.id}, {embedding: node_data.embedding}, {})
            YIELD node
            RETURN count(node)
            """,
            nodes=nodes
        )

        # Use UNWIND to create all relationships in a single transaction
        # NOTE: This requires the APOC plugin in Neo4j for dynamic relationship types
        await session.run(
            """
            UNWIND $relationships AS rel_data
            MATCH (source {id: rel_data.source})
            MATCH (target {id: rel_data.target})
            // Use apoc.merge.relationship for dynamic relationship types
            CALL apoc.merge.relationship(source, rel_data.type, {}, {}, target)
            YIELD rel
            RETURN count(rel)
            """,
            relationships=relationships
        )
    await driver.close()
    
    status_placeholder.success("Graph data loaded successfully!")

    # == STEP 4: CREATE the vector index ==
    status_placeholder.info("4/4 - Creating vector index... 🔍")
    driver = AsyncGraphDatabase.driver(**driver_config)
    async with driver.session() as session:
        await session.run(
            f"""
            CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
            FOR (n) ON (n.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIM},
                `vector.similarity_function`: 'cosine'
            }} }}
            """
        )
    await driver.close()
    status_placeholder.success("Graph building complete!")


async def answer_question_with_graphrag(driver_config, question, llm, embedder):
    """
    Performs the RAG process: retrieves context from the graph and generates an answer using Ollama.
    """
    driver = AsyncGraphDatabase.driver(**driver_config)

    # 1. Embed the user's question
    question_embedding = await embedder.embed_query(question)

    # 2. Retrieve context from the graph
    cypher_query = f"""
        CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', 5, $embedding) YIELD node, score
        MATCH (node)-[r]-(neighbor)
        RETURN node.id as entity, type(r) as relationship, neighbor.id as neighbor
    """
    async with driver.session() as session:
        result = await session.run(cypher_query, embedding=question_embedding)
        context_list = [record.data() async for record in result]

    if not context_list:
        context = "No information found in the graph."
    else:
        context = "\n".join([str(item) for item in context_list])

    # 3. Generate an answer using the LLM
    prompt = f"""
    You are a helpful assistant. Use the following information from a knowledge graph
    to answer the question.

    Context from the graph:
    {context}

    Question: {question}

    Answer:
    """
    
    with st.expander("📝 View Retrieved Context & Prompt"):
        st.code(f"Context:\n{context}\n\nQuestion:\n{question}", language="text")

    response = await llm.ainvoke(prompt)
    await driver.close()
    return response.content

# --- Streamlit UI ---

st.set_page_config(page_title="GraphRAG with Neo4j & Ollama", layout="wide")
st.title("GraphRAG with Neo4j & Ollama 📈")
st.info("This app demonstrates how to build a Knowledge Graph from text and answer questions about it using local LLMs with Ollama.")

# Initialize session state
if "graph_built" not in st.session_state:
    st.session_state.graph_built = False

driver_config = {"uri": NEO4J_URI, "auth": (NEO4J_USERNAME, NEO4J_PASSWORD)}

# --- Main App Layout ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Build Knowledge Graph")
    dune_text_default = (
        "The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of House "
        "Atreides, an aristocratic family that rules the planet Caladan."
    )
    text_input = st.text_area("Enter text to build the graph from:", value=dune_text_default, height=150)
    
    if st.button("Build Graph", type="primary"):
        if not text_input:
            st.error("Please enter some text to build the graph.")
        else:
            with st.spinner("Processing..."):
                try:
                    # Instantiate models here
                    llm = OllamaLLM(model_name=OLLAMA_LLM_MODEL)
                    embedder = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
                    
                    asyncio.run(build_graph_manually(driver_config, text_input, llm, embedder))
                    st.session_state.graph_built = True
                except Exception as e:
                    st.error(f"An error occurred: {e}")

with col2:
    st.header("2. Ask a Question")
    question = st.text_input("Ask a question about the text:", "Who rules the planet Caladan?")

    if st.button("Get Answer"):
        if not st.session_state.graph_built:
            st.warning("Please build the graph first before asking a question.")
        elif not question:
            st.error("Please enter a question.")
        else:
            with st.spinner("Finding answer..."):
                try:
                    # Instantiate models here for the question-answering part
                    llm = OllamaLLM(model_name=OLLAMA_LLM_MODEL)
                    embedder = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
                    
                    answer = asyncio.run(answer_question_with_graphrag(driver_config, question, llm, embedder))
                    st.success("Answer:")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"An error occurred: {e}")