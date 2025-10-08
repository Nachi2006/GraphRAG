# utils/llm_utils.py
import os
from neo4j_graphrag.llm import OllamaLLM

def get_llm():
    model = os.getenv("OLLAMA_LLM_MODEL")
    if not model:
        raise ValueError("OLLAMA_LLM_MODEL not set in .env")
    llm = OllamaLLM(model=model)
    # We assume the OllamaLLM is callable: llm(prompt) -> str
    return llm
