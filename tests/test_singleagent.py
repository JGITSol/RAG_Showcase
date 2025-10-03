import sys
import types
import asyncio
import pytest
import os

# Create minimal stubs for external modules before importing the target
langchain = types.ModuleType("langchain")
langchain.text_splitter = types.SimpleNamespace()

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=512, chunk_overlap=50, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    def split_text(self, text):
        # simple split by sentences for testing
        return [t.strip() for t in text.split('\n') if t.strip()]

langchain.text_splitter.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter

langchain.docstore = types.SimpleNamespace()
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}
langchain.docstore.document = types.SimpleNamespace(Document=Document)

langchain.embeddings = types.SimpleNamespace()
class OllamaEmbeddings:
    def __init__(self, base_url=None, model=None):
        self.base_url = base_url
        self.model = model
langchain.embeddings.OllamaEmbeddings = OllamaEmbeddings

langchain.vectorstores = types.SimpleNamespace()
class Chroma:
    def __init__(self, persist_directory=None, embedding_function=None, collection_name=None):
        self._docs = []
        self._collection = types.SimpleNamespace(count=lambda: len(self._docs))
    def add_documents(self, docs):
        self._docs.extend(docs)
    def similarity_search_with_score(self, query, k=5):
        # return up to k fake docs with scores
        return [(Document(f"doc for {i}: {query}", {"source": f"doc_{i}"}), 0.75) for i in range(min(k, 3))]
langchain.vectorstores.Chroma = Chroma

# Insert stubs into sys.modules to satisfy imports
sys.modules["langchain"] = langchain
sys.modules["langchain.text_splitter"] = langchain.text_splitter
sys.modules["langchain.docstore.document"] = langchain.docstore.document
sys.modules["langchain.embeddings"] = langchain.embeddings
sys.modules["langchain.vectorstores"] = langchain.vectorstores

# Minimal chromadb and requests stubs
sys.modules["chromadb"] = types.ModuleType("chromadb")
sys.modules["requests"] = types.ModuleType("requests")

# Stub for ollama
ollama = types.ModuleType("ollama")
class DummyClient:
    def __init__(self, host=None):
        self.host = host
    def list(self):
        return ["modelA"]
    def generate(self, model, prompt, options=None):
        txt = prompt.lower()
        # numeric evaluation
        if "relevance score" in txt or "rate" in txt or "relevance" in txt:
            return {"response": "0.8"}
        if "respond with exactly" in txt and "[retrieve]" in txt:
            return {"response": "[Retrieve]"}
        # default generation
        return {"response": "This is a generated answer for testing."}
ollama.Client = DummyClient
sys.modules["ollama"] = ollama

# Now import the agent module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SingleAgent')))
from rag_agent import OllamaRAGAgent, RAGResult

@pytest.mark.asyncio
async def test_add_and_stats(tmp_path):
    agent = OllamaRAGAgent()
    # add documents
    docs = ["Line one\nLine two", "Another document"]
    agent.add_documents(docs)
    stats = agent.get_knowledge_base_stats()
    assert stats["status"] == "ready"
    assert stats["collection_name"] == agent.config["vector_db"]["collection_name"]

@pytest.mark.asyncio
async def test_query_crag_and_self_and_deep():
    agent = OllamaRAGAgent()
    # small knowledge base
    agent.add_documents(["Testing document content."])

    res_crag = await agent.query_crag("What is testing?")
    assert isinstance(res_crag, RAGResult)
    assert res_crag.method == "CRAG"
    assert res_crag.confidence >= 0.0

    res_self = await agent.query_self_rag("Explain testing briefly")
    assert isinstance(res_self, RAGResult)
    assert res_self.method == "Self-RAG"

    res_deep = await agent.query_deep_rag("Analyze testing depth")
    assert isinstance(res_deep, RAGResult)
    assert res_deep.method == "Deep RAG"

@pytest.mark.asyncio
async def test_compare_methods():
    agent = OllamaRAGAgent()
    agent.add_documents(["Doc A", "Doc B", "Doc C"]) 
    results = await agent.compare_methods("How does RAG work?")
    # compare_methods returns a dict where each value is RAGResult or None
    assert isinstance(results, dict)
    for key in ["crag", "self-rag", "deep-rag"]:
        assert key in results
        # results should be not None since stubs provide responses
        assert results[key] is not None
