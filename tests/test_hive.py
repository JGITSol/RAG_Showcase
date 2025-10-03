import sys
import types
import asyncio
import pytest
import os

# Minimal stubs for langchain and ollama similar to singleagent tests
langchain = types.ModuleType("langchain")
langchain.text_splitter = types.SimpleNamespace()

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    def split_text(self, text):
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
        return [(Document(f"doc for {i}: {query}", {"source": f"doc_{i}"}), 0.75) for i in range(min(k, 3))]
langchain.vectorstores.Chroma = Chroma

sys.modules["langchain"] = langchain
sys.modules["langchain.text_splitter"] = langchain.text_splitter
sys.modules["langchain.docstore.document"] = langchain.docstore.document
sys.modules["langchain.embeddings"] = langchain.embeddings
sys.modules["langchain.vectorstores"] = langchain.vectorstores

sys.modules["chromadb"] = types.ModuleType("chromadb")

ollama = types.ModuleType("ollama")
class DummyClient:
    def __init__(self, host=None):
        self.host = host
    def list(self):
        return ["modelA"]
    def generate(self, model, prompt, options=None):
        txt = prompt.lower()
        if "relevance score" in txt or "rate" in txt:
            return {"response": "0.7"}
        if "respond with exactly" in txt and "[retrieve]" in txt:
            return {"response": "[Retrieve]"}
        return {"response": "Ensemble fused answer"}
ollama.Client = DummyClient
sys.modules["ollama"] = ollama

# import the Hive system after stubbing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Hive')))
from hiverag_system import HiveRAGSystem

@pytest.mark.asyncio
async def test_hive_init_and_add_docs():
    hive = HiveRAGSystem()
    # add small docset
    hive.add_documents(["Doc A\nDoc B"], metadata=[{"title":"A"}])
    status = hive.get_system_status()
    assert status["config"]["total_agents"] >= 1

@pytest.mark.asyncio
async def test_hive_query_flow():
    hive = HiveRAGSystem()
    hive.add_documents(["Testing doc content."])
    # Run a sample query
    result = await hive.query("Explain the system")
    assert result is not None
    assert result.method == "HiveRAG Ensemble"
    assert isinstance(result.confidence, float)
