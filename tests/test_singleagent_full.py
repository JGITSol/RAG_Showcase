import asyncio
import pytest

from SingleAgent import rag_agent as ra
from tests.helpers import FakeChroma, FakeOllamaClientText, FakeOllamaClientNumeric


@pytest.mark.asyncio
async def test_singleagent_crag_and_no_docs(monkeypatch):
    # Patch vector store to be empty and Ollama client to return numeric evals
    monkeypatch.setattr(ra, "Chroma", lambda **kw: FakeChroma(docs=[]))
    monkeypatch.setattr(ra, "OllamaEmbeddings", lambda **kw: None)
    # Patch ollama.Client to return our fake numeric client
    monkeypatch.setattr(ra, "ollama", type("M", (), {"Client": lambda host=None: FakeOllamaClientNumeric(host=host, numeric_response='0.0')}))

    # Create agent with patched internals
    agent = ra.OllamaRAGAgent()

    # Force vector store to be empty
    agent.vector_store = FakeChroma(docs=[])

    result = await agent.query_crag("No docs available test")
    assert result is not None
    assert result.method == "CRAG"
    assert result.method_specific_data.get("correction_applied") is True


@pytest.mark.asyncio
async def test_singleagent_self_and_deep_and_compare(monkeypatch):
    # Patch embeddings and vector store
    monkeypatch.setattr(ra, "OllamaEmbeddings", lambda **kw: None)
    monkeypatch.setattr(ra, "Chroma", lambda **kw: FakeChroma())
    # Patch ollama.Client to return deterministic text responses
    monkeypatch.setattr(ra, "ollama", type("M", (), {"Client": lambda host=None: FakeOllamaClientText(host=host, response='[Retrieve]')}))

    # Create agent and override clients directly with fakes
    agent = ra.OllamaRAGAgent()
    agent.ollama_client = FakeOllamaClientText(response='[Retrieve]')
    agent.vector_store = FakeChroma()

    # Self-RAG path where retrieval decision returns True
    res_self = await agent.query_self_rag("Is retrieval needed?")
    assert res_self.method == "Self-RAG"

    # Deep RAG path - exercising reasoning and retrieval points
    agent.ollama_client = FakeOllamaClientText(response='Complexity: 3')
    res_deep = await agent.query_deep_rag("Deep reasoning test")
    assert res_deep.method == "Deep RAG"

    # Compare methods concurrently
    # Patch first to use our fake clients for each method call
    agent.ollama_client = FakeOllamaClientText(response='0.8')
    agent.vector_store = FakeChroma()
    results = await agent.compare_methods("Compare methods test")
    assert isinstance(results, dict)
    assert set(results.keys()) == {"crag", "self-rag", "deep-rag"}
import sys
import os
import asyncio
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SingleAgent')))

from rag_agent import OllamaRAGAgent, RAGResult


@pytest.mark.asyncio
async def test_agent_add_documents_and_stats():
    agent = OllamaRAGAgent()
    # Add empty list should be handled gracefully
    agent.add_documents([])

    # Add some real docs
    docs = ["Doc A content.", "Doc B content."]
    agent.add_documents(docs)
    stats = agent.get_knowledge_base_stats()
    assert stats["status"] == "ready"


@pytest.mark.asyncio
async def test_rerank_and_query_routes():
    agent = OllamaRAGAgent()
    agent.add_documents(["Alpha", "Beta", "Gamma"]) 

    # Directly call rerank on top of vector store results
    results = agent.vector_store.similarity_search_with_score("Alpha", k=3)
    reranked = await agent.rerank_documents("Alpha", results)
    assert isinstance(reranked, list)

    # Test query entrypoints
    res_crag = await agent.query_crag("What is alpha?")
    assert isinstance(res_crag, RAGResult)
    res_self = await agent.query_self_rag("Explain beta")
    assert isinstance(res_self, RAGResult)
    res_deep = await agent.query_deep_rag("Analyze gamma")
    assert isinstance(res_deep, RAGResult)


@pytest.mark.asyncio
async def test_compare_methods_async_and_fallback(monkeypatch):
    agent = OllamaRAGAgent()
    agent.add_documents(["A1", "B1"])

    # Simulate one method raising to ensure compare captures None for that method
    async def _raise_query(query, method="crag"):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent, "query", _raise_query)

    results = await agent.compare_methods("x")
    assert isinstance(results, dict)
    # All should be None due to monkeypatch
    assert all(v is None for v in results.values())
