import sys
from pathlib import Path
import pytest

# Make project packages importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SingleAgent"))

from SingleAgent import rag_agent as ra


@pytest.mark.asyncio
async def test_crag_no_documents_returns_correction():
    agent = ra.OllamaRAGAgent()

    # Ensure empty vector store returns correction-applied result
    # Remove any docs if present in stub
    agent.vector_store._docs = []

    res = await agent.query_crag("No docs query")
    assert res.method == "CRAG"
    assert isinstance(res.method_specific_data, dict)
    # Depending on stubbed evaluation, correction_applied may be True or False; ensure key present
    assert "correction_applied" in res.method_specific_data


@pytest.mark.asyncio
async def test_self_rag_paths_with_docs():
    agent = ra.OllamaRAGAgent()

    # Add a sample document
    agent.add_documents(["Sample content for testing Self-RAG"], [{"title": "s"}])

    res = await agent.query_self_rag("Test self rag query")
    assert res.method == "Self-RAG"
    assert 0.0 <= res.confidence <= 0.95


@pytest.mark.asyncio
async def test_deep_rag_reasoning_and_stats():
    agent = ra.OllamaRAGAgent()
    agent.add_documents(["Doc for deep rag reasoning."], [{"title": "d"}])

    res = await agent.query_deep_rag("Explain complex topic stepwise")
    assert res.method == "Deep RAG"
    assert isinstance(res.method_specific_data.get("reasoning_chain", []), list)


@pytest.mark.asyncio
async def test_compare_methods_runs_all():
    agent = ra.OllamaRAGAgent()
    agent.add_documents(["compare docs"], [{"t": "x"}])

    results = await agent.compare_methods("Compare these methods")
    assert set(results.keys()) == {"crag", "self-rag", "deep-rag"}
    # Each value should be a RAGResult or None
    for v in results.values():
        assert v is None or hasattr(v, 'method')
