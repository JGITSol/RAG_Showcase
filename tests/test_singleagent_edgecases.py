import asyncio
import pytest
from types import SimpleNamespace

from SingleAgent import rag_agent as ra
from tests.helpers import FakeChroma, FakeOllamaClientText, FakeOllamaClientNumeric


def test_query_dispatch_unknown_method():
    # Create agent with patched small dependencies
    ra.OllamaEmbeddings = lambda **kw: None
    ra.Chroma = lambda **kw: FakeChroma()
    ra.ollama = type("M", (), {"Client": lambda host=None: FakeOllamaClientText(host=host)})

    agent = ra.OllamaRAGAgent()
    with pytest.raises(ValueError):
        asyncio.run(agent.query("q", method="unknown-method"))


@pytest.mark.asyncio
async def test_demo_basic_usage_runs(monkeypatch, capsys):
    # Patch heavy deps
    monkeypatch.setattr(ra, "OllamaEmbeddings", lambda **kw: None)
    monkeypatch.setattr(ra, "Chroma", lambda **kw: FakeChroma())
    monkeypatch.setattr(ra, "ollama", type("M", (), {"Client": lambda host=None: FakeOllamaClientText(host=host, response='0.8')}))

    await ra.demo_basic_usage()
    out = capsys.readouterr().out
    assert "SOTA RAG Agent" in out or "Basic Usage Demo" in out


@pytest.mark.asyncio
async def test_compare_methods_with_failure(monkeypatch):
    # Patch agent internals
    monkeypatch.setattr(ra, "OllamaEmbeddings", lambda **kw: None)
    monkeypatch.setattr(ra, "Chroma", lambda **kw: FakeChroma())
    monkeypatch.setattr(ra, "ollama", type("M", (), {"Client": lambda host=None: FakeOllamaClientText(host=host, response='0.8')}))

    agent = ra.OllamaRAGAgent()

    # Make deep_rag raise to exercise failure path
    async def raise_deep(*args, **kwargs):
        raise RuntimeError("deep failed")

    agent.query_deep_rag = raise_deep

    results = await agent.compare_methods("test")
    assert results["deep-rag"] is None


def test_get_knowledge_base_stats_error(monkeypatch):
    ra.OllamaEmbeddings = lambda **kw: None
    ra.Chroma = lambda **kw: FakeChroma()
    ra.ollama = type("M", (), {"Client": lambda host=None: FakeOllamaClientText(host=host)})

    agent = ra.OllamaRAGAgent()

    # Make underlying collection.count raise
    class BadCollection:
        def count(self):
            raise RuntimeError("db error")

    agent.vector_store._collection = BadCollection()
    stats = agent.get_knowledge_base_stats()
    assert stats.get("status") == "error"


@pytest.mark.asyncio
async def test_rerank_clamping(monkeypatch):
    monkeypatch.setattr(ra, "OllamaEmbeddings", lambda **kw: None)
    monkeypatch.setattr(ra, "Chroma", lambda **kw: FakeChroma())
    monkeypatch.setattr(ra, "ollama", type("M", (), {"Client": lambda host=None: FakeOllamaClientNumeric(host=host, numeric_response='1.5')}))

    agent = ra.OllamaRAGAgent()
    agent.ollama_client = FakeOllamaClientNumeric(numeric_response='1.5')

    # synthetic docs
    doc = SimpleNamespace(page_content="doc", metadata={'source': 's'})
    docs = [(doc, 0.2)]
    reranked = await agent.rerank_documents("q", docs)
    # ensure rerank_score clamped to <= 1.0
    assert reranked[0][1] <= 1.0
