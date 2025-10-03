import asyncio
import pytest
from types import SimpleNamespace

from SingleAgent import rag_agent as ra
from tests.helpers import FakeChroma, FakeOllamaClientText, FakeOllamaClientNumeric


@pytest.mark.asyncio
async def test_rerank_fallback_and_stats_and_print(monkeypatch, capsys):
    # Patch embeddings and Chroma to avoid heavy deps
    monkeypatch.setattr(ra, "OllamaEmbeddings", lambda **kw: None)
    monkeypatch.setattr(ra, "Chroma", lambda **kw: FakeChroma())
    # Patch ollama.Client for initialization
    monkeypatch.setattr(ra, "ollama", type("M", (), {"Client": lambda host=None: FakeOllamaClientNumeric(host=host, numeric_response='0.8')}))

    agent = ra.OllamaRAGAgent()

    # Provide vector store docs and a docs_with_scores sample
    doc = SimpleNamespace(page_content="Important facts", metadata={'source': 'doc_1'})
    docs_with_scores = [(doc, 0.4)]

    # Create a client that raises for rerank to force fallback
    class RaisingClient:
        def list(self):
            return []
        def generate(self, **kwargs):
            raise RuntimeError("rerank failed")

    agent.ollama_client = RaisingClient()

    reranked = await agent.rerank_documents("q", docs_with_scores)
    # fallback should return original ranking when rerank fails
    assert reranked[0][1] == 0.4

    # Test get_knowledge_base_stats with a fake underlying collection
    agent.vector_store._collection = SimpleNamespace(count=lambda: 5)
    stats = agent.get_knowledge_base_stats()
    assert stats["total_documents"] == 5

    # Test print_comparison_results prints expected sections
    sample_res = ra.RAGResult(answer="ok", method="CRAG", confidence=0.5, sources=["s"], retrieval_time=0.1, generation_time=0.1, total_time=0.2, method_specific_data={})
    agent.print_comparison_results({"crag": sample_res, "self-rag": None, "deep-rag": sample_res})
    captured = capsys.readouterr()
    assert "SOTA RAG METHODS COMPARISON" in captured.out or "CRAG" in captured.out


@pytest.mark.asyncio
async def test_query_crag_evaluation_fallback(monkeypatch):
    # Patch Chroma and Ollama client; make evaluation generate throw to force fallback
    monkeypatch.setattr(ra, "OllamaEmbeddings", lambda **kw: None)
    monkeypatch.setattr(ra, "Chroma", lambda **kw: FakeChroma())
    monkeypatch.setattr(ra, "ollama", type("M", (), {"Client": lambda host=None: FakeOllamaClientNumeric(host=host, numeric_response='0.0')}))

    agent = ra.OllamaRAGAgent()
    agent.vector_store = FakeChroma()  # has synthetic docs

    # Make the ollama client generate raise for evaluation to force default score branch
    class EvalRaisingClient(FakeOllamaClientNumeric):
        def generate(self, **kwargs):
            # If prompt includes 'Rate' or 'Evaluate', raise to force fallback
            if 'Evaluate' in kwargs.get('prompt', '') or 'Rate' in kwargs.get('prompt', ''):
                raise RuntimeError("eval fail")
            return super().generate(**kwargs)

    agent.ollama_client = EvalRaisingClient()

    res = await agent.query_crag("test crag behavior")
    assert res.method == "CRAG"
    # Even if eval fails, we should get a RAGResult
    assert isinstance(res.confidence, float)
