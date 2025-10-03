import importlib.util
import os
from types import SimpleNamespace

import pytest

from tests.helpers import FakeOllamaClientNumeric, FakeChroma, EmptyChroma

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SingleAgent', 'rag_agent.py'))


def load_rag_module():
    spec = importlib.util.spec_from_file_location('rag_module', MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_rerank_documents_and_crag_no_docs(monkeypatch):
    mod = load_rag_module()

    monkeypatch.setattr(mod, 'ollama', SimpleNamespace(Client=lambda host=None: FakeOllamaClientNumeric(host, numeric_response='0.9')))
    monkeypatch.setattr(mod, 'OllamaEmbeddings', lambda **k: None)
    monkeypatch.setattr(mod, 'Chroma', EmptyChroma)

    agent = mod.OllamaRAGAgent()

    # With empty vector store, CRAG should return fallback result
    res = await agent.query_crag('no docs')
    assert res.method == 'CRAG'
    assert res.confidence == 0.0


@pytest.mark.asyncio
async def test_self_rag_and_compare(monkeypatch):
    mod = load_rag_module()

    monkeypatch.setattr(mod, 'ollama', SimpleNamespace(Client=lambda host=None: FakeOllamaClientNumeric(host, numeric_response='0.7', text_response='[No Retrieve]')))
    monkeypatch.setattr(mod, 'OllamaEmbeddings', lambda **k: None)
    monkeypatch.setattr(mod, 'Chroma', FakeChroma)

    agent = mod.OllamaRAGAgent()

    self_res = await agent.query_self_rag('q')
    assert self_res.method == 'Self-RAG'

    # compare methods should return dict with keys
    comp = await agent.compare_methods('compare test')
    assert isinstance(comp, dict)
    assert 'crag' in comp and 'self-rag' in comp and 'deep-rag' in comp
