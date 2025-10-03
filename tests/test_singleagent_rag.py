import importlib.util
import os
from types import SimpleNamespace

import pytest

from tests.helpers import FakeOllamaClientNumeric, FakeChroma

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SingleAgent', 'rag_agent.py'))


def load_rag_module():
    spec = importlib.util.spec_from_file_location('rag_module', MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_crag_basic(monkeypatch):
    mod = load_rag_module()

    monkeypatch.setattr(mod, 'ollama', SimpleNamespace(Client=lambda host=None: FakeOllamaClientNumeric(host, numeric_response='0.8')))
    monkeypatch.setattr(mod, 'OllamaEmbeddings', lambda **k: None)
    monkeypatch.setattr(mod, 'Chroma', FakeChroma)

    agent = mod.OllamaRAGAgent()

    res = await agent.query_crag("What is testing?")

    assert res is not None
    assert res.method == "CRAG"
    assert isinstance(res.confidence, float)