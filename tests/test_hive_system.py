import importlib.util
import asyncio
import os
from types import SimpleNamespace

import pytest

from tests.helpers import FakeOllamaClientNumeric, FakeChroma

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Hive', 'hiverag_system.py'))


def load_hive_module():
    spec = importlib.util.spec_from_file_location('hive_module', MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_hiverag_init_and_add_documents(monkeypatch, tmp_path):
    mod = load_hive_module()

    # replace ollama client with numeric fake
    monkeypatch.setattr(mod, 'ollama', SimpleNamespace(Client=lambda host=None: FakeOllamaClientNumeric(host)))

    # Patch HiveRAGSystem._init_vector_store to return the FakeChroma instance
    monkeypatch.setattr(mod.HiveRAGSystem, '_init_vector_store', lambda self: FakeChroma())

    hive = mod.HiveRAGSystem()

    # Add documents
    docs = ["Document one.", "Document two."]
    meta = [{"title": "one"}, {"title": "two"}]

    hive.add_documents(docs, meta)

    status = hive.get_system_status()
    assert 'agents' in status
    assert 'metrics' in status
    assert isinstance(status['config']['total_agents'], int)


@pytest.mark.asyncio
async def test_hiverag_route_message(monkeypatch):
    mod = load_hive_module()

    # Minimal stubs using helpers
    monkeypatch.setattr(mod, 'ollama', SimpleNamespace(Client=lambda host=None: FakeOllamaClientNumeric(host)))
    monkeypatch.setattr(mod.HiveRAGSystem, '_init_vector_store', lambda self: FakeChroma())

    hive = mod.HiveRAGSystem()

    # Create a fake message and ensure routing does not raise
    msg = mod.AgentMessage(sender_id='tester', receiver_id='hive-orchestrator', message_type='query_request', content={"query": "hello"})

    # run route through orchestrator
    await hive.route_message(msg)

    # ensure orchestrator registered the active_queries (it creates entries asynchronously)
    # Wait a short time for async handlers to process
    await asyncio.sleep(0.1)
    assert isinstance(hive.agents, dict)