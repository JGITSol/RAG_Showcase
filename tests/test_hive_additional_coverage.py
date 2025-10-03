import importlib.util
import os
import asyncio
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
async def test_orchestrator_complexity_and_assignment():
    mod = load_hive_module()
    orch = mod.HiveOrchestrator()

    low = await orch.analyze_query_complexity('short question')
    assert low == 'low'

    medium = await orch.analyze_query_complexity('this query has enough words to be medium complexity with analyze term')
    assert medium in ('medium', 'high')

    high = await orch.analyze_query_complexity('Explain and analyze why this complex multi-part question requires deep reasoning and comparison across many concepts ' * 5)
    assert high == 'high'

    assign_low = await orch.determine_agent_assignment('short', 'low')
    assert 'selfrag-agent' in assign_low


@pytest.mark.asyncio
async def test_context_optimize_and_release(monkeypatch):
    mod = load_hive_module()
    # small window to trigger optimization
    ctx = mod.ContextManager()
    ctx.max_context_window = 1000

    # monkeypatch send_message to no-op
    async def _send(self, *a, **k):
        return None

    monkeypatch.setattr(mod.ContextManager, 'send_message', _send)

    # allocate several small segments
    for i in range(6):
        msg = mod.AgentMessage(sender_id=f'a{i}', receiver_id='context-manager', message_type='context_request', content={'tokens': 300})
        await ctx.allocate_context(msg)

    before = len(ctx.context_segments)
    await ctx.optimize_context_allocation()
    after = len(ctx.context_segments)
    assert after <= before


@pytest.mark.asyncio
async def test_ensemble_coordinate_and_send(monkeypatch):
    mod = load_hive_module()
    fake_client = FakeOllamaClientNumeric()
    ensemble = mod.EnsembleAgent(fake_client)

    captured = {}

    async def fake_send(self, receiver_id, message_type, content):
        captured['msg'] = (receiver_id, message_type, content)

    monkeypatch.setattr(mod.EnsembleAgent, 'send_message', fake_send)

    # prepare agent_results with two agents
    query_data = {
        'query': 'q',
        'start_time': 0,
        'requester': 'u',
        'results': {
            'crag-agent': {'response': 'A', 'confidence': 0.8, 'method': 'CRAG', 'sources': ['s1'], 'processing_time': 0.1},
            'selfrag-agent': {'response': 'B', 'confidence': 0.7, 'method': 'Self-RAG', 'sources': ['s2'], 'processing_time': 0.2}
        }
    }

    msg = mod.AgentMessage(sender_id='tester', receiver_id='ensemble-agent', message_type='coordinate_results', content={'query_data': query_data})
    await ensemble.coordinate_ensemble_results(msg)

    assert 'msg' in captured
    # final_result should be sent to requester 'u'
    receiver_id, mtype, content = captured['msg']
    assert mtype == 'final_result'
    assert 'result' in content


@pytest.mark.asyncio
async def test_hive_query_timeout(monkeypatch):
    mod = load_hive_module()
    monkeypatch.setattr(mod, 'ollama', SimpleNamespace(Client=lambda host=None: FakeOllamaClientNumeric(host)))
    monkeypatch.setattr(mod.HiveRAGSystem, '_init_vector_store', lambda self: FakeChroma())

    hive = mod.HiveRAGSystem()

    # Simulate immediate timeout by making asyncio.wait_for raise
    orig_wait = asyncio.wait_for
    def _raise_timeout(fut, timeout):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(asyncio, 'wait_for', _raise_timeout)

    res = await hive.query('q')
    assert res.confidence == 0.0

    monkeypatch.setattr(asyncio, 'wait_for', orig_wait)
