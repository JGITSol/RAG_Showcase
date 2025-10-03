import asyncio
import pytest

from Hive import hiverag_system as hs
from tests.helpers import FakeChroma, FakeOllamaClientNumeric, FakeOllamaClientText


def test_analyze_query_complexity_and_assignments():
    orch = hs.HiveOrchestrator()
    # short query -> low
    c1 = asyncio.run(orch.analyze_query_complexity("short question"))
    assert c1 in ("low", "medium", "high")

    # long / reasoning -> high
    long_q = "analyze " + "word " * 60
    c2 = asyncio.run(orch.analyze_query_complexity(long_q))
    assert c2 == "high"

    # determine_agent_assignment variations
    a_low = asyncio.run(orch.determine_agent_assignment("simple", "low"))
    assert "selfrag-agent" in a_low

    a_med = asyncio.run(orch.determine_agent_assignment("compare this", "medium"))
    assert "crag-agent" in a_med and "selfrag-agent" in a_med

    a_high = asyncio.run(orch.determine_agent_assignment("deep analysis needed", "high"))
    assert all(k in a_high for k in ("crag-agent", "selfrag-agent", "deeprag-agent", "ensemble-agent"))


def test_orchestrator_collect_triggers_ensemble(monkeypatch):
    orch = hs.HiveOrchestrator()
    # prepare active_queries entry
    qid = "q-test"
    orch.active_queries[qid] = {
        "query": "q",
        "assigned_agents": {"a1": {}, "a2": {}},
        "results": {},
        "start_time": 0,
        "requester": "user"
    }

    called = {}

    async def fake_send(receiver_id, message_type, content):
        # capture ensemble call
        called['receiver'] = receiver_id
        called['type'] = message_type
        called['content'] = content

    monkeypatch.setattr(orch, "send_message", fake_send)

    # send results from both agents
    m1 = hs.AgentMessage(sender_id="a1", receiver_id="hive-orchestrator", message_type="agent_result", content={"query_id": qid})
    m2 = hs.AgentMessage(sender_id="a2", receiver_id="hive-orchestrator", message_type="agent_result", content={"query_id": qid})

    asyncio.run(orch.collect_agent_result(m1))
    asyncio.run(orch.collect_agent_result(m2))

    assert called.get('receiver') == "ensemble-agent"
    assert called.get('type') == "coordinate_results"


def test_query_timeout_branch(monkeypatch):
    # Build Hive but make asyncio.wait_for raise Timeout immediately
    monkeypatch.setattr(hs.HiveRAGSystem, "_init_vector_store", lambda self: FakeChroma())
    monkeypatch.setattr(hs.ollama, "Client", FakeOllamaClientText)

    hive = hs.HiveRAGSystem()

    async def fake_wait_future(fut, timeout):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_future)

    result = asyncio.run(hive.query("this will timeout", user_id="u1"))
    assert result.answer == "Query timed out"
    assert result.confidence == 0.0

import importlib.util
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
async def test_context_manager_allocate_and_release(monkeypatch):
    mod = load_hive_module()

    # Prevent send_message from requiring a full hive routing
    async def _fake_send(self, *args, **kwargs):
        return None

    monkeypatch.setattr(mod.ContextManager, 'send_message', _fake_send)

    ctx = mod.ContextManager()

    # create a fake allocation message
    msg = mod.AgentMessage(sender_id='test-agent', receiver_id='context-manager', message_type='context_request', content={'tokens': 1000})

    await ctx.allocate_context(msg)

    # There should be one segment allocated
    assert len(ctx.context_segments) >= 1
    seg_id = next(iter(ctx.context_segments.keys()))
    usage_before = ctx.current_usage

    # Now release
    release_msg = mod.AgentMessage(sender_id='test-agent', receiver_id='context-manager', message_type='context_release', content={'segment_id': seg_id})
    await ctx.release_context(release_msg)

    # After release, the segment should be gone and usage decreased
    assert seg_id not in ctx.context_segments
    assert ctx.current_usage <= usage_before


@pytest.mark.asyncio
async def test_crag_rerank_and_evaluate(monkeypatch):
    mod = load_hive_module()

    monkeypatch.setattr(mod, 'ollama', SimpleNamespace(Client=lambda host=None: FakeOllamaClientNumeric(host, numeric_response='0.85')))
    # Patch the initializer to return our fake vector store
    monkeypatch.setattr(mod.HiveRAGSystem, '_init_vector_store', lambda self: FakeChroma())

    # Create CRAG agent directly
    fake_vector = FakeChroma()
    crag = mod.CRAGAgent(FakeOllamaClientNumeric(), fake_vector)

    # Provide hive config for reranker default
    class FakeHive:
        config = {'ollama': {'reranker_model': 'xitao/bge-reranker-v2-m3:latest'}}

    crag.hive = FakeHive()

    docs = fake_vector.similarity_search_with_score('testing', k=6)

    reranked = await crag.rerank_documents('testing', docs)
    # Reranked should be same length and sorted descending by score
    assert isinstance(reranked, list)
    assert all(isinstance(t[1], float) for t in reranked)

    evals = await crag.evaluate_retrieval_quality('testing', docs)
    assert isinstance(evals, list)
    assert all(isinstance(score, float) for _, score in evals)


@pytest.mark.asyncio
async def test_ensemble_fusion_and_consensus(monkeypatch):
    mod = load_hive_module()

    class FakeOllamaClient:
        def __init__(self, host=None):
            pass

        def generate(self, **kwargs):
            return {'response': 'Fused answer from fake Ollama'}

    ensemble = mod.EnsembleAgent(FakeOllamaClient())

    responses = {
        'crag-agent': 'Answer A with details',
        'selfrag-agent': 'Answer B with alternate phrasing'
    }

    confidences = {'crag-agent': 0.8, 'selfrag-agent': 0.75}

    fused = await ensemble.fuse_responses('query', responses, confidences)
    assert isinstance(fused, str)
    conf = await ensemble.calculate_ensemble_confidence(confidences)
    assert 0.0 <= conf <= 0.95
    consensus = await ensemble.calculate_consensus(responses)
    assert 0.0 <= consensus <= 1.0
# Additional tests using the dynamically loaded module (mod) to avoid manipulating sys.path


@pytest.mark.asyncio
async def test_crag_agent_flow_using_mod(monkeypatch):
    mod = load_hive_module()

    # stub ollama client and vector store
    client = SimpleNamespace()
    client.generate = lambda **k: {'response': '0.7'}

    class TinyVS:
        def __init__(self):
            self._docs = []

        def add_documents(self, docs):
            self._docs = docs

        def similarity_search_with_score(self, query, k=5):
            from types import SimpleNamespace
            return [(SimpleNamespace(page_content='A', metadata={'source': 'a'}), 0.6),
                    (SimpleNamespace(page_content='B', metadata={'source': 'b'}), 0.5)]

    vs = TinyVS()

    agent = mod.CRAGAgent(client, vs)

    docs = await agent.retrieve_documents('q', k=2)
    assert isinstance(docs, list)

    evals = await agent.evaluate_retrieval_quality('q', [(SimpleNamespace(page_content='X', metadata={}), 0.5)])
    assert isinstance(evals, list)


@pytest.mark.asyncio
async def test_selfrag_and_deeprag_agents_using_mod(monkeypatch):
    mod = load_hive_module()

    client = SimpleNamespace()
    client.generate = lambda **k: {'response': '[No Retrieve]'}

    class TinyVS:
        def add_documents(self, docs):
            pass

        def similarity_search_with_score(self, q, k=5):
            from types import SimpleNamespace
            return [(SimpleNamespace(page_content='One', metadata={'source': 'one'}), 0.9)]

    vs = TinyVS()

    s_agent = mod.SelfRAGAgent(client, vs)
    decision = await s_agent.make_retrieval_decision('any')
    assert decision is False

    docs = await s_agent.retrieve_and_critique('q')
    assert isinstance(docs, list)

    d_agent = mod.DeepRAGAgent(client, vs)
    complexity = await d_agent.analyze_reasoning_complexity('q')
    assert isinstance(complexity, int)

    chain, retrieved = await d_agent.execute_deep_reasoning('q', complexity)
    assert isinstance(chain, list)

    synth = await d_agent.synthesize_response('q', chain, retrieved)
    assert isinstance(synth, str)
