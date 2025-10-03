import asyncio
import pytest

from Hive import hiverag_system as hs
from tests.helpers import FakeChroma, FakeOllamaClientText


@pytest.mark.asyncio
async def test_hive_query_end_to_end(monkeypatch):
    # Use fake vector store and fake ollama client for deterministic behavior
    monkeypatch.setattr(hs.HiveRAGSystem, "_init_vector_store", lambda self: FakeChroma())
    monkeypatch.setattr(hs.ollama, "Client", FakeOllamaClientText)

    hive = hs.HiveRAGSystem(config_path=None)

    # Single simple query should complete and return a QueryResult
    result = await hive.query("Explain HiveRAG architecture briefly", user_id="tester")

    assert result is not None
    assert hasattr(result, "answer")
    assert result.method == "HiveRAG Ensemble"


@pytest.mark.asyncio
async def test_hive_demo_and_context_management(monkeypatch):
    # Demo runs the full system; patch vector store and ollama client
    monkeypatch.setattr(hs.HiveRAGSystem, "_init_vector_store", lambda self: FakeChroma())
    monkeypatch.setattr(hs.ollama, "Client", FakeOllamaClientText)

    # Run demo which uses add_documents, queries and status printing
    await hs.demo_hiverag()


def test_context_manager_allocation_and_release(monkeypatch):
    # Test ContextManager allocate/release and optimize flow
    cm = hs.ContextManager()

    # Replace send_message to simply record messages instead of routing
    called = {}

    async def fake_send(agent_id, message_type, content):
        called['last'] = (agent_id, message_type, content)

    cm.send_message = fake_send

    # Allocate a segment
    msg = hs.AgentMessage(sender_id="agent-x", receiver_id="context-manager", message_type="context_request", content={"tokens": 1000})
    asyncio.run(cm.receive_message(msg))

    # There should be at least one context segment
    assert len(cm.context_segments) >= 1

    # Release it and ensure usage decreases
    seg_id = next(iter(cm.context_segments.keys()))
    release_msg = hs.AgentMessage(sender_id="agent-x", receiver_id="context-manager", message_type="context_release", content={"segment_id": seg_id})
    asyncio.run(cm.receive_message(release_msg))

    # After release, segment should be gone
    assert seg_id not in cm.context_segments
import sys
import os
import asyncio
import pytest

# Ensure Hive package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Hive')))

from hiverag_system import (
    HiveRAGSystem,
    AgentMessage,
    ContextManager,
    CRAGAgent,
    SelfRAGAgent,
    DeepRAGAgent,
    EnsembleAgent,
)


@pytest.mark.asyncio
async def test_orchestrator_analyze_and_assignment():
    hive = HiveRAGSystem()
    orch = hive.agents["hive-orchestrator"]

    low = await orch.analyze_query_complexity("What is RAG?")
    assert low in ("low", "medium", "high")

    high = await orch.analyze_query_complexity("""Explain in detail and compare several methods, analyze, evaluate and provide reasoning steps that exceed twenty words.""")
    assert high in ("low", "medium", "high")

    assign_low = await orch.determine_agent_assignment("What is RAG?", "low")
    assert "selfrag-agent" in assign_low

    assign_high = await orch.determine_agent_assignment("Explain and analyze deeply", "high")
    assert "crag-agent" in assign_high and "deeprag-agent" in assign_high


@pytest.mark.asyncio
async def test_context_manager_allocate_optimize(monkeypatch):
    cm = ContextManager()
    # make max small so allocation triggers optimization quickly
    cm.max_context_window = 100

    # fill usage to near limit
    cm.current_usage = 90

    # Create a dummy message to request more tokens
    msg = AgentMessage(sender_id="agent-x", content={"tokens": 20})

    # allocate_context should call optimize_context_allocation when over limit
    await cm.allocate_context(msg)
    # ensure a segment was allocated
    assert any(s["agent_id"] == "agent-x" for s in cm.context_segments.values())


@pytest.mark.asyncio
async def test_crag_self_deep_and_ensemble_flow():
    hive = HiveRAGSystem()
    # use provided stubs from conftest for ollama and vectorstore
    crag = CRAGAgent(hive.ollama_client, hive.vector_store)
    crag.set_hive(hive)

    # Add a dummy document to vector store so retrieval returns something
    hive.vector_store.add_documents([type("D", (), {"page_content": "doc text", "metadata": {"source": "s1"}})()])

    docs = await crag.retrieve_documents("test query")
    assert isinstance(docs, list)

    evals = await crag.evaluate_retrieval_quality("test query", docs)
    assert isinstance(evals, list)

    corrected, applied = await crag.apply_corrections("test query", docs, evals)
    assert isinstance(corrected, list)

    resp = await crag.generate_response("test query", corrected, applied)
    assert isinstance(resp, str)

    # SelfRAG
    selfrag = SelfRAGAgent(hive.ollama_client, hive.vector_store)
    selfrag.set_hive(hive)
    decision = await selfrag.make_retrieval_decision("simple question")
    assert isinstance(decision, bool)

    # DeepRAG
    deep = DeepRAGAgent(hive.ollama_client, hive.vector_store)
    deep.set_hive(hive)
    complexity = await deep.analyze_reasoning_complexity("analyze this deeply")
    assert isinstance(complexity, int)

    reasoning_chain, docs_retrieved = await deep.execute_deep_reasoning("query", complexity)
    assert isinstance(reasoning_chain, list)

    syn = await deep.synthesize_response("query", reasoning_chain, docs_retrieved)
    assert isinstance(syn, str)

    # Ensemble
    ens = EnsembleAgent(hive.ollama_client)
    fused = await ens.fuse_responses("q", {"a": "ans1", "b": "ans2"}, {"a": 0.6, "b": 0.7})
    assert isinstance(fused, str)
    conf = await ens.calculate_ensemble_confidence({"a": 0.6, "b": 0.7})
    assert isinstance(conf, float)
    cons = await ens.calculate_consensus({"a": "hello world", "b": "hello everyone"})
    assert isinstance(cons, float)
    div = await ens.calculate_diversity({"a": "x", "b": "y"})
    assert isinstance(div, float)


@pytest.mark.asyncio
async def test_hiverag_query_timeout(monkeypatch):
    hive = HiveRAGSystem()

    # Patch asyncio.wait_for to immediately raise TimeoutError to hit timeout branch
    async def _fake_wait(fut, timeout):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(asyncio, "wait_for", _fake_wait)

    res = await hive.query("will timeout")
    assert res.method == "HiveRAG Ensemble"
    assert res.confidence == 0.0
