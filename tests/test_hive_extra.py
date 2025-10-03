import sys
from pathlib import Path
import asyncio
import pytest

# Make project packages importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Hive"))

from Hive import hiverag_system as hs


@pytest.mark.asyncio
async def test_analyze_query_complexity_and_assignment():
    orch = hs.HiveOrchestrator()

    low = await orch.analyze_query_complexity("Short question?")
    assert low in ("low", "medium", "high")

    med = await orch.analyze_query_complexity("Please explain how and why this works in some detail with multiple points to consider and some reasoning terms")
    assert med in ("low", "medium", "high")

    high_q = "Explain in detail how to design a complex distributed system that handles many edge cases, analyze tradeoffs, and compare alternatives"
    high = await orch.analyze_query_complexity(high_q)
    assert high in ("low", "medium", "high")

    # Ensure assignment contains ensemble-agent always
    assign = await orch.determine_agent_assignment("Simple test", "low")
    assert "ensemble-agent" in assign


@pytest.mark.asyncio
async def test_context_manager_allocate_and_status(monkeypatch):
    cm = hs.ContextManager()
    # Monkeypatch send_message to capture allocation message
    sent = {}

    async def fake_send(receiver_id, message_type, content):
        sent['last'] = (receiver_id, message_type, content)

    cm.send_message = fake_send

    # Create a fake AgentMessage-like structure content
    await cm.allocate_context(type('M', (), {'sender_id': 'test-agent'}), ) if False else None

    # Use direct call to allocate_context through a constructed message
    msg = hs.AgentMessage(sender_id='test-agent', receiver_id='context-manager', message_type='context_request', content={'tokens': 1000})
    await cm.allocate_context(msg)
    # After allocation, context_segments should have at least one entry
    assert cm.current_usage >= 1000


def test_ensemble_agent_fuse_and_metrics():
    # Use dummy ollama client from stubs
    client = __import__('ollama')
    ea = hs.EnsembleAgent(client.Client())

    # Empty responses -> no responses available
    res = asyncio.run(ea.fuse_responses("q", {}, {}))
    assert "No responses" in res

    # Test confidence calc with empty and with values
    conf_empty = asyncio.run(ea.calculate_ensemble_confidence({}))
    assert conf_empty == 0.0

    confs = {'a': 0.5, 'b': 0.8}
    conf_val = asyncio.run(ea.calculate_ensemble_confidence(confs))
    assert 0.0 < conf_val <= 0.95

    # Consensus with single response == 1.0
    cons_single = asyncio.run(ea.calculate_consensus({'a': 'hello'}))
    assert cons_single == 1.0

    # Consensus with two responses produces a float
    cons = asyncio.run(ea.calculate_consensus({'a': 'apple orange', 'b': 'apple banana'}))
    assert 0.0 <= cons <= 1.0

    # Diversity is inverse of consensus
    div = asyncio.run(ea.calculate_diversity({'a': 'x', 'b': 'y'}))
    assert 0.0 <= div <= 1.0


@pytest.mark.asyncio
async def test_hiverag_system_add_docs_and_query_timeout():
    hive = hs.HiveRAGSystem()

    # Add documents and ensure they are registered in vector store
    sample = ["a short document"]
    hive.add_documents(sample, [{'title': 't'}])
    # Chroma stub keeps docs internally
    assert hive.vector_store._collection.count() >= 1

    # Remove orchestrator to trigger RuntimeError branch
    hive.agents.pop('hive-orchestrator', None)
    with pytest.raises(RuntimeError):
        await hive.query("Will raise because orchestrator missing")
