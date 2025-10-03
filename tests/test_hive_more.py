import asyncio
import pytest
from types import SimpleNamespace

from Hive import hiverag_system as hs
from tests.helpers import FakeChroma, FakeOllamaClientNumeric, FakeOllamaClientText


def test_update_agent_status_missing_content():
    orch = hs.HiveOrchestrator()
    # Missing content should return early without exception
    msg = hs.AgentMessage(sender_id="x", receiver_id="hive-orchestrator", message_type="status_update", content={})
    asyncio.run(orch.update_agent_status(msg))


def test_collect_agent_result_unknown_query():
    orch = hs.HiveOrchestrator()
    # Send result for unknown query id
    msg = hs.AgentMessage(sender_id="a1", receiver_id="hive-orchestrator", message_type="agent_result", content={"query_id": "unknown"})
    # Should not raise
    asyncio.run(orch.collect_agent_result(msg))


@pytest.mark.asyncio
async def test_crag_agent_evaluate_and_apply(monkeypatch):
    fake_vs = FakeChroma()
    fake_client = FakeOllamaClientNumeric(numeric_response='0.9', text_response='Good answer')

    agent = hs.CRAGAgent(fake_client, fake_vs)
    # Prevent real send_message routing
    sent = {}

    async def fake_send(receiver_id, message_type, content):
        sent['last'] = (receiver_id, message_type, content)

    agent.send_message = fake_send

    # Prepare task assignment message
    msg = hs.AgentMessage(sender_id="orch", receiver_id="crag-agent", message_type="task_assignment", content={"query": "test query", "query_id": "q1"})

    await agent.process_task_assignment(msg)

    # Ensure a result was sent back to orchestrator
    assert sent.get('last') is not None


@pytest.mark.asyncio
async def test_selfrag_no_retrieve(monkeypatch):
    fake_vs = FakeChroma()
    fake_client = FakeOllamaClientText(response='[No Retrieve]')
    agent = hs.SelfRAGAgent(fake_client, fake_vs)

    sent = {}

    async def fake_send(receiver_id, message_type, content):
        sent['last'] = (receiver_id, message_type, content)

    agent.send_message = fake_send

    # send task assignment
    msg = hs.AgentMessage(sender_id="u", receiver_id="selfrag-agent", message_type="task_assignment", content={"query": "short query", "query_id": "q2"})
    await agent.process_task_assignment(msg)

    assert sent.get('last') is not None


@pytest.mark.asyncio
async def test_deeprag_analyze_and_execute(monkeypatch):
    fake_vs = FakeChroma()
    # Return complexity 4 in the analysis prompt
    fake_client = FakeOllamaClientText(response='Complexity: 4')
    agent = hs.DeepRAGAgent(fake_client, fake_vs)

    # run analyze
    c = await agent.analyze_reasoning_complexity("some complex query")
    assert isinstance(c, int)

    # execute deep reasoning
    chain, docs = await agent.execute_deep_reasoning("q", 4)
    assert isinstance(chain, list)
