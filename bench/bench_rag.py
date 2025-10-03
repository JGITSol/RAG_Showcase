r"""Benchmark runner for HiveRAG and SingleAgent RAG implementations.

This script uses the lightweight fakes in `tests/helpers.py` to run a controlled
benchmark locally (no external network). It times stages: retrieval, rerank,
evaluation, generation, and aggregation; and computes simple retrieval metrics
(Recall@1, Recall@5, MRR) using synthetic ground-truth annotations.

To run:
  D:\REPOS\RAG_Showcase\.venv\Scripts\Activate.ps1
  python .\bench\bench_rag.py

If you want to benchmark against real datasets (NarrativeQA/HotpotQA/etc.),
provide a dataset loader and set `USE_REAL_DATA = True` — network/IO will be required.
"""
import asyncio
import time
import sys
from types import SimpleNamespace
import importlib
import importlib.util
import os
import types
import argparse

# CLI: choose mode 'fake' (no heavy deps) or 'real' (use installed libs)
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['fake', 'real'], default='real', help='Benchmark mode: fake (fast) or real (uses installed libs)')
ARGS, _ = parser.parse_known_args()
MODE = ARGS.mode

# Ensure repo root is on sys.path so imports like 'Hive.hiverag_system' work
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

if MODE == 'fake':
    # Lightweight stubs for heavy optional deps so we can import modules for benchmarking
    def _make_stub(name):
        m = types.ModuleType(name)
        return m

    # stub langchain submodules used by repo
    sys.modules.setdefault('langchain', _make_stub('langchain'))
    sys.modules.setdefault('langchain.text_splitter', _make_stub('langchain.text_splitter'))
    sys.modules.setdefault('langchain.docstore', _make_stub('langchain.docstore'))
    sys.modules.setdefault('langchain.docstore.document', _make_stub('langchain.docstore.document'))
    sys.modules.setdefault('langchain.vectorstores', _make_stub('langchain.vectorstores'))
    sys.modules.setdefault('langchain.embeddings', _make_stub('langchain.embeddings'))

    # Minimal Document and RecursiveCharacterTextSplitter placeholders
    _SN = SimpleNamespace
    class _Doc:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    class _Splitter:
        def __init__(self, chunk_size=512, chunk_overlap=50, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        def split_text(self, text):
            # naive split by chunk_size
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    setattr(sys.modules['langchain.docstore.document'], 'Document', _Doc)
    setattr(sys.modules['langchain.text_splitter'], 'RecursiveCharacterTextSplitter', _Splitter)

    # stub ollama
    sys.modules.setdefault('ollama', _make_stub('ollama'))
    setattr(sys.modules['ollama'], 'Client', lambda host=None: _SN(list=lambda: [], generate=lambda **k: {'response': '0.5'}))

    # Minimal Chroma stub compatible with _init_vector_store usage
    class _ChromaStub:
        def __init__(self, persist_directory=None, embedding_function=None, collection_name=None, docs=None):
            # keep a simple in-memory list
            self._docs = docs or []
        def add_documents(self, docs):
            self._docs.extend(docs)
        def similarity_search_with_score(self, query, k=5):
            # return doc metadata sources as in FakeChroma
            if self._docs:
                out = []
                for i, d in enumerate(self._docs[:k]):
                    out.append((d, 0.5 + 0.1 * (i % 3)))
                return out
            return []

    setattr(sys.modules['langchain.vectorstores'], 'Chroma', _ChromaStub)

# Load fakes module by path to support running as a script
helpers_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'helpers.py')
helpers_path = os.path.abspath(helpers_path)
spec = importlib.util.spec_from_file_location('tests.helpers', helpers_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load helpers from {helpers_path}")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

# fakes
FakeOllamaClientText = helpers.FakeOllamaClientText
FakeChroma = helpers.FakeChroma


def make_synthetic_corpus(queries, docs_per_query=5, long_size=2000):
    """Create a synthetic long-context corpus where each query has one guaranteed relevant doc.
    Returns list of SimpleNamespace documents with metadata indicating relevance.
    """
    docs = []
    for qidx, q in enumerate(queries):
        # Add one long relevant doc
        text = (f"Relevant for: {q} " + (" lorem ipsum " * (long_size // 10))).strip()
        meta = {"source": f"q{qidx}_relevant", "relevant_for": [q]}
        docs.append(SimpleNamespace(page_content=text, metadata=meta))

        # Add distractors
        for j in range(docs_per_query - 1):
            text = (f"Distractor {j} for {q} " + (" lorem ipsum " * (long_size // 20))).strip()
            meta = {"source": f"q{qidx}_d{j}", "relevant_for": []}
            docs.append(SimpleNamespace(page_content=text, metadata=meta))

    # Add some general docs
    for k in range(5):
        text = ("General knowledge " + (" lorem ipsum " * (long_size // 30))).strip()
        meta = {"source": f"general_{k}", "relevant_for": []}
        docs.append(SimpleNamespace(page_content=text, metadata=meta))

    return docs


def _ensure_phi4_config(cfg):
    """Force use of phi4-mini as the primary model for benchmark runs."""
    try:
        if isinstance(cfg.get('ollama'), dict):
            cfg['ollama']['primary_model'] = 'phi4-mini'
            # keep embeddings/reranker as configured if present
    except Exception:
        pass


def recall_mrr_for_result(retrieved_sources, positive_source):
    """Compute Recall@1, Recall@5, and MRR for a single query given ordered retrieved sources."""
    top1 = 1.0 if retrieved_sources and retrieved_sources[0] == positive_source else 0.0
    top5 = 1.0 if positive_source in retrieved_sources[:5] else 0.0
    mrr = 0.0
    for rank, s in enumerate(retrieved_sources, start=1):
        if s == positive_source:
            mrr = 1.0 / rank
            break
    return top1, top5, mrr


async def bench_singleagent(agent_module, queries, vector_docs):
    """Benchmark the SingleAgent implementation. Returns per-query metrics and timings."""
    # construct agent bypassing heavy __init__ by creating instance via __new__ and setting attrs
    AgentClass = agent_module.OllamaRAGAgent
    agent = object.__new__(AgentClass)
    agent.config = agent_module.OllamaRAGAgent._load_config(agent, None)
    # force phi4-mini for fairness / best available small model
    _ensure_phi4_config(agent.config)
    # set fake clients and vector store
    # choose client based on mode
    if MODE == 'fake':
        agent.ollama_client = FakeOllamaClientText(response="fake answer from singleagent")
    else:
        import ollama
        try:
            agent.ollama_client = ollama.Client(host=agent.config["ollama"]["base_url"])
        except Exception as e:
            # fallback to fake but warn
            print(f"Warning: could not connect to Ollama client: {e}; falling back to fake client for SingleAgent")
            agent.ollama_client = FakeOllamaClientText(response="fake answer from singleagent")
    agent.embeddings = None
    agent.vector_store = FakeChroma(docs=vector_docs)
    # set configs
    agent.crag_config = agent.config["rag_methods"]["crag"]
    agent.self_rag_config = agent.config["rag_methods"]["self_rag"]
    agent.deep_rag_config = agent.config["rag_methods"]["deep_rag"]

    results = []

    for q in queries:
        # Run each method and collect timings reported by the agent
        row = {"query": q}
        q_start = time.time()
        for method in ["crag", "self-rag", "deep-rag"]:
            res = await getattr(agent, f"query_{method.replace('-', '_')}")(q)

            # retrieval_time and generation_time are provided inside RAGResult
            retrieved_sources = res.sources or []
            # determine positive source from our synthetic scheme
            # if any of configured vector_docs metadata has relevant_for containing q
            positive = None
            for d in vector_docs:
                if q in d.metadata.get("relevant_for", []):
                    positive = d.metadata.get("source")
                    break

            top1, top5, mrr = recall_mrr_for_result(retrieved_sources, positive)

            row[method] = {
                "total_time": res.total_time,
                "retrieval_time": res.retrieval_time,
                "generation_time": res.generation_time,
                "confidence": res.confidence,
                "recall@1": top1,
                "recall@5": top5,
                "mrr": mrr,
                "sources": retrieved_sources
            }

        # total wall-clock for all methods for this query
        row['total_time'] = time.time() - q_start
        results.append(row)

    return results


async def bench_hive(hive_module, queries, vector_docs):
    """Benchmark the Hive implementation. Returns per-query metrics and timings.

    We instrument agent retrieval/generation functions by wrapping them to collect per-stage timings.
    """
    SystemClass = hive_module.HiveRAGSystem
    hive = object.__new__(SystemClass)
    hive.config = SystemClass._load_config(hive, None)
    # force phi4-mini
    _ensure_phi4_config(hive.config)
    # ensure fields that __init__ would have set are present when bench constructs via __new__
    if not getattr(hive, 'system_metrics', None):
        hive.system_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_response_time": 0.0,
            "ensemble_accuracy": 0.0
        }
    # attach fakes
    if MODE == 'fake':
        hive.ollama_client = FakeOllamaClientText(response="fake answer from hive")
    else:
        import ollama
        try:
            hive.ollama_client = ollama.Client(host=hive.config['ollama']['base_url'])
        except Exception as e:
            print(f"Warning: could not connect to Ollama client for Hive: {e}; falling back to fake client")
            hive.ollama_client = FakeOllamaClientText(response="fake answer from hive")
    hive.vector_store = FakeChroma(docs=vector_docs)

    # init agents (this will create agent objects and link hive ref)
    SystemClass._init_agents(hive)

    # instrument worker agents to record internal stage timings
    def wrap_async_method(obj, name):
        orig = getattr(obj, name, None)
        if not orig:
            return

        async def wrapper(*args, **kwargs):
            t0 = time.time()
            res = await orig(*args, **kwargs)
            t1 = time.time()
            # store last call duration on agent
            setattr(obj, f"_last_{name}_time", t1 - t0)
            return res

        setattr(obj, name, wrapper)

    # Wrap key methods for CRAG, SelfRAG, DeepRAG and ensemble
    for agent_id in ["crag-agent", "selfrag-agent", "deeprag-agent", "ensemble-agent"]:
        agent = hive.agents.get(agent_id)
        if not agent:
            continue
        if agent_id == "crag-agent":
            wrap_async_method(agent, "retrieve_documents")
            wrap_async_method(agent, "rerank_documents")
            wrap_async_method(agent, "evaluate_retrieval_quality")
            wrap_async_method(agent, "generate_response")
        if agent_id == "selfrag-agent":
            wrap_async_method(agent, "make_retrieval_decision")
            wrap_async_method(agent, "retrieve_and_critique")
            wrap_async_method(agent, "generate_self_rag_response")
        if agent_id == "deeprag-agent":
            wrap_async_method(agent, "analyze_reasoning_complexity")
            wrap_async_method(agent, "execute_deep_reasoning")
            wrap_async_method(agent, "synthesize_response")
        if agent_id == "ensemble-agent":
            wrap_async_method(agent, "fuse_responses")
            wrap_async_method(agent, "calculate_ensemble_confidence")

    results = []

    for q in queries:
        # ensure hive has a fresh active_queries state
        start = time.time()
        # send query and await result
        try:
            res = await hive.query(q)
        except Exception as e:
            print(f"Hive query failed: {e}")
            res = None

        total = time.time() - start

        # Determine positive source
        positive = None
        for d in vector_docs:
            if q in d.metadata.get("relevant_for", []):
                positive = d.metadata.get("source")
                break

        # collect per-agent timings from attributes set by wrappers
        agent_stage_times = {}
        for aid in ["crag-agent", "selfrag-agent", "deeprag-agent", "ensemble-agent"]:
            a = hive.agents.get(aid)
            if not a:
                continue
            times = {}
            for attr in dir(a):
                if attr.startswith("_last_") and attr.endswith("_time"):
                    times[attr.replace("_last_", "").replace("_time", "")] = getattr(a, attr)
            agent_stage_times[aid] = times

        # assemble retrieved_sources from ensemble result if available
        retrieved_sources = []
        if res:
            try:
                retrieved_sources = res.sources or []
            except Exception:
                retrieved_sources = []

        top1, top5, mrr = recall_mrr_for_result(retrieved_sources, positive)

        results.append({
            "query": q,
            "total_time": total,
            "ensemble_processing_time": getattr(res, "processing_time", None) if res else None,
            "retrieved_sources": retrieved_sources,
            "recall@1": top1,
            "recall@5": top5,
            "mrr": mrr,
            "agent_stage_times": agent_stage_times
        })

    return results


def summarize(results, label):
    print("\n" + "=" * 80)
    print(f"Benchmark summary for: {label}")
    print("=" * 80)
    def _safe_float(val):
        """Try to coerce val to float, return None on failure."""
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            # try to extract a decimal via regex
            import re as _re
            try:
                m = _re.search(r"([0-9]*\.?[0-9]+)", str(val))
                if m:
                    return float(m.group(1))
            except Exception:
                pass
        return None

    def _fmt_metric(val, fmt="{:.2f}"):
        f = _safe_float(val)
        if f is None:
            return "N/A"
        try:
            return fmt.format(f)
        except Exception:
            # fallback
            return str(f)

    # per-query rows
    for r in results:
        q = r.get('query', '<unknown>')
        print(f"Query: {q}")
        tt = r.get('total_time')
        if tt is not None:
            try:
                print(f"  Total time: {float(tt):.3f}s")
            except Exception:
                print(f"  Total time: {tt}")
        else:
            # older structure: compute approx from method totals if available
            method_times = []
            for k, v in r.items():
                if isinstance(v, dict):
                    t = v.get('total_time')
                    try:
                        if t is not None:
                            method_times.append(float(t))
                    except Exception:
                        continue
            if method_times:
                print(f"  Approx total (sum of methods): {sum(method_times):.3f}s")
            else:
                print("  Total time: N/A")

        # ensemble time (Hive)
        eproc = r.get('ensemble_processing_time')
        if eproc is not None:
            try:
                print(f"  Ensemble processing_time: {float(eproc):.3f}s")
            except Exception:
                print(f"  Ensemble processing_time: {eproc}")

        # Detect SingleAgent style (methods nested) vs Hive style (flat metrics)
        if any(isinstance(v, dict) for v in r.values()):
            # SingleAgent per-method breakdown
            for method in ["crag", "self-rag", "deep-rag"]:
                m = r.get(method)
                if not m or not isinstance(m, dict):
                    continue
                recall1 = _fmt_metric(m.get('recall@1'))
                recall5 = _fmt_metric(m.get('recall@5'))
                mrr = _fmt_metric(m.get('mrr'), "{:.3f}")
                total_m = m.get('total_time')
                total_m_str = f"{float(total_m):.3f}s" if total_m is not None else "N/A"
                conf = _fmt_metric(m.get('confidence'))
                sources = m.get('sources') or []
                if isinstance(sources, (list, tuple)):
                    preview = sources[:5]
                else:
                    preview = str(sources)[:120]
                print(f"  Method: {method}")
                print(f"    Total time: {total_m_str}")
                print(f"    Confidence: {conf}")
                print(f"    Recall@1: {recall1}, Recall@5: {recall5}, MRR: {mrr}")
                print(f"    Retrieved sources (preview): {preview}")
        else:
            # Hive flat metrics
            recall1 = _fmt_metric(r.get('recall@1'))
            recall5 = _fmt_metric(r.get('recall@5'))
            mrr = _fmt_metric(r.get('mrr'), "{:.3f}")
            print(f"  Recall@1: {recall1}, Recall@5: {recall5}, MRR: {mrr}")
            retrieved = r.get('retrieved_sources') or []
            try:
                preview = retrieved[:5]
            except Exception:
                preview = str(retrieved)[:120]
            print(f"  Retrieved sources (preview): {preview}")

        if r.get('agent_stage_times'):
            print("  Agent stage times:")
            for aid, times in r['agent_stage_times'].items():
                print(f"    {aid}: {times}")
        print("-" * 40)


async def main():
    # queries representative of long-context tasks
    queries = [
        "Analyze the historical evolution of the concept of sovereignty across multiple political systems and its implications for modern federalism.",
        "Compare and contrast the mechanisms of action, efficacy, and known side effects of the top five antiviral strategies used in pandemic response.",
        "Provide a detailed plan for migrating a monolithic legacy application to a distributed microservices architecture preserving transactional integrity."
    ]

    # Create synthetic long-doc corpus
    vector_docs = make_synthetic_corpus(queries, docs_per_query=6, long_size=3000)

    # clear reranker failures log for fresh run
    try:
        with open(os.path.join(repo_root, 'reranker_raw_failures.log'), 'w', encoding='utf-8') as _f:
            _f.write('')
    except Exception:
        pass

    # Import modules
    hive_module = importlib.import_module('Hive.hiverag_system')
    single_module = importlib.import_module('SingleAgent.rag_agent')

    print("Running SingleAgent benchmark...")
    sa_res = await bench_singleagent(single_module, queries, vector_docs)
    summarize(sa_res, "SingleAgent SOTA RAG")

    print("\nRunning Hive benchmark...")
    hive_res = await bench_hive(hive_module, queries, vector_docs)
    summarize(hive_res, "Hive Multi-Agent RAG")


if __name__ == '__main__':
    asyncio.run(main())
