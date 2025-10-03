# RAG_Showcase

## Overview
RAG_Showcase is a multi-agent, modular Retrieval-Augmented Generation (RAG) benchmark and demo suite. It features two main systems:
- **HiveRAG**: Multi-agent, ensemble-based RAG pipeline with robust reranking and long-context support.
- **SingleAgent**: Minimalist RAG agent for baseline and ablation studies.

## Key Features
- Modernized RAG implementations (HiveRAG, SingleAgent)
- Configurable model selection (Ollama, Chroma, LangChain)
- Robust reranker integration (default: `bge-large:latest`)
- Automated benchmarking with per-stage timings and raw response logging
- Test suite for coverage and regression
- Modular architecture for easy extension

## Current Setup
- **Python 3.11+**
- **Ollama** local server for model inference
- **Chroma** vector DB for document retrieval
- **LangChain** (optional, auto-detect)
- **Default reranker**: `bge-large:latest` (best benchmarked)
- **Benchmarks**: NDCG, MRR, MAP, Precision@K, Recall@K, parse_rate, latency
- **Raw responses**: logged in `bench/*.ndjson` for diagnostics
- **.gitignore**: excludes artifacts, logs, checkpoints, archives, model binaries

## Versioning System
- **Current Version:** 1.0.0 (2025-10-03)
- **Versioning Scheme:**
  - MAJOR: Breaking changes, new agent architectures, major refactors
  - MINOR: New features, model integrations, benchmark extensions
  - PATCH: Bugfixes, performance tweaks, documentation
- **Release Tags:**
  - `v1.0.0`: Initial public release, HiveRAG + SingleAgent, robust reranking, full benchmark suite

## Roadmap
### v1.1.x
- Add support for distributed multi-node RAG (HiveRAG)
- Integrate more rerankers (cross-encoder, OpenAI, custom)
- Expand test coverage (edge cases, long-context, multi-modal)
- Add CI/CD pipeline and automated regression benchmarks

### v1.2.x
- Add UI dashboard for benchmark visualization
- Enable dynamic agent orchestration (task assignment, agent spawning)
- Support for external document sources (web, DB, API)

### v2.0.x
- Major refactor: plug-and-play agent framework
- Support for streaming retrieval and generation
- Advanced ensemble voting and agent collaboration
- Model registry and auto-discovery

## How to Run
1. Install Python 3.11+, Ollama, Chroma
2. Clone repo and install dependencies (`pip install -r requirements.txt`)
3. Start Ollama server
4. Run benchmarks: `python reranker_bench_v2.py --models bge-large:latest --runs 1`
5. Explore HiveRAG and SingleAgent pipelines

## Contributing
- Fork, branch, and submit PRs
- See roadmap for priority features
- All contributions must pass tests and benchmarks

## License
MIT

---
*Last updated: 2025-10-03*