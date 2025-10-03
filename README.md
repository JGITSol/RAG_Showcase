---
title: RAG_Showcase
subtitle: Multi-Agent RAG Benchmark & Demo Suite
version: 1.0.0
release_date: 2025-10-03
authors:
  - JGITSol
  - AI Research Assistant
tags:
  - RAG
  - Retrieval-Augmented Generation
  - Multi-Agent
  - Benchmark
  - Python
  - Ollama
  - Chroma
  - LangChain
  - Ensemble
  - SOTA
  - Open Source
  - Research
  - Demo
  - Modular
  - Pipeline
  - Reranker
  - bge-large
  - CI/CD
  - Roadmap
---

# RAG_Showcase

> Multi-agent, modular RAG benchmark and demo suite for modern AI retrieval pipelines.

## 🚀 Key Features
- **HiveRAG**: Multi-agent, ensemble-based RAG pipeline with robust reranking and long-context support.
- **SingleAgent**: Minimalist RAG agent for baseline and ablation studies.
- **Configurable model selection** (Ollama, Chroma, LangChain)
- **Robust reranker integration** (default: `bge-large:latest`)
- **Automated benchmarking** (NDCG, MRR, MAP, Precision@K, Recall@K, latency)
- **Raw response logging** for diagnostics
- **Comprehensive test suite**
- **Modular, extensible architecture**

## 🛠️ Current Setup
- Python 3.11+
- Ollama local server
- Chroma vector DB
- LangChain (optional)
- Default reranker: `bge-large:latest` (best benchmarked)
- Benchmarks & raw responses: `bench/*.ndjson`
- .gitignore for artifacts, logs, checkpoints, archives

## 🏷️ Versioning
- **Current Version:** 1.0.0 (2025-10-03)
- **Scheme:** MAJOR.MINOR.PATCH
- **Release Tag:** `v1.0.0` (HiveRAG + SingleAgent, robust reranking, full benchmark suite)

## 🗺️ Roadmap
### v1.1.x
- Distributed multi-node RAG (HiveRAG)
- More rerankers (cross-encoder, OpenAI, custom)
- Expanded test coverage (edge cases, long-context, multi-modal)
- CI/CD pipeline, automated regression benchmarks

### v1.2.x
- UI dashboard for benchmark visualization
- Dynamic agent orchestration
- External document sources (web, DB, API)

### v2.0.x
- Plug-and-play agent framework
- Streaming retrieval/generation
- Advanced ensemble voting/collaboration
- Model registry & auto-discovery

## 📦 How to Run
1. Install Python 3.11+, Ollama, Chroma
2. Clone repo & install dependencies (`pip install -r requirements.txt`)
3. Start Ollama server
4. Run benchmarks: `python reranker_bench_v2.py --models bge-large:latest --runs 1`
5. Explore HiveRAG and SingleAgent pipelines

## 🤝 Contributing
- Fork, branch, submit PRs
- See roadmap for priority features
- All contributions must pass tests & benchmarks

## 📄 License
MIT

---
*Last updated: 2025-10-03*