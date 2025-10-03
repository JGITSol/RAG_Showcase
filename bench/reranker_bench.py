"""Quick reranker benchmark.

This script queries local Ollama models with a strict numeric reranker prompt
and measures:
 - parseable_rate: fraction of responses parsed as a numeric score
 - avg_latency: average request time (s)
 - discrimination: mean(score|relevant) - mean(score|nonrelevant)

Usage:
  & .venv\Scripts\Activate.ps1
  python .\bench\reranker_bench.py --models "xitao/bge-reranker-v2-m3:latest" --runs 3

Outputs:
 - prints a summary table
 - writes JSON results to bench/reranker_results.json
 - writes any raw failed responses to reranker_bench_failures.log
"""
import time
import re
import json
import argparse
import os
from types import SimpleNamespace

try:
    import ollama
except Exception:
    ollama = None

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def make_synthetic_corpus(queries, docs_per_query=4, long_size=800):
    docs = []
    for qidx, q in enumerate(queries):
        # one relevant long doc
        text = (f"Relevant for: {q} " + (" lorem ipsum " * (long_size // 10))).strip()
        meta = {"source": f"q{qidx}_relevant", "relevant_for": [q]}
        docs.append(SimpleNamespace(page_content=text, metadata=meta))
        for j in range(docs_per_query - 1):
            text = (f"Distractor {j} for {q} " + (" lorem ipsum " * (long_size // 20))).strip()
            meta = {"source": f"q{qidx}_d{j}", "relevant_for": []}
            docs.append(SimpleNamespace(page_content=text, metadata=meta))
    return docs

def parse_score(text):
    if text is None:
        return None
    t = str(text).strip()
    # try direct float
    try:
        return float(t)
    except Exception:
        pass
    # regex
    m = re.search(r"([0-9]*\.?[0-9]+)", t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # try JSON
    try:
        j = json.loads(t)
        if isinstance(j, dict) and 'score' in j:
            return float(j['score'])
    except Exception:
        pass
    return None

def call_model(client, model, prompt, timeout=30.0):
    t0 = time.time()
    try:
        resp = client.generate(model=model, prompt=prompt, options={"temperature":0.0})
        text = resp.get('response') if isinstance(resp, dict) else getattr(resp, 'response', '')
    except Exception as e:
        text = f"<error> {e}"
    return text, time.time() - t0

def reranker_prompt(query, doc):
    return f"""Rate how relevant this document is to the query.

Query: {query}
Document: {doc[:400]}...

IMPORTANT: Respond with a single decimal number between 0.0 and 1.0 and nothing else (e.g., 0.73).
"""

def json_retry_prompt(query, doc):
    return f"""Rate relevance as JSON.

Query: {query}
Document: {doc[:400]}...

Respond only with JSON: {{"score": <decimal between 0.0 and 1.0>}}
Example: {{"score": 0.82}}
"""

def run_for_model(client, model, queries, docs, runs=1):
    results = {
        'model': model,
        'runs': runs,
        'per_query': []
    }
    safe_name = model.replace('/', '_').replace(':', '_')
    failures_log = os.path.join(repo_root, f'rereranker_bench_failures_{safe_name}.log')
    for q in queries:
        qr = {'query': q, 'pairs': []}
        for doc in docs:
            is_rel = q in doc.metadata.get('relevant_for', [])
            pair = {'source': doc.metadata.get('source'), 'relevant': is_rel, 'scores': [], 'latencies': [], 'parsed': []}
            for r in range(runs):
                p = reranker_prompt(q, doc.page_content)
                text, lat = call_model(client, model, p)
                sc = parse_score(text)
                if sc is None:
                    # retry with JSON strict
                    p2 = json_retry_prompt(q, doc.page_content)
                    text2, lat2 = call_model(client, model, p2)
                    sc = parse_score(text2)
                    lat += lat2
                    text = text2 if sc is not None else text2
                pair['scores'].append(sc)
                pair['latencies'].append(lat)
                pair['parsed'].append(sc is not None)
                if sc is None:
                    try:
                        with open(failures_log, 'a', encoding='utf-8') as f:
                            f.write(f"QUERY: {q[:200]}\nDOC_SRC: {doc.metadata.get('source')}\nRESPONSE:\n{text}\n----\n")
                    except Exception:
                        pass
            qr['pairs'].append(pair)
        results['per_query'].append(qr)
    return results

def summarize_model(res):
    # compute parseable rate, avg latency, discrimination
    parsed = 0
    total = 0
    lat_sum = 0.0
    rel_scores = []
    nonrel_scores = []
    for q in res['per_query']:
        for p in q['pairs']:
            for i, s in enumerate(p['scores']):
                total += 1
                if p['parsed'][i]:
                    parsed += 1
                lat_sum += p['latencies'][i]
                if p['relevant']:
                    if s is not None:
                        rel_scores.append(s)
                else:
                    if s is not None:
                        nonrel_scores.append(s)
    parse_rate = parsed / total if total else 0.0
    avg_lat = lat_sum / total if total else None
    discr = (sum(rel_scores)/len(rel_scores) if rel_scores else 0.0) - (sum(nonrel_scores)/len(nonrel_scores) if nonrel_scores else 0.0)
    return {'model': res['model'], 'parse_rate': parse_rate, 'avg_latency': avg_lat, 'discrimination': discr, 'num_pairs': total}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default='xitao/bge-reranker-v2-m3:latest', help='Comma-separated model ids to test')
    parser.add_argument('--runs', type=int, default=2, help='Number of runs per pair')
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    queries = [
        'Analyze the historical evolution of the concept of sovereignty across political systems',
        'Compare antiviral strategies used in pandemic response',
        'Plan migrating monolith to microservices preserving transactions'
    ]
    docs = make_synthetic_corpus(queries, docs_per_query=6, long_size=600)

    if ollama is None:
        print('Ollama client not installed or importable. Install ollama and ensure local server is running.')
        return

    client = ollama.Client(host='http://localhost:11434')

    all_results = []
    for model in models:
        print(f'Running reranker bench for model: {model}')
        res = run_for_model(client, model, queries, docs, runs=args.runs)
        summary = summarize_model(res)
        all_results.append({'summary': summary, 'detail': res})
        print('  ->', summary)

    out_file = os.path.join(repo_root, 'bench', 'reranker_results.json')
    try:
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
    except Exception:
        pass

    print('\nWritten results to', out_file)

if __name__ == '__main__':
    main()
