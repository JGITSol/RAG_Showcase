"""
Robust Reranker Benchmark
Supports both generative LLMs and specialized reranker models
Uses industry-standard metrics: NDCG, MRR, MAP, Precision@K
"""

import requests
import time
import re
import argparse
import json
import os
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
from dataclasses import dataclass, asdict

_SANITIZE = re.compile(r'[^a-zA-Z0-9_\-]')


@dataclass
class QueryDocPair:
    """Query with document and relevance label"""
    query: str
    document: str
    relevance: int  # 0-3: 0=not relevant, 1=marginally, 2=relevant, 3=highly relevant


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a model"""
    model: str
    model_type: str  # 'generative' or 'reranker'

    # Performance metrics
    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float
    map_score: float
    precision_at_5: float
    recall_at_5: float

    # Operational metrics
    parse_rate: float
    avg_latency: float
    total_queries: int
    failed_queries: int

    # Diagnostic info
    error_types: Dict[str, int]


class RerankBenchmark:
    def print_results(self, results: List[BenchmarkResult]):
        """Print formatted benchmark results"""
        print("\n" + "="*120)
        print("RERANKER BENCHMARK RESULTS")
        print("="*120)
        results.sort(key=lambda x: x.ndcg_at_10, reverse=True)
        print(f"{'Model':<35} {'Type':<12} {'NDCG@5':<8} {'NDCG@10':<8} {'MRR':<8} {'MAP':<8} {'P@5':<8} {'R@5':<8} {'Parse%':<8} {'Latency':<8}")
        print("-"*120)
        for result in results:
            print(f"{result.model:<35} {result.model_type:<12} "
                  f"{result.ndcg_at_5:<8.3f} {result.ndcg_at_10:<8.3f} "
                  f"{result.mrr:<8.3f} {result.map_score:<8.3f} "
                  f"{result.precision_at_5:<8.3f} {result.recall_at_5:<8.3f} "
                  f"{result.parse_rate:<8.1%} {result.avg_latency:<8.2f}s")
        print("="*120)
        print("\nMetrics explanation:")
        print("  NDCG@K: Normalized Discounted Cumulative Gain - quality of ranking (0-1, higher is better)")
        print("  MRR: Mean Reciprocal Rank - position of first relevant result (0-1, higher is better)")
        print("  MAP: Mean Average Precision - overall precision (0-1, higher is better)")
        print("  P@5: Precision at 5 - fraction of relevant docs in top 5 (0-1, higher is better)")
        print("  R@5: Recall at 5 - fraction of all relevant docs found in top 5 (0-1, higher is better)")
        print("  Parse%: Percentage of successful API calls")
        print("  Latency: Average time per query-document pair")
    """Comprehensive reranker evaluation framework"""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url

        # Create diverse test dataset with clear relevance signals
        self.test_data = self._create_test_dataset()

    def _create_test_dataset(self) -> List[Tuple[str, List[QueryDocPair]]]:
        """
        Generate diverse queries with documents of varying relevance
        Each query has 10 documents with known relevance scores
        """
        datasets = []

        # Technology queries
        datasets.append((
            "What is machine learning?",
            [
                QueryDocPair("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.", 3),
                QueryDocPair("What is machine learning?", "Deep learning uses neural networks with multiple layers to process complex patterns in data.", 2),
                QueryDocPair("What is machine learning?", "Supervised learning involves training models on labeled data to make predictions.", 2),
                QueryDocPair("What is machine learning?", "Python is a popular programming language used in data science and web development.", 1),
                QueryDocPair("What is machine learning?", "Cloud computing provides on-demand access to computing resources over the internet.", 0),
                QueryDocPair("What is machine learning?", "The Renaissance was a period of cultural rebirth in Europe during the 14th-17th centuries.", 0),
                QueryDocPair("What is machine learning?", "Basketball is played with two teams of five players each on a rectangular court.", 0),
                QueryDocPair("What is machine learning?", "Quantum computing leverages quantum mechanics to solve complex computational problems.", 1),
                QueryDocPair("What is machine learning?", "Artificial intelligence encompasses systems that can perform tasks requiring human intelligence.", 2),
                QueryDocPair("What is machine learning?", "The recipe for chocolate chip cookies includes flour, butter, sugar, eggs, and chocolate chips.", 0),
            ]
        ))

        # Medical queries
        datasets.append((
            "What causes diabetes?",
            [
                QueryDocPair("What causes diabetes?", "Type 2 diabetes is primarily caused by insulin resistance and insufficient insulin production by the pancreas.", 3),
                QueryDocPair("What causes diabetes?", "Risk factors for diabetes include obesity, sedentary lifestyle, family history, and age.", 3),
                QueryDocPair("What causes diabetes?", "Type 1 diabetes is an autoimmune condition where the body attacks insulin-producing cells.", 2),
                QueryDocPair("What causes diabetes?", "Blood sugar levels are regulated by insulin, a hormone produced by the pancreas.", 2),
                QueryDocPair("What causes diabetes?", "High blood pressure can damage blood vessels and increase cardiovascular disease risk.", 1),
                QueryDocPair("What causes diabetes?", "The mitochondria is the powerhouse of the cell, producing energy through cellular respiration.", 0),
                QueryDocPair("What causes diabetes?", "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.", 0),
                QueryDocPair("What causes diabetes?", "Photosynthesis is the process by which plants convert light energy into chemical energy.", 0),
                QueryDocPair("What causes diabetes?", "Gestational diabetes occurs during pregnancy due to hormonal changes affecting insulin sensitivity.", 2),
                QueryDocPair("What causes diabetes?", "The Great Wall of China was built over centuries to protect against invasions.", 0),
            ]
        ))

        # Historical queries
        datasets.append((
            "When did World War 2 end?",
            [
                QueryDocPair("When did World War 2 end?", "World War 2 ended in 1945 with Germany surrendering in May and Japan in September.", 3),
                QueryDocPair("When did World War 2 end?", "The atomic bombs dropped on Hiroshima and Nagasaki in August 1945 led to Japan's surrender.", 3),
                QueryDocPair("When did World War 2 end?", "VE Day on May 8, 1945 marked the end of war in Europe.", 3),
                QueryDocPair("When did World War 2 end?", "The Battle of Stalingrad was a major turning point in World War 2 during 1942-1943.", 2),
                QueryDocPair("When did World War 2 end?", "D-Day invasion of Normandy occurred on June 6, 1944, leading to the liberation of Western Europe.", 2),
                QueryDocPair("When did World War 2 end?", "World War 1 ended in 1918 with the signing of the Treaty of Versailles.", 1),
                QueryDocPair("When did World War 2 end?", "The Cold War was a period of geopolitical tension between the Soviet Union and United States.", 1),
                QueryDocPair("When did World War 2 end?", "Pizza Margherita was invented in Naples, Italy in 1889.", 0),
                QueryDocPair("When did World War 2 end?", "Electric cars use batteries instead of internal combustion engines.", 0),
                QueryDocPair("When did World War 2 end?", "Mozart composed over 600 works including symphonies, operas, and chamber music.", 0),
            ]
        ))

        # Scientific queries  
        datasets.append((
            "How does photosynthesis work?",
            [
                QueryDocPair("How does photosynthesis work?", "Photosynthesis converts light energy into chemical energy, producing glucose and oxygen from carbon dioxide and water.", 3),
                QueryDocPair("How does photosynthesis work?", "Chlorophyll in plant cells absorbs light energy, primarily in the blue and red wavelengths.", 3),
                QueryDocPair("How does photosynthesis work?", "The light-dependent reactions occur in the thylakoid membranes, while the Calvin cycle happens in the stroma.", 2),
                QueryDocPair("How does photosynthesis work?", "Plants are autotrophs that produce their own food through photosynthesis.", 2),
                QueryDocPair("How does photosynthesis work?", "Cellular respiration is the process that breaks down glucose to release energy in cells.", 1),
                QueryDocPair("How does photosynthesis work?", "Carbon dioxide is a greenhouse gas that contributes to climate change.", 1),
                QueryDocPair("How does photosynthesis work?", "The Pythagorean theorem states that in a right triangle, a² + b² = c².", 0),
                QueryDocPair("How does photosynthesis work?", "Jazz music originated in African American communities in New Orleans.", 0),
                QueryDocPair("How does photosynthesis work?", "C4 and CAM photosynthesis are adaptations to hot, dry environments.", 2),
                QueryDocPair("How does photosynthesis work?", "The Roman Empire fell in 476 CE when the last emperor was deposed.", 0),
            ]
        ))

        # Programming queries
        datasets.append((
            "What is recursion in programming?",
            [
                QueryDocPair("What is recursion in programming?", "Recursion is a programming technique where a function calls itself to solve a problem.", 3),
                QueryDocPair("What is recursion in programming?", "Recursive functions must have a base case to prevent infinite loops.", 3),
                QueryDocPair("What is recursion in programming?", "The factorial function is a classic example of recursion: n! = n * (n-1)!.", 3),
                QueryDocPair("What is recursion in programming?", "Recursive algorithms can be converted to iterative versions using stacks.", 2),
                QueryDocPair("What is recursion in programming?", "Tree traversal algorithms often use recursion to visit all nodes.", 2),
                QueryDocPair("What is recursion in programming?", "Object-oriented programming uses classes and objects to structure code.", 1),
                QueryDocPair("What is recursion in programming?", "SQL is used to query and manipulate relational databases.", 0),
                QueryDocPair("What is recursion in programming?", "The Eiffel Tower is 324 meters tall and located in Paris.", 0),
                QueryDocPair("What is recursion in programming?", "Variables store data values that can be referenced and manipulated in programs.", 1),
                QueryDocPair("What is recursion in programming?", "Bananas are rich in potassium and provide quick energy.", 0),
            ]
        ))

        return datasets

    def _call_generative_model(self, model: str, query: str, document: str) -> Tuple[Optional[float], str]:
        """Call generative LLM with reranking prompt"""
    # Now uses strict JSON prompt only
        prompt = (
            f"Answer in English. Return ONLY a JSON object with a single key 'score' whose value is a number between 0 and 1.\n"
            f"Example: {{'score': 0.9}}\n\n"
            f"Query: {query}\n\nDocument: {document}\n\nJSON:"
        )

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 10
                    }
                },
                timeout=30
            )

            if response.status_code != 200:
                return (None, '')

            try:
                result = response.json()
            except Exception:
                return (None, '')

            # Ollama usually returns the generated text under 'response'
            text = result.get("response") or result.get("text") or ''
            text = (text or '').strip()

            # Try to extract a floating number (decimals or integers)
            match = re.search(r'([-+]?\d*\.\d+|\d+)', text)
            if match:
                try:
                    val = float(match.group(1))
                except Exception:
                    return (None, text)

                # Normalize heuristics:
                # - If value is in 0..10 assume 0-10 scale, divide by 10
                # - If in 0..1 assume already normalized
                # - Otherwise scale down conservatively
                if 0 <= val <= 10:
                    return (max(0.0, min(1.0, val / 10.0)), text)
                if 0 <= val <= 1:
                    return (val, text)
                # fallback scaling
                return (max(0.0, min(1.0, val / 100.0)), text)

            # If no parseable number, attempt a strict JSON-retry
            try:
                json_prompt = (
                    f"Return ONLY a JSON object with a single key \"score\" whose value is a number between 0 and 1.\n"
                    f"Example: {json.dumps({'score': 0.73})}\n\n"
                    f"Query: {query}\n\nDocument: {document}\n\nJSON:")

                response2 = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": json_prompt,
                        "stream": False,
                        "options": {"temperature": 0.0, "max_tokens": 32}
                    },
                    timeout=30
                )

                result2 = response2.json()
                text2 = result2.get("response", "")
                if not text2:
                    return (None, text)

                # Try to parse JSON from text2
                try:
                    parsed = json.loads(text2)
                    if isinstance(parsed, dict) and 'score' in parsed:
                        s = float(parsed['score'])
                        return (max(0.0, min(1.0, s)), text2)
                except Exception:
                    # fallthrough
                    pass

            except Exception:
                pass

            return (None, text)

        except Exception:
            return (None, '')

    def _call_reranker_model(self, model: str, query: str, document: str) -> Tuple[Optional[float], str]:
        """Call specialized reranker model (BGE-style)"""
        # Try embedding endpoint with concatenated text - many rerankers expose a scalar or short embedding
        try:
            text = f"query: {query} passage: {document}"

            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": text
                },
                timeout=30
            )

            if response.status_code == 200:
                try:
                    result = response.json()
                except Exception:
                    result = {}

                # Look for common keys
                embedding = result.get('embedding') or result.get('embeddings') or result.get('data')
                if isinstance(embedding, list) and embedding:
                    # If it's a nested structure (data -> [ {embedding: [...] } ]) try to extract
                    if isinstance(embedding[0], dict) and 'embedding' in embedding[0]:
                        vec = embedding[0].get('embedding', [])
                    else:
                        # embedding is a vector - compute a simple aggregate as proxy score
                        vec = embedding

                    if isinstance(vec, list) and vec:
                        # Use first coordinate as proxy and squash with sigmoid
                        coord = float(vec[0])
                        score = 1 / (1 + math.exp(-coord))
                        # return score and a textual representation of the embedding
                        return (score, json.dumps({'embedding_first': coord}))

        except Exception:
            pass

        # If embeddings didn't work, try a strict JSON generative prompt to obtain a score
        try:
            json_prompt = (
                f"Return ONLY a JSON object with a single key \"score\" whose value is a number between 0 and 1.\n"
                f"Example: {json.dumps({'score': 0.73})}\n\n"
                f"query: {query}\n\npassage: {document}\n\nJSON:")

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": json_prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "max_tokens": 32}
                },
                timeout=30
            )

            if response.status_code == 200:
                try:
                    result = response.json()
                except Exception:
                    result = {}
                text = result.get('response') or ''
                if text:
                    # Try JSON parse
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict) and 'score' in parsed:
                            s = float(parsed['score'])
                            return (max(0.0, min(1.0, s)), text)
                    except Exception:
                        # Fallback to regex
                        match = re.search(r'([-+]?\d*\.\d+|\d+)', text)
                        if match:
                            v = float(match.group(1))
                            if 0 <= v <= 1:
                                return (v, text)
                            if 0 <= v <= 10:
                                return (v / 10.0, text)
        except Exception:
            pass

        return (None, '')

    def _detect_model_type(self, model: str) -> str:
        """Detect if model is a specialized reranker or generative LLM"""
        model_lower = model.lower()

        reranker_keywords = ['rerank', 'bge-reranker', 'cross-encoder', 'bge-large']

        for keyword in reranker_keywords:
            if keyword in model_lower:
                return 'reranker'

        return 'generative'

    def _calculate_dcg(self, relevances: List[int], k: int) -> float:
        """Calculate Discounted Cumulative Gain"""
        dcg = 0.0
        for i, rel in enumerate(relevances[:k]):
            dcg += (2**rel - 1) / math.log2(i + 2)
        return dcg

    def _calculate_ndcg(self, predicted_relevances: List[int], ideal_relevances: List[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K"""
        dcg = self._calculate_dcg(predicted_relevances, k)
        idcg = self._calculate_dcg(ideal_relevances, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _calculate_mrr(self, relevances: List[int]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, rel in enumerate(relevances):
            if rel >= 2:  # Consider docs with relevance ≥ 2 as relevant
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_map(self, relevances: List[int]) -> float:
        """Calculate Mean Average Precision"""
        relevant_count = 0
        precision_sum = 0.0

        for i, rel in enumerate(relevances):
            if rel >= 2:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precision_sum += precision

        if relevant_count == 0:
            return 0.0

        return precision_sum / relevant_count

    def _calculate_precision_at_k(self, relevances: List[int], k: int) -> float:
        """Calculate Precision@K"""
        relevant = sum(1 for rel in relevances[:k] if rel >= 2)
        return relevant / k if k > 0 else 0.0

    def _calculate_recall_at_k(self, relevances: List[int], k: int) -> float:
        """Calculate Recall@K"""
        total_relevant = sum(1 for rel in relevances if rel >= 2)
        if total_relevant == 0:
            return 0.0

        relevant_in_k = sum(1 for rel in relevances[:k] if rel >= 2)
        return relevant_in_k / total_relevant

    def benchmark_model(self, model: str, runs: int = 1) -> BenchmarkResult:
        """Run complete benchmark for a model"""

        model_type = self._detect_model_type(model)
        print(f"\nBenchmarking {model} (type: {model_type})")

        all_ndcg_5 = []
        all_ndcg_10 = []
        all_mrr = []
        all_map = []
        all_precision_5 = []
        all_recall_5 = []

        latencies = []
        # prepare raw responses log (ndjson)
        safe_name = _SANITIZE.sub('_', model)
        raw_out_dir = os.path.join("bench")
        os.makedirs(raw_out_dir, exist_ok=True)
        raw_path = os.path.join(raw_out_dir, f"raw_responses_{safe_name}.ndjson")

        all_ndcg_5 = []
        all_ndcg_10 = []
        all_mrr = []
        all_map = []
        all_precision_5 = []
        all_recall_5 = []

        latencies = []
        error_types = defaultdict(int)
        failed_queries = 0
        total_calls = 0
        successful_calls = 0

        with open(raw_path, "w", encoding="utf-8") as raw_f:
            for run in range(runs):
                for query_text, pairs in self.test_data:
                    scored_docs = []
                    for pair in pairs:
                        start = time.time()
                        if model_type == 'reranker':
                            score_tuple = self._call_reranker_model(model, pair.query, pair.document)
                        else:
                            score_tuple = self._call_generative_model(model, pair.query, pair.document)
                        latency = time.time() - start
                        latencies.append(latency)
                        total_calls += 1
                        # normalize tuple shape
                        if isinstance(score_tuple, tuple) and len(score_tuple) == 2:
                            score_val, raw_text = score_tuple
                        else:
                            score_val, raw_text = (None, "")
                        # log raw response
                        try:
                            entry = {
                                "model": model,
                                "query": pair.query,
                                "document": pair.document,
                                "raw": raw_text,
                                "score": score_val,
                                "latency": latency
                            }
                            raw_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                        if score_val is not None:
                            try:
                                num_score = float(score_val)
                                scored_docs.append((pair, num_score))
                                successful_calls += 1
                            except Exception:
                                error_types['not_numeric'] += 1
                        else:
                            error_types['parsing_failed'] += 1
                    if scored_docs:
                        scored_docs.sort(key=lambda x: x[1], reverse=True)
                        predicted_relevances = [doc[0].relevance for doc in scored_docs]
                        ideal_relevances = sorted([p.relevance for p in pairs], reverse=True)
                        all_ndcg_5.append(self._calculate_ndcg(predicted_relevances, ideal_relevances, 5))
                        all_ndcg_10.append(self._calculate_ndcg(predicted_relevances, ideal_relevances, 10))
                        all_mrr.append(self._calculate_mrr(predicted_relevances))
                        all_map.append(self._calculate_map(predicted_relevances))
                        all_precision_5.append(self._calculate_precision_at_k(predicted_relevances, 5))
                        all_recall_5.append(self._calculate_recall_at_k(predicted_relevances, 5))
                    else:
                        failed_queries += 1

        result = BenchmarkResult(
            model=model,
            model_type=model_type,
            ndcg_at_5=sum(all_ndcg_5) / len(all_ndcg_5) if all_ndcg_5 else 0.0,
            ndcg_at_10=sum(all_ndcg_10) / len(all_ndcg_10) if all_ndcg_10 else 0.0,
            mrr=sum(all_mrr) / len(all_mrr) if all_mrr else 0.0,
            map_score=sum(all_map) / len(all_map) if all_map else 0.0,
            precision_at_5=sum(all_precision_5) / len(all_precision_5) if all_precision_5 else 0.0,
            recall_at_5=sum(all_recall_5) / len(all_recall_5) if all_recall_5 else 0.0,
            parse_rate=successful_calls / total_calls if total_calls > 0 else 0.0,
            avg_latency=sum(latencies) / len(latencies) if latencies else 0.0,
            total_queries=len(self.test_data) * runs,
            failed_queries=failed_queries,
            error_types=dict(error_types)
        )
        return result
        print("="*120)
        print("\nMetrics explanation:")
        print("  NDCG@K: Normalized Discounted Cumulative Gain - quality of ranking (0-1, higher is better)")
        print("  MRR: Mean Reciprocal Rank - position of first relevant result (0-1, higher is better)")
        print("  MAP: Mean Average Precision - overall precision (0-1, higher is better)")
        print("  P@5: Precision at 5 - fraction of relevant docs in top 5 (0-1, higher is better)")
        print("  R@5: Recall at 5 - fraction of all relevant docs found in top 5 (0-1, higher is better)")
        print("  Parse%: Percentage of successful API calls")
        print("  Latency: Average time per query-document pair")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive reranker benchmark")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of models")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per model")
    parser.add_argument("--url", type=str, default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--output", type=str, default="bench/reranker_v2_results.json", help="JSON output file path")

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    benchmark = RerankBenchmark(ollama_url=args.url)
    results = []

    for model in models:
        try:
            result = benchmark.benchmark_model(model, runs=args.runs)
            results.append(result)

            # Print individual result
            print(f"  NDCG@10: {result.ndcg_at_10:.3f} | MRR: {result.mrr:.3f} | "
                  f"Parse: {result.parse_rate:.1%} | Latency: {result.avg_latency:.2f}s")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Print summary
    benchmark.print_results(results)

    # Save to JSON if requested
    if args.output:
        outdir = os.path.dirname(args.output)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
