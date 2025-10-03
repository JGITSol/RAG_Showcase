#!/usr/bin/env python3
"""
Test Suite for SingleAgent RAG System with Available Models
Tests the single-agent RAG system using different models from the available list
"""

import asyncio
import json
import time
from typing import Dict
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_agent import OllamaRAGAgent

class SingleAgentModelTestSuite:
    """Test suite for evaluating SingleAgent RAG with different models"""

    def __init__(self):
        self.available_models = [
            "phi4-mini:latest",
            "snowflake-arctic-embed2:latest",
            "bge-large:latest",
            "xitao/bge-reranker-v2-m3:latest",
            "gemma3:1b",
            "gemma3:latest",
            "cogito:3b",
            "tinydolphin:latest"
        ]

        self.test_queries = [
            {
                "query": "What are the key innovations in RAG systems?",
                "category": "technical",
                "complexity": "high"
            },
            {
                "query": "How does CRAG improve upon traditional RAG methods?",
                "category": "method_comparison",
                "complexity": "medium"
            },
            {
                "query": "Explain the role of retrieval in RAG",
                "category": "conceptual",
                "complexity": "medium"
            },
            {
                "query": "What is the difference between Self-RAG and Deep RAG?",
                "category": "method_comparison",
                "complexity": "low"
            }
        ]

        self.sample_documents = [
            """
            Retrieval-Augmented Generation (RAG) enhances large language models by incorporating
            external knowledge retrieval. This approach addresses limitations like outdated information
            and hallucinations by grounding responses in retrieved documents.
            """,
            """
            CRAG (Corrective RAG) introduces self-correction mechanisms that evaluate retrieval quality
            and apply corrections when necessary. This method achieves 51% accuracy improvement over
            traditional RAG by implementing quality thresholds and fallback strategies.
            """,
            """
            Self-RAG uses reflection tokens to make adaptive retrieval decisions, while Deep RAG
            implements end-to-end reasoning with strategic information gathering. Both methods
            represent significant advancements in RAG technology.
            """,
            """
            The retrieval component in RAG systems is crucial for providing relevant context to the
            generation process. Effective retrieval ensures that generated responses are grounded in
            accurate, up-to-date information from knowledge bases.
            """
        ]

        self.results = {}

    async def test_model_configuration(self, primary_model: str, embedding_model: str, reranker_model: str = None) -> Dict:
        """Test SingleAgent RAG with specific model configuration"""
        print(f"\n🔬 Testing configuration:")
        print(f"   Primary: {primary_model}")
        print(f"   Embedding: {embedding_model}")
        print(f"   Reranker: {reranker_model or 'None'}")

        # Create custom config
        config = {
            "ollama": {
                "base_url": "http://localhost:11434",
                "primary_model": primary_model,
                "embedding_model": embedding_model,
                "reranker_model": reranker_model or "xitao/bge-reranker-v2-m3:latest",
                "temperature": 0.1,
                "num_ctx": 4096
            }
        }

        try:
            # Initialize agent with custom config
            agent = OllamaRAGAgent()
            agent.config.update(config)

            # Reinitialize components with new models
            agent.ollama_client = agent.ollama_client  # Keep same client
            agent.embeddings = agent.embeddings  # Would need to reinitialize in real implementation
            agent.vector_store = agent._init_vector_store()

            # Add test documents
            agent.add_documents(self.sample_documents)

            # Test all methods for each query
            method_results = {}
            for method in ["crag", "self-rag", "deep-rag"]:
                query_results = []
                for test_query in self.test_queries:
                    start_time = time.time()
                    result = await agent.query(test_query["query"], method)
                    end_time = time.time()

                    query_result = {
                        "query": test_query["query"],
                        "category": test_query["category"],
                        "complexity": test_query["complexity"],
                        "result": {
                            "answer": result.answer,
                            "confidence": result.confidence,
                            "processing_time": result.total_time,
                            "retrieval_time": result.retrieval_time,
                            "generation_time": result.generation_time,
                            "sources": len(result.sources),
                            "method": result.method
                        },
                        "total_time": end_time - start_time
                    }
                    query_results.append(query_result)

                # Calculate method metrics
                avg_confidence = sum(qr["result"]["confidence"] for qr in query_results) / len(query_results)
                avg_time = sum(qr["total_time"] for qr in query_results) / len(query_results)
                total_sources = sum(qr["result"]["sources"] for qr in query_results)

                method_results[method] = {
                    "metrics": {
                        "average_confidence": avg_confidence,
                        "average_time": avg_time,
                        "total_sources": total_sources,
                        "queries_tested": len(query_results)
                    },
                    "query_results": query_results
                }

            return {
                "configuration": {
                    "primary_model": primary_model,
                    "embedding_model": embedding_model,
                    "reranker_model": reranker_model
                },
                "method_results": method_results,
                "status": "success"
            }

        except Exception as e:
            print(f"❌ Test failed: {e}")
            return {
                "configuration": {
                    "primary_model": primary_model,
                    "embedding_model": embedding_model,
                    "reranker_model": reranker_model
                },
                "error": str(e),
                "status": "failed"
            }

    async def run_comprehensive_tests(self):
        """Run comprehensive tests with different model combinations"""
        print("🚀 SingleAgent RAG Model Test Suite")
        print("=" * 60)

        # Test different primary models with best embedding/reranker
        primary_models = ["gemma3:latest", "phi4-mini:latest", "cogito:3b", "tinydolphin:latest"]
        embedding_model = "snowflake-arctic-embed2:latest"
        reranker_model = "xitao/bge-reranker-v2-m3:latest"

        print(f"📊 Testing {len(primary_models)} primary models with:")
        print(f"   Embedding: {embedding_model}")
        print(f"   Reranker: {reranker_model}")

        for primary_model in primary_models:
            result = await self.test_model_configuration(primary_model, embedding_model, reranker_model)
            self.results[primary_model] = result

            if result["status"] == "success":
                crag_conf = result["method_results"]["crag"]["metrics"]["average_confidence"]
                self_rag_conf = result["method_results"]["self-rag"]["metrics"]["average_confidence"]
                deep_rag_conf = result["method_results"]["deep-rag"]["metrics"]["average_confidence"]
                avg_time = (result["method_results"]["crag"]["metrics"]["average_time"] +
                           result["method_results"]["self-rag"]["metrics"]["average_time"] +
                           result["method_results"]["deep-rag"]["metrics"]["average_time"]) / 3
                print(f"✅ {primary_model}: CRAG={crag_conf:.2f}, Self-RAG={self_rag_conf:.2f}, Deep-RAG={deep_rag_conf:.2f}, {avg_time:.2f}s avg time")
            else:
                print(f"❌ {primary_model}: Failed - {result.get('error', 'Unknown error')}")

        # Test different embedding models
        print(f"\n🔍 Testing embedding model variations with Gemma3 primary...")
        embedding_models = ["snowflake-arctic-embed2:latest", "bge-large:latest"]

        for emb_model in embedding_models:
            if emb_model != embedding_model:  # Skip if already tested
                result = await self.test_model_configuration("gemma3:latest", emb_model, reranker_model)
                self.results[f"gemma3_{emb_model}"] = result

                if result["status"] == "success":
                    crag_conf = result["method_results"]["crag"]["metrics"]["average_confidence"]
                    print(f"✅ {emb_model}: CRAG={crag_conf:.2f} avg confidence")
                else:
                    print(f"❌ {emb_model}: Failed")

    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("📊 SINGLEAGENT RAG MODEL TEST SUITE REPORT")
        print("="*80)

        successful_tests = [r for r in self.results.values() if r["status"] == "success"]
        failed_tests = [r for r in self.results.values() if r["status"] == "failed"]

        print(f"\n📈 Summary:")
        print(f"   Total configurations tested: {len(self.results)}")
        print(f"   Successful: {len(successful_tests)}")
        print(f"   Failed: {len(failed_tests)}")

        if successful_tests:
            print(f"\n🏆 Best Performing Configurations (by CRAG confidence):")

            # Sort by CRAG confidence
            sorted_results = sorted(successful_tests,
                                  key=lambda x: x["method_results"]["crag"]["metrics"]["average_confidence"],
                                  reverse=True)

            for i, result in enumerate(sorted_results[:3], 1):
                config = result["configuration"]
                crag_metrics = result["method_results"]["crag"]["metrics"]
                print(f"   {i}. {config['primary_model']} + {config['embedding_model']}")
                print(f"      CRAG Confidence: {crag_metrics['average_confidence']:.3f}")
                print(f"      CRAG Avg Time: {crag_metrics['average_time']:.2f}s")

            print(f"\n🔬 Method Performance Comparison:")
            for result in sorted_results[:1]:  # Show best config methods
                print(f"   Configuration: {result['configuration']['primary_model']}")
                for method in ["crag", "self-rag", "deep-rag"]:
                    metrics = result["method_results"][method]["metrics"]
                    print(f"      {method.upper()}: {metrics['average_confidence']:.3f} confidence, {metrics['average_time']:.2f}s")

        if failed_tests:
            print(f"\n❌ Failed Configurations:")
            for result in failed_tests:
                config = result["configuration"]
                print(f"   - {config['primary_model']}: {result.get('error', 'Unknown error')}")

        # Save detailed results
        with open("singleagent_model_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: singleagent_model_test_results.json")

        return self.results

async def main():
    """Main test execution"""
    suite = SingleAgentModelTestSuite()
    await suite.run_comprehensive_tests()
    suite.generate_report()

if __name__ == "__main__":
    asyncio.run(main())
