#!/usr/bin/env python3
"""
Test Suite for HiveRAG System with Available Models
Tests the multi-agent RAG system using different models from the available list
"""

import asyncio
import json
import time
from typing import Dict, List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hiverag_system import HiveRAGSystem

class ModelTestSuite:
    """Test suite for evaluating HiveRAG with different models"""

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
                "query": "What are the key innovations in multi-agent RAG systems?",
                "category": "technical",
                "complexity": "high"
            },
            {
                "query": "How does CRAG improve upon traditional RAG methods?",
                "category": "method_comparison",
                "complexity": "medium"
            },
            {
                "query": "Explain the role of ensemble methods in RAG",
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
            Multi-agent RAG systems represent a paradigm shift in knowledge retrieval and generation.
            By distributing specialized tasks across multiple AI agents, these systems achieve superior
            performance through coordinated intelligence. Key innovations include hierarchical agent
            architectures, dynamic task allocation, and ensemble decision-making processes.
            """,
            """
            CRAG (Corrective RAG) introduces self-correction mechanisms that evaluate retrieval quality
            and apply corrections when necessary. This method achieves 51% accuracy improvement over
            traditional RAG by implementing quality thresholds and fallback strategies.
            """,
            """
            Ensemble methods in RAG combine multiple retrieval and generation strategies to produce
            more robust and accurate results. By leveraging diverse approaches and consensus mechanisms,
            ensemble RAG systems can mitigate individual method weaknesses and capitalize on their strengths.
            """,
            """
            Self-RAG uses reflection tokens to make adaptive retrieval decisions, while Deep RAG
            implements end-to-end reasoning with strategic information gathering. Both methods
            represent significant advancements in RAG technology, with Self-RAG focusing on efficiency
            and Deep RAG emphasizing reasoning depth.
            """
        ]

        self.results = {}

    async def test_model_configuration(self, primary_model: str, embedding_model: str, reranker_model: str = None) -> Dict:
        """Test HiveRAG with specific model configuration"""
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
                "reranker_model": reranker_model or "xitao/bge-reranker-v2-m3:latest"
            }
        }

        try:
            # Initialize system with custom config
            system = HiveRAGSystem()
            system.config.update(config)

            # Reinitialize components with new models
            system.ollama_client = system.ollama_client  # Keep same client
            system.vector_store = system._init_vector_store()

            # Add test documents
            system.add_documents(self.sample_documents)

            # Test queries
            query_results = []
            for test_query in self.test_queries:
                start_time = time.time()
                result = await system.query(test_query["query"])
                end_time = time.time()

                query_result = {
                    "query": test_query["query"],
                    "category": test_query["category"],
                    "complexity": test_query["complexity"],
                    "result": {
                        "answer": result.answer,
                        "confidence": result.confidence,
                        "processing_time": result.processing_time,
                        "agent_contributions": len(result.agent_contributions),
                        "sources": len(result.sources),
                        "method": result.method
                    },
                    "total_time": end_time - start_time
                }
                query_results.append(query_result)

            # Calculate aggregate metrics
            avg_confidence = sum(qr["result"]["confidence"] for qr in query_results) / len(query_results)
            avg_time = sum(qr["total_time"] for qr in query_results) / len(query_results)
            total_sources = sum(qr["result"]["sources"] for qr in query_results)

            return {
                "configuration": {
                    "primary_model": primary_model,
                    "embedding_model": embedding_model,
                    "reranker_model": reranker_model
                },
                "metrics": {
                    "average_confidence": avg_confidence,
                    "average_time": avg_time,
                    "total_sources": total_sources,
                    "queries_tested": len(query_results)
                },
                "query_results": query_results,
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
        print("🚀 HiveRAG Model Test Suite")
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
                print(f"✅ {primary_model}: {result['metrics']['average_confidence']:.2f} avg confidence, {result['metrics']['average_time']:.2f}s avg time")
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
                    print(f"✅ {emb_model}: {result['metrics']['average_confidence']:.2f} avg confidence")
                else:
                    print(f"❌ {emb_model}: Failed")

    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("📊 HIVERAG MODEL TEST SUITE REPORT")
        print("="*80)

        successful_tests = [r for r in self.results.values() if r["status"] == "success"]
        failed_tests = [r for r in self.results.values() if r["status"] == "failed"]

        print(f"\n📈 Summary:")
        print(f"   Total configurations tested: {len(self.results)}")
        print(f"   Successful: {len(successful_tests)}")
        print(f"   Failed: {len(failed_tests)}")

        if successful_tests:
            print(f"\n🏆 Best Performing Configurations:")

            # Sort by average confidence
            sorted_results = sorted(successful_tests,
                                  key=lambda x: x["metrics"]["average_confidence"],
                                  reverse=True)

            for i, result in enumerate(sorted_results[:3], 1):
                config = result["configuration"]
                metrics = result["metrics"]
                print(f"   {i}. {config['primary_model']} + {config['embedding_model']}")
                print(f"      Confidence: {metrics['average_confidence']:.3f}")
                print(f"      Avg Time: {metrics['average_time']:.2f}s")
                print(f"      Total Sources: {metrics['total_sources']}")

        if failed_tests:
            print(f"\n❌ Failed Configurations:")
            for result in failed_tests:
                config = result["configuration"]
                print(f"   - {config['primary_model']}: {result.get('error', 'Unknown error')}")

        # Save detailed results
        with open("hiverag_model_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: hiverag_model_test_results.json")

        return self.results

async def main():
    """Main test execution"""
    suite = ModelTestSuite()
    await suite.run_comprehensive_tests()
    suite.generate_report()

if __name__ == "__main__":
    asyncio.run(main())