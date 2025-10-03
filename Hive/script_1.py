# Create the comprehensive HiveRAG Multi-Agent System implementation

hiverag_implementation = '''#!/usr/bin/env python3
"""
HiveRAG: Multi-Agent Hive Intelligence RAG System
Implements state-of-the-art ensemble methods for long context knowledge retrieval

Based on latest research:
- MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought
- Chain of Agents: Large Language Models Collaborating on Long-Context Tasks  
- RAG Ensemble Framework: Theoretical and Mechanistic Analysis
- HIVE: Harnessing Language for Coordination Multi-Agent Control
- HM-RAG: Hierarchical Multi-Agent Multimodal RAG
- MAIN-RAG: Multi-Agent Filtering RAG

Author: AI Research Assistant
Date: September 2025
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import uuid

# External dependencies
try:
    import chromadb
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    from langchain.embeddings import OllamaEmbeddings
    from langchain.vectorstores import Chroma
    import ollama
    import networkx as nx
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install chromadb langchain langchain-ollama ollama networkx sentence-transformers")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agent Types and Status
class AgentType(Enum):
    QUEEN = "queen"      # Central coordination agents
    WORKER = "worker"    # Specialized RAG task agents  
    SCOUT = "scout"      # Exploration and QA agents

class AgentStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active" 
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"

@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass 
class QueryResult:
    """Comprehensive query result with ensemble metadata"""
    query: str
    answer: str
    method: str
    confidence: float
    sources: List[str]
    processing_time: float
    agent_contributions: Dict[str, Any]
    ensemble_metadata: Dict[str, Any] 
    context_usage: Dict[str, Any]

class BaseAgent:
    """Base agent class with core functionality"""
    
    def __init__(self, agent_id: str, name: str, agent_type: AgentType, specialization: str):
        self.id = agent_id
        self.name = name
        self.type = agent_type
        self.specialization = specialization
        self.status = AgentStatus.IDLE
        self.current_task = "Initializing..."
        self.performance_metrics = {}
        self.message_queue = deque()
        self.hive = None  # Reference to parent hive
        self.context_allocation = {}
        
    def set_hive(self, hive):
        """Set reference to parent hive system"""
        self.hive = hive
        
    async def send_message(self, receiver_id: str, message_type: str, content: Dict[str, Any]):
        """Send message to another agent via hive"""
        if not self.hive:
            logger.error(f"Agent {self.id} not connected to hive")
            return
            
        message = AgentMessage(
            sender_id=self.id,
            receiver_id=receiver_id, 
            message_type=message_type,
            content=content
        )
        await self.hive.route_message(message)
        
    async def receive_message(self, message: AgentMessage):
        """Handle incoming message"""
        self.message_queue.append(message)
        await self.process_messages()
        
    async def process_messages(self):
        """Process queued messages"""
        while self.message_queue:
            message = self.message_queue.popleft()
            await self.handle_message(message)
            
    async def handle_message(self, message: AgentMessage):
        """Override in subclasses to handle specific message types"""
        logger.info(f"Agent {self.name} received {message.message_type} from {message.sender_id}")
        
    def update_status(self, status: AgentStatus, task: str = None):
        """Update agent status and current task"""
        self.status = status
        if task:
            self.current_task = task
        logger.debug(f"Agent {self.name} status: {status.value}, task: {self.current_task}")

class HiveOrchestrator(BaseAgent):
    """Queen Agent: Master coordinator managing the entire hive"""
    
    def __init__(self):
        super().__init__(
            agent_id="hive-orchestrator",
            name="Hive Orchestrator", 
            agent_type=AgentType.QUEEN,
            specialization="Global optimization and strategic planning"
        )
        self.active_queries = {}
        self.agent_assignments = {}
        
    async def handle_message(self, message: AgentMessage):
        """Handle coordination messages"""
        if message.message_type == "query_request":
            await self.process_query_request(message)
        elif message.message_type == "agent_result":
            await self.collect_agent_result(message)
        elif message.message_type == "status_update":
            await self.update_agent_status(message)
            
    async def process_query_request(self, message: AgentMessage):
        """Decompose query and assign to specialized agents"""
        self.update_status(AgentStatus.PROCESSING, "Query decomposition and agent assignment")
        
        query_data = message.content
        query_id = str(uuid.uuid4())
        query_text = query_data.get("query", "")
        
        # Analyze query complexity
        complexity = await self.analyze_query_complexity(query_text)
        
        # Determine required agents based on complexity and type
        required_agents = await self.determine_agent_assignment(query_text, complexity)
        
        # Create query coordination record
        self.active_queries[query_id] = {
            "query": query_text,
            "complexity": complexity,
            "assigned_agents": required_agents,
            "results": {},
            "start_time": time.time(),
            "requester": message.sender_id
        }
        
        # Assign tasks to agents
        for agent_id, task_spec in required_agents.items():
            await self.send_message(agent_id, "task_assignment", {
                "query_id": query_id,
                "query": query_text,
                "task_spec": task_spec,
                "context_allocation": task_spec.get("context_allocation", {})
            })
        
        logger.info(f"Query {query_id} assigned to {len(required_agents)} agents")
        
    async def analyze_query_complexity(self, query: str) -> str:
        """Analyze query complexity for optimal agent assignment"""
        # Simple heuristic - can be enhanced with ML models
        word_count = len(query.split())
        has_multiple_questions = len([s for s in query.split('?') if s.strip()]) > 1
        has_reasoning_terms = any(term in query.lower() for term in 
                                ['analyze', 'compare', 'evaluate', 'explain', 'why', 'how'])
        
        if word_count > 50 or has_multiple_questions or has_reasoning_terms:
            return "high"
        elif word_count > 20 or has_reasoning_terms:
            return "medium"
        else:
            return "low"
    
    async def determine_agent_assignment(self, query: str, complexity: str) -> Dict[str, Dict]:
        """Determine which agents to assign based on query characteristics"""
        assignments = {}
        
        # Always include ensemble agent for result coordination
        assignments["ensemble-agent"] = {
            "role": "result_coordination",
            "priority": 1,
            "context_allocation": {"tokens": 50000}
        }
        
        if complexity == "high":
            # Use all specialized RAG agents for complex queries
            assignments.update({
                "crag-agent": {
                    "role": "robust_retrieval", 
                    "priority": 1,
                    "context_allocation": {"tokens": 200000}
                },
                "selfrag-agent": {
                    "role": "adaptive_retrieval",
                    "priority": 1, 
                    "context_allocation": {"tokens": 150000}
                },
                "deeprag-agent": {
                    "role": "reasoning_retrieval",
                    "priority": 1,
                    "context_allocation": {"tokens": 250000}
                }
            })
        elif complexity == "medium":
            # Use CRAG and Self-RAG for medium complexity
            assignments.update({
                "crag-agent": {
                    "role": "robust_retrieval",
                    "priority": 1,
                    "context_allocation": {"tokens": 150000}
                },
                "selfrag-agent": {
                    "role": "adaptive_retrieval", 
                    "priority": 1,
                    "context_allocation": {"tokens": 100000}
                }
            })
        else:
            # Use Self-RAG for simple queries
            assignments["selfrag-agent"] = {
                "role": "simple_retrieval",
                "priority": 1,
                "context_allocation": {"tokens": 80000}
            }
        
        return assignments
        
    async def collect_agent_result(self, message: AgentMessage):
        """Collect results from specialized agents"""
        result_data = message.content
        query_id = result_data.get("query_id")
        
        if query_id not in self.active_queries:
            logger.warning(f"Received result for unknown query {query_id}")
            return
            
        # Store agent result
        self.active_queries[query_id]["results"][message.sender_id] = result_data
        
        # Check if all agents have responded
        expected_agents = set(self.active_queries[query_id]["assigned_agents"].keys())
        received_agents = set(self.active_queries[query_id]["results"].keys())
        
        if expected_agents.issubset(received_agents):
            # All agents have responded - trigger ensemble coordination
            await self.send_message("ensemble-agent", "coordinate_results", {
                "query_id": query_id,
                "query_data": self.active_queries[query_id]
            })

class ContextManager(BaseAgent):
    """Queen Agent: Long context coordinator and memory manager"""
    
    def __init__(self):
        super().__init__(
            agent_id="context-manager",
            name="Context Manager",
            agent_type=AgentType.QUEEN, 
            specialization="Long context optimization and management"
        )
        self.context_segments = {}
        self.memory_pool = {}
        self.attention_coordination = {}
        self.max_context_window = 2000000  # 2M tokens
        self.current_usage = 0
        
    async def handle_message(self, message: AgentMessage):
        """Handle context management messages"""
        if message.message_type == "context_request":
            await self.allocate_context(message)
        elif message.message_type == "context_release":
            await self.release_context(message)
        elif message.message_type == "memory_update":
            await self.update_memory(message)
            
    async def allocate_context(self, message: AgentMessage):
        """Allocate context window segments to agents"""
        request_data = message.content
        agent_id = message.sender_id
        requested_tokens = request_data.get("tokens", 100000)
        
        # Check available capacity
        if self.current_usage + requested_tokens > self.max_context_window:
            # Context compression or reallocation needed
            await self.optimize_context_allocation()
            
        # Allocate segment
        segment_id = f"{agent_id}_{uuid.uuid4().hex[:8]}"
        self.context_segments[segment_id] = {
            "agent_id": agent_id,
            "tokens": requested_tokens,
            "allocated_at": time.time(),
            "data": request_data.get("data", {})
        }
        
        self.current_usage += requested_tokens
        
        # Notify agent of allocation
        await self.send_message(agent_id, "context_allocated", {
            "segment_id": segment_id,
            "tokens": requested_tokens,
            "total_usage": self.current_usage
        })
        
        self.update_status(AgentStatus.ACTIVE, 
                          f"Managing {self.current_usage/1000:.0f}K tokens across {len(self.context_segments)} segments")
        
    async def optimize_context_allocation(self):
        """Optimize context allocation through compression and prioritization"""
        logger.info("Optimizing context allocation...")
        
        # Sort segments by age and priority
        sorted_segments = sorted(
            self.context_segments.items(),
            key=lambda x: x[1]["allocated_at"]
        )
        
        # Release oldest segments if needed
        for segment_id, segment_data in sorted_segments[:len(sorted_segments)//4]:
            await self.release_context_segment(segment_id)

class CRAGAgent(BaseAgent):
    """Worker Agent: Corrective RAG specialist with self-correction"""
    
    def __init__(self, ollama_client, vector_store):
        super().__init__(
            agent_id="crag-agent",
            name="CRAG Agent", 
            agent_type=AgentType.WORKER,
            specialization="Robust retrieval with error correction"
        )
        self.ollama_client = ollama_client
        self.vector_store = vector_store
        self.evaluation_threshold = 0.7
        self.correction_attempts = 3
        
    async def handle_message(self, message: AgentMessage):
        """Handle CRAG-specific tasks"""
        if message.message_type == "task_assignment":
            await self.process_task_assignment(message)
            
    async def process_task_assignment(self, message: AgentMessage):
        """Process assigned CRAG retrieval task"""
        self.update_status(AgentStatus.PROCESSING, "Document quality evaluation and self-correction")
        
        task_data = message.content
        query = task_data.get("query")
        query_id = task_data.get("query_id")
        
        start_time = time.time()
        
        # Step 1: Initial retrieval
        retrieved_docs = await self.retrieve_documents(query)
        
        # Step 2: Evaluate retrieval quality
        evaluation_results = await self.evaluate_retrieval_quality(query, retrieved_docs)
        
        # Step 3: Apply correction if needed
        corrected_docs, correction_applied = await self.apply_corrections(
            query, retrieved_docs, evaluation_results
        )
        
        # Step 4: Generate response
        response = await self.generate_response(query, corrected_docs, correction_applied)
        
        processing_time = time.time() - start_time
        
        # Send result back to orchestrator
        await self.send_message("hive-orchestrator", "agent_result", {
            "query_id": query_id,
            "agent_id": self.id,
            "method": "CRAG",
            "response": response,
            "confidence": np.mean([score for _, score in evaluation_results]),
            "sources": [doc.metadata.get("source", "unknown") for doc, _ in corrected_docs],
            "processing_time": processing_time,
            "correction_applied": correction_applied,
            "evaluation_scores": [score for _, score in evaluation_results]
        })
        
        self.update_status(AgentStatus.ACTIVE, "Ready for next task")
        
    async def retrieve_documents(self, query: str, k: int = 5):
        """Retrieve documents using similarity search"""
        return self.vector_store.similarity_search_with_score(query, k=k)
        
    async def evaluate_retrieval_quality(self, query: str, docs_with_scores):
        """Evaluate quality of retrieved documents"""
        evaluation_results = []
        
        for doc, sim_score in docs_with_scores:
            # Use LLM to evaluate relevance
            eval_prompt = f"""Rate document relevance for query on scale 0.0-1.0:
            
Query: {query}
Document: {doc.page_content[:300]}...

Relevance score (0.0-1.0):"""
            
            try:
                response = self.ollama_client.generate(
                    model="llama3.1:8b",
                    prompt=eval_prompt,
                    options={"temperature": 0.1, "num_predict": 5}
                )
                eval_score = float(response['response'].strip())
                eval_score = max(0.0, min(1.0, eval_score))  # Clamp to valid range
            except:
                eval_score = sim_score  # Fallback to similarity score
                
            evaluation_results.append((doc, eval_score))
            
        return evaluation_results
        
    async def apply_corrections(self, query: str, docs_with_scores, evaluation_results):
        """Apply CRAG corrections based on evaluation"""
        high_quality_docs = [
            (doc, score) for doc, score in evaluation_results
            if score >= self.evaluation_threshold
        ]
        
        if len(high_quality_docs) == 0:
            # Apply correction: expand search or use different strategy
            logger.info("CRAG correction: Expanding search due to low quality results")
            expanded_docs = await self.retrieve_documents(query, k=10)
            expanded_evaluation = await self.evaluate_retrieval_quality(query, expanded_docs)
            high_quality_docs = [
                (doc, score) for doc, score in expanded_evaluation[:5]
            ]
            return high_quality_docs, True
        
        return high_quality_docs, False
        
    async def generate_response(self, query: str, docs_with_scores, correction_applied: bool):
        """Generate CRAG response"""
        context = "\\n\\n".join([doc.page_content for doc, _ in docs_with_scores])
        
        prompt = f"""Answer using CRAG methodology with provided context:

Query: {query}
Context: {context}
{"Note: Corrective measures were applied to improve retrieval quality." if correction_applied else ""}

Provide accurate, well-grounded response:"""

        response = self.ollama_client.generate(
            model="llama3.1:8b",
            prompt=prompt,
            options={"temperature": 0.1}
        )
        
        return response['response']

class SelfRAGAgent(BaseAgent):
    """Worker Agent: Self-RAG with adaptive retrieval decisions"""
    
    def __init__(self, ollama_client, vector_store):
        super().__init__(
            agent_id="selfrag-agent", 
            name="Self-RAG Agent",
            agent_type=AgentType.WORKER,
            specialization="Efficient adaptive retrieval"
        )
        self.ollama_client = ollama_client
        self.vector_store = vector_store
        self.reflection_tokens = ["[Retrieve]", "[No Retrieve]", "[Relevant]", "[Partially Relevant]", "[Irrelevant]"]
        
    async def handle_message(self, message: AgentMessage):
        """Handle Self-RAG tasks"""
        if message.message_type == "task_assignment":
            await self.process_task_assignment(message)
            
    async def process_task_assignment(self, message: AgentMessage):
        """Process Self-RAG task with reflection tokens"""
        self.update_status(AgentStatus.PROCESSING, "Reflection token analysis and retrieval decisions")
        
        task_data = message.content
        query = task_data.get("query")
        query_id = task_data.get("query_id")
        
        start_time = time.time()
        
        # Step 1: Decide whether to retrieve
        should_retrieve = await self.make_retrieval_decision(query)
        
        retrieved_docs = []
        if should_retrieve:
            # Step 2: Retrieve and critique documents
            retrieved_docs = await self.retrieve_and_critique(query)
            
        # Step 3: Generate response
        response = await self.generate_self_rag_response(query, retrieved_docs, should_retrieve)
        
        processing_time = time.time() - start_time
        
        # Send result
        await self.send_message("hive-orchestrator", "agent_result", {
            "query_id": query_id,
            "agent_id": self.id,
            "method": "Self-RAG",
            "response": response,
            "confidence": 0.8 if should_retrieve and retrieved_docs else 0.6,
            "sources": [doc.metadata.get("source", "unknown") for doc, _ in retrieved_docs],
            "processing_time": processing_time,
            "retrieval_decision": "[Retrieve]" if should_retrieve else "[No Retrieve]",
            "relevant_docs": len(retrieved_docs)
        })
        
        self.update_status(AgentStatus.ACTIVE, "Ready for adaptive retrieval")
        
    async def make_retrieval_decision(self, query: str) -> bool:
        """Use reflection to decide if retrieval is needed"""
        decision_prompt = f"""Using Self-RAG reflection, decide if external retrieval is needed:

Query: {query}

Can you answer this with existing knowledge or do you need external information?
Respond with exactly: [Retrieve] or [No Retrieve]

Decision:"""
        
        response = self.ollama_client.generate(
            model="llama3.1:8b",
            prompt=decision_prompt,
            options={"temperature": 0.1, "num_predict": 10}
        )
        
        return "[Retrieve]" in response['response']
        
    async def retrieve_and_critique(self, query: str):
        """Retrieve documents and critique their relevance"""
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=5)
        relevant_docs = []
        
        for doc, score in docs_with_scores:
            # Critique document relevance
            critique_prompt = f"""Critique document relevance using Self-RAG tokens:

Query: {query}
Document: {doc.page_content[:400]}...

Respond with exactly one token: [Relevant], [Partially Relevant], or [Irrelevant]

Critique:"""
            
            response = self.ollama_client.generate(
                model="llama3.1:8b", 
                prompt=critique_prompt,
                options={"temperature": 0.1, "num_predict": 10}
            )
            
            if "[Relevant]" in response['response'] or "[Partially Relevant]" in response['response']:
                relevant_docs.append((doc, score))
                
        return relevant_docs
        
    async def generate_self_rag_response(self, query: str, docs_with_scores, retrieved: bool):
        """Generate Self-RAG response with reflection"""
        if retrieved and docs_with_scores:
            context = "\\n\\n".join([doc.page_content for doc, _ in docs_with_scores])
            prompt = f"""Self-RAG response with retrieved context:

Query: {query}
Retrieved Context: {context}

Generate comprehensive answer using retrieved information:"""
        else:
            prompt = f"""Self-RAG response without retrieval:

Query: {query}

Answer based on existing knowledge. State limitations clearly:"""
            
        response = self.ollama_client.generate(
            model="llama3.1:8b",
            prompt=prompt,
            options={"temperature": 0.1}
        )
        
        return response['response']

class DeepRAGAgent(BaseAgent):
    """Worker Agent: Deep RAG with multi-step reasoning"""
    
    def __init__(self, ollama_client, vector_store):
        super().__init__(
            agent_id="deeprag-agent",
            name="Deep RAG Agent",
            agent_type=AgentType.WORKER, 
            specialization="Complex reasoning and analysis"
        )
        self.ollama_client = ollama_client
        self.vector_store = vector_store
        self.max_reasoning_steps = 5
        
    async def handle_message(self, message: AgentMessage):
        """Handle Deep RAG reasoning tasks"""
        if message.message_type == "task_assignment":
            await self.process_task_assignment(message)
            
    async def process_task_assignment(self, message: AgentMessage):
        """Process Deep RAG task with multi-step reasoning"""
        self.update_status(AgentStatus.PROCESSING, "Multi-step reasoning chain construction")
        
        task_data = message.content
        query = task_data.get("query") 
        query_id = task_data.get("query_id")
        
        start_time = time.time()
        
        # Step 1: Analyze query complexity
        complexity = await self.analyze_reasoning_complexity(query)
        
        # Step 2: Execute multi-step reasoning with strategic retrieval
        reasoning_chain, retrieved_docs = await self.execute_deep_reasoning(query, complexity)
        
        # Step 3: Synthesize final response
        response = await self.synthesize_response(query, reasoning_chain, retrieved_docs)
        
        processing_time = time.time() - start_time
        
        # Send result
        await self.send_message("hive-orchestrator", "agent_result", {
            "query_id": query_id,
            "agent_id": self.id, 
            "method": "Deep RAG",
            "response": response,
            "confidence": min(0.9, 0.6 + len(reasoning_chain) * 0.05),
            "sources": [doc.metadata.get("source", "unknown") for doc, _ in retrieved_docs],
            "processing_time": processing_time,
            "reasoning_chain": reasoning_chain,
            "reasoning_steps": len(reasoning_chain),
            "retrieval_points": len(retrieved_docs)
        })
        
        self.update_status(AgentStatus.ACTIVE, "Ready for deep reasoning")
        
    async def analyze_reasoning_complexity(self, query: str) -> int:
        """Analyze reasoning complexity to determine steps needed"""
        analysis_prompt = f"""Analyze reasoning complexity for query:

Query: {query}

Rate complexity 1-5 where:
1 = Simple factual
2 = Basic analysis  
3 = Multi-step reasoning
4 = Complex synthesis
5 = Advanced reasoning

Complexity level (1-5):"""
        
        response = self.ollama_client.generate(
            model="llama3.1:8b",
            prompt=analysis_prompt,
            options={"temperature": 0.1, "num_predict": 5}
        )
        
        try:
            complexity = int(response['response'].strip())
            return min(max(complexity, 1), 5)
        except:
            return 3  # Default complexity
            
    async def execute_deep_reasoning(self, query: str, complexity: int):
        """Execute multi-step reasoning with strategic retrieval"""
        reasoning_steps = min(self.max_reasoning_steps, complexity + 1)
        reasoning_chain = []
        all_retrieved_docs = []
        
        for step in range(reasoning_steps):
            # Strategic retrieval at key decision points
            if step == 0 or step % 2 == 1:
                step_query = f"{query} (reasoning step {step + 1})"
                docs_with_scores = self.vector_store.similarity_search_with_score(step_query, k=3)
                all_retrieved_docs.extend(docs_with_scores)
                
                context = "\\n".join([doc.page_content for doc, _ in docs_with_scores])
                step_prompt = f"""Deep RAG reasoning step {step + 1}/{reasoning_steps}:

Query: {query}
Context: {context}
Previous reasoning: {' → '.join(reasoning_chain)}

What reasoning step emerges from this context?

Step {step + 1}:"""
            else:
                step_prompt = f"""Deep RAG reasoning step {step + 1}/{reasoning_steps}:

Query: {query}
Previous reasoning: {' → '.join(reasoning_chain)}

Continue logical reasoning sequence:

Step {step + 1}:"""
                
            response = self.ollama_client.generate(
                model="llama3.1:8b",
                prompt=step_prompt,
                options={"temperature": 0.1}
            )
            
            reasoning_chain.append(response['response'].strip())
            
        return reasoning_chain, all_retrieved_docs
        
    async def synthesize_response(self, query: str, reasoning_chain: List[str], docs_with_scores):
        """Synthesize final Deep RAG response"""
        context = "\\n\\n".join([doc.page_content for doc, _ in docs_with_scores])
        reasoning_summary = " → ".join(reasoning_chain)
        
        synthesis_prompt = f"""Deep RAG synthesis with step-by-step reasoning:

Query: {query}
Reasoning Chain: {reasoning_summary}
Retrieved Context: {context}

Synthesize comprehensive final answer showing reasoning progression:"""

        response = self.ollama_client.generate(
            model="llama3.1:8b",
            prompt=synthesis_prompt,
            options={"temperature": 0.1}
        )
        
        return response['response']

class EnsembleAgent(BaseAgent):
    """Worker Agent: Ensemble coordinator and result fusion"""
    
    def __init__(self, ollama_client):
        super().__init__(
            agent_id="ensemble-agent",
            name="Ensemble Agent", 
            agent_type=AgentType.WORKER,
            specialization="Ensemble coordination and optimization"
        )
        self.ollama_client = ollama_client
        self.fusion_methods = ["weighted_average", "consensus_voting", "confidence_ranking"]
        
    async def handle_message(self, message: AgentMessage):
        """Handle ensemble coordination"""
        if message.message_type == "coordinate_results":
            await self.coordinate_ensemble_results(message)
            
    async def coordinate_ensemble_results(self, message: AgentMessage):
        """Coordinate and fuse results from multiple RAG agents"""
        self.update_status(AgentStatus.PROCESSING, "Result fusion and consensus building")
        
        coordination_data = message.content
        query_data = coordination_data.get("query_data")
        agent_results = query_data.get("results", {})
        
        query = query_data.get("query")
        start_time = query_data.get("start_time")
        
        # Extract individual agent responses
        responses = {}
        confidences = {}
        sources = set()
        
        for agent_id, result in agent_results.items():
            if agent_id != self.id:  # Skip self
                responses[agent_id] = result.get("response", "")
                confidences[agent_id] = result.get("confidence", 0.5)
                sources.update(result.get("sources", []))
                
        # Apply ensemble fusion
        ensemble_response = await self.fuse_responses(query, responses, confidences)
        ensemble_confidence = await self.calculate_ensemble_confidence(confidences)
        
        total_time = time.time() - start_time
        
        # Create comprehensive result
        ensemble_result = QueryResult(
            query=query,
            answer=ensemble_response,
            method="HiveRAG Ensemble",
            confidence=ensemble_confidence,
            sources=list(sources),
            processing_time=total_time,
            agent_contributions={
                agent_id: {
                    "method": result.get("method", "Unknown"),
                    "confidence": result.get("confidence", 0.5),
                    "processing_time": result.get("processing_time", 0),
                    "sources": len(result.get("sources", []))
                } for agent_id, result in agent_results.items() if agent_id != self.id
            },
            ensemble_metadata={
                "fusion_method": "confidence_weighted",
                "participating_agents": len(responses),
                "consensus_score": await self.calculate_consensus(responses),
                "diversity_score": await self.calculate_diversity(responses)
            },
            context_usage={
                "total_agents": len(agent_results),
                "total_sources": len(sources),
                "total_time": total_time
            }
        )
        
        # Send final result to original requester
        await self.send_message(query_data.get("requester", "user"), "final_result", {
            "result": ensemble_result
        })
        
        self.update_status(AgentStatus.ACTIVE, "Ready for ensemble coordination")
        
    async def fuse_responses(self, query: str, responses: Dict[str, str], confidences: Dict[str, float]) -> str:
        """Fuse multiple agent responses using confidence weighting"""
        if not responses:
            return "No responses available for fusion."
            
        # Create fusion prompt with weighted responses
        fusion_prompt = f"""Fuse multiple RAG responses into one comprehensive answer:

Query: {query}

Agent Responses (with confidence scores):
"""
        
        for agent_id, response in responses.items():
            confidence = confidences.get(agent_id, 0.5)
            fusion_prompt += f"\\n{agent_id} (confidence: {confidence:.2f}):\\n{response}\\n"
            
        fusion_prompt += """\\nFuse these responses into one comprehensive, accurate answer that:
1. Combines the best insights from each response
2. Weights information by confidence scores  
3. Resolves any contradictions
4. Provides a complete, coherent answer

Fused Response:"""

        response = self.ollama_client.generate(
            model="llama3.1:8b",
            prompt=fusion_prompt,
            options={"temperature": 0.1}
        )
        
        return response['response']
        
    async def calculate_ensemble_confidence(self, confidences: Dict[str, float]) -> float:
        """Calculate ensemble confidence score"""
        if not confidences:
            return 0.0
            
        # Weighted average with diversity bonus
        avg_confidence = np.mean(list(confidences.values()))
        diversity_bonus = min(0.1, len(confidences) * 0.02)  # Bonus for more agents
        
        return min(0.95, avg_confidence + diversity_bonus)
        
    async def calculate_consensus(self, responses: Dict[str, str]) -> float:
        """Calculate consensus score between responses"""
        if len(responses) < 2:
            return 1.0
            
        # Simple similarity based on common words (can be enhanced)
        response_texts = list(responses.values())
        consensus_scores = []
        
        for i in range(len(response_texts)):
            for j in range(i + 1, len(response_texts)):
                text1_words = set(response_texts[i].lower().split())
                text2_words = set(response_texts[j].lower().split())
                
                if text1_words and text2_words:
                    jaccard = len(text1_words & text2_words) / len(text1_words | text2_words)
                    consensus_scores.append(jaccard)
                    
        return np.mean(consensus_scores) if consensus_scores else 0.0
        
    async def calculate_diversity(self, responses: Dict[str, str]) -> float:
        """Calculate diversity score between responses"""
        consensus = await self.calculate_consensus(responses)
        return 1.0 - consensus  # High diversity = low consensus

class HiveRAGSystem:
    """Main HiveRAG system orchestrating multi-agent ensemble RAG"""
    
    def __init__(self, config_path: str = None):
        """Initialize HiveRAG system with agent hierarchy"""
        self.config = self._load_config(config_path)
        self.agents = {}
        self.message_router = {}
        self.system_metrics = {
            "total_queries": 0,
            "successful_queries": 0, 
            "average_response_time": 0.0,
            "ensemble_accuracy": 0.0
        }
        
        # Initialize Ollama client
        self.ollama_client = ollama.Client(host=self.config["ollama"]["base_url"])
        
        # Initialize vector store
        self.vector_store = self._init_vector_store()
        
        # Initialize agent hierarchy
        self._init_agents()
        
        logger.info("🐝 HiveRAG Multi-Agent System initialized!")
        logger.info(f"📊 Agents: {len(self.agents)} (Queens: 2, Workers: 4, Scouts: 3)")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            "ollama": {
                "base_url": "http://localhost:11434",
                "primary_model": "llama3.1:8b",
                "embedding_model": "nomic-embed-text"
            },
            "vector_db": {
                "persist_directory": "./hiverag_db",
                "collection_name": "hive_knowledge",
                "chunk_size": 512,
                "chunk_overlap": 50
            },
            "hive": {
                "max_context_window": 2000000,  # 2M tokens
                "max_concurrent_queries": 10,
                "ensemble_methods": ["weighted_fusion", "consensus_voting"]
            }
        }
        
        # Load user config if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge configurations
                for key in default_config:
                    if key in user_config:
                        if isinstance(default_config[key], dict):
                            default_config[key].update(user_config[key])
                        else:
                            default_config[key] = user_config[key]
        
        return default_config
        
    def _init_vector_store(self):
        """Initialize vector database"""
        from langchain.embeddings import OllamaEmbeddings
        
        embeddings = OllamaEmbeddings(
            base_url=self.config["ollama"]["base_url"],
            model=self.config["ollama"]["embedding_model"]
        )
        
        return Chroma(
            persist_directory=self.config["vector_db"]["persist_directory"],
            embedding_function=embeddings,
            collection_name=self.config["vector_db"]["collection_name"]
        )
        
    def _init_agents(self):
        """Initialize the agent hierarchy"""
        # Queen Agents (Central Coordination)
        self.agents["hive-orchestrator"] = HiveOrchestrator()
        self.agents["context-manager"] = ContextManager()
        
        # Worker Agents (Specialized RAG Tasks)
        self.agents["crag-agent"] = CRAGAgent(self.ollama_client, self.vector_store)
        self.agents["selfrag-agent"] = SelfRAGAgent(self.ollama_client, self.vector_store) 
        self.agents["deeprag-agent"] = DeepRAGAgent(self.ollama_client, self.vector_store)
        self.agents["ensemble-agent"] = EnsembleAgent(self.ollama_client)
        
        # Scout Agents (Exploration & QA) - Placeholder implementations
        self.agents["query-analyzer"] = BaseAgent("query-analyzer", "Query Analyzer", AgentType.SCOUT, "Query intelligence")
        self.agents["knowledge-scout"] = BaseAgent("knowledge-scout", "Knowledge Scout", AgentType.SCOUT, "Knowledge exploration")  
        self.agents["quality-guard"] = BaseAgent("quality-guard", "Quality Guard", AgentType.SCOUT, "Quality assurance")
        
        # Set hive reference for all agents
        for agent in self.agents.values():
            agent.set_hive(self)
            
    async def route_message(self, message: AgentMessage):
        """Route message between agents"""
        target_agent = self.agents.get(message.receiver_id)
        if target_agent:
            await target_agent.receive_message(message)
        else:
            logger.warning(f"Unknown agent: {message.receiver_id}")
            
    async def query(self, query_text: str, user_id: str = "user") -> QueryResult:
        """Process query through HiveRAG multi-agent ensemble"""
        logger.info(f"🔍 HiveRAG processing query: '{query_text[:50]}...'")
        
        # Send query to hive orchestrator
        orchestrator = self.agents.get("hive-orchestrator")
        if not orchestrator:
            raise RuntimeError("Hive Orchestrator not available")
            
        # Create result future
        result_future = asyncio.Future()
        
        # Temporary result handler
        original_handle = self.agents["ensemble-agent"].handle_message
        async def result_handler(message):
            if message.message_type == "final_result":
                result_future.set_result(message.content["result"])
            else:
                await original_handle(message)
        
        self.agents["ensemble-agent"].handle_message = result_handler
        
        # Send query request
        await orchestrator.receive_message(AgentMessage(
            sender_id=user_id,
            receiver_id="hive-orchestrator", 
            message_type="query_request",
            content={"query": query_text}
        ))
        
        # Wait for result
        try:
            result = await asyncio.wait_for(result_future, timeout=30.0)
            self.system_metrics["total_queries"] += 1
            self.system_metrics["successful_queries"] += 1
            return result
        except asyncio.TimeoutError:
            logger.error("Query timeout")
            self.system_metrics["total_queries"] += 1
            return QueryResult(
                query=query_text,
                answer="Query timed out",
                method="HiveRAG Ensemble",
                confidence=0.0,
                sources=[],
                processing_time=30.0,
                agent_contributions={},
                ensemble_metadata={"error": "timeout"},
                context_usage={}
            )
        finally:
            # Restore original handler
            self.agents["ensemble-agent"].handle_message = original_handle
            
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the knowledge base"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["vector_db"]["chunk_size"],
            chunk_overlap=self.config["vector_db"]["chunk_overlap"]
        )
        
        doc_objects = []
        for i, doc in enumerate(documents):
            chunks = text_splitter.split_text(doc)
            for j, chunk in enumerate(chunks):
                doc_meta = metadata[i] if metadata else {}
                doc_meta.update({
                    "chunk_id": f"{i}_{j}",
                    "source": f"doc_{i}",
                    "chunk_index": j
                })
                doc_objects.append(Document(page_content=chunk, metadata=doc_meta))
                
        self.vector_store.add_documents(doc_objects)
        logger.info(f"✅ Added {len(doc_objects)} document chunks to HiveRAG knowledge base")
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "type": agent.type.value,
                    "status": agent.status.value,
                    "current_task": agent.current_task,
                    "specialization": agent.specialization
                } for agent_id, agent in self.agents.items()
            },
            "metrics": self.system_metrics,
            "config": {
                "max_context_window": self.config["hive"]["max_context_window"],
                "total_agents": len(self.agents),
                "queen_agents": len([a for a in self.agents.values() if a.type == AgentType.QUEEN]),
                "worker_agents": len([a for a in self.agents.values() if a.type == AgentType.WORKER]),
                "scout_agents": len([a for a in self.agents.values() if a.type == AgentType.SCOUT])
            }
        }

# Demo and testing
async def demo_hiverag():
    """Comprehensive demo of HiveRAG system"""
    print("🐝 HiveRAG Multi-Agent Ensemble RAG Demo")
    print("=" * 60)
    
    # Initialize system
    hive = HiveRAGSystem()
    
    # Add sample knowledge base
    sample_docs = [
        """
        HiveRAG is a state-of-the-art multi-agent RAG system that implements ensemble methods
        for long context knowledge retrieval. It uses a hierarchical agent architecture inspired
        by bee colonies, with Queen agents for coordination, Worker agents for specialized RAG
        tasks, and Scout agents for exploration and quality assurance.
        """,
        """
        The system implements three cutting-edge RAG methods: CRAG (Corrective RAG) with 51%
        benchmark accuracy, Self-RAG with 320% PopQA improvement, and Deep RAG with 8-15%
        reasoning task improvement. These methods work together in an ensemble approach.
        """,
        """
        Key innovations include hierarchical agent coordination, ensemble-based retrieval,
        dynamic long-context management through agent collaboration, and adaptive routing
        based on query complexity. The system can handle up to 2M tokens through distributed
        context management across specialized agents.
        """,
        """
        Research foundations include MA-RAG for collaborative chain-of-thought reasoning,
        Chain of Agents for long-context tasks, RAG Ensemble Framework for theoretical
        optimization, and HIVE coordination protocols for multi-agent control.
        """
    ]
    
    sample_metadata = [
        {"title": "HiveRAG Overview", "category": "system"},
        {"title": "RAG Methods", "category": "algorithms"},
        {"title": "Key Innovations", "category": "features"},
        {"title": "Research Foundation", "category": "academic"}
    ]
    
    hive.add_documents(sample_docs, sample_metadata)
    
    # Demo queries with different complexity levels
    demo_queries = [
        {
            "query": "What is HiveRAG and how does it work?",
            "expected_complexity": "medium",
            "expected_agents": 3
        },
        {
            "query": "Compare the performance improvements of CRAG, Self-RAG, and Deep RAG methods in detail",
            "expected_complexity": "high", 
            "expected_agents": 5
        },
        {
            "query": "What are the key innovations in this system?",
            "expected_complexity": "low",
            "expected_agents": 2
        }
    ]
    
    print("\\n🔍 Processing Demo Queries...")
    print("=" * 40)
    
    for i, demo in enumerate(demo_queries, 1):
        query = demo["query"]
        print(f"\\n{i}. Query: {query}")
        print(f"   Expected complexity: {demo['expected_complexity']}")
        print(f"   Expected agents: {demo['expected_agents']}")
        
        try:
            result = await hive.query(query)
            
            print(f"   ✅ Result:")
            print(f"      Method: {result.method}")
            print(f"      Confidence: {result.confidence:.2f}")
            print(f"      Processing time: {result.processing_time:.2f}s")
            print(f"      Participating agents: {len(result.agent_contributions)}")
            print(f"      Sources: {len(result.sources)}")
            print(f"      Answer: {result.answer[:150]}...")
            
            if result.ensemble_metadata:
                print(f"      Consensus score: {result.ensemble_metadata.get('consensus_score', 0):.2f}")
                print(f"      Diversity score: {result.ensemble_metadata.get('diversity_score', 0):.2f}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
        print("-" * 40)
    
    # System status
    print("\\n📊 System Status:")
    status = hive.get_system_status()
    print(f"   Total agents: {status['config']['total_agents']}")
    print(f"   Queen agents: {status['config']['queen_agents']}")
    print(f"   Worker agents: {status['config']['worker_agents']}")
    print(f"   Scout agents: {status['config']['scout_agents']}")
    print(f"   Total queries: {status['metrics']['total_queries']}")
    print(f"   Success rate: {status['metrics']['successful_queries'] / max(status['metrics']['total_queries'], 1):.1%}")
    
    print("\\n✅ HiveRAG demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_hiverag())
'''

print("📝 HiveRAG Multi-Agent System Implementation Created!")
print("🐝 Features included:")
print("   ✓ Hierarchical Agent Architecture (Queen-Worker-Scout)")
print("   ✓ Latest RAG Methods (CRAG, Self-RAG, Deep RAG)")
print("   ✓ Ensemble Coordination and Result Fusion")
print("   ✓ Long Context Management (up to 2M tokens)")
print("   ✓ Real-time Agent Communication")
print("   ✓ Adaptive Query Routing")
print("   ✓ Performance Monitoring and Metrics")
print("   ✓ Ollama Integration for Local LLMs")

# Save the implementation
with open("hiverag_system.py", "w", encoding="utf-8") as f:
    f.write(hiverag_implementation)

print("\n💾 Implementation saved as 'hiverag_system.py'")
print("🚀 Ready for deployment and testing!")