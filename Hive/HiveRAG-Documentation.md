# HiveRAG: Multi-Agent Ensemble RAG System

## Overview

HiveRAG is a state-of-the-art multi-agent RAG (Retrieval-Augmented Generation) system that implements the latest research in ensemble methods and hive intelligence for long context knowledge retrieval. The system is designed around a hierarchical agent architecture inspired by bee colonies, featuring specialized agents that collaborate to achieve superior performance on complex knowledge-intensive tasks.

## 🏆 Key Research Foundations

This implementation is based on cutting-edge research from 2025:

### Primary Research Papers
- **MA-RAG**: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning
- **Chain of Agents**: Large Language Models Collaborating on Long-Context Tasks (10% improvement over full-context)
- **RAG Ensemble Framework**: Theoretical and Mechanistic Analysis (2-11% accuracy improvement)
- **HIVE**: Harnessing Language for Coordination Multi-Agent Control (up to 2,000 agents coordination)
- **HM-RAG**: Hierarchical Multi-Agent Multimodal RAG (12.95% improvement in answer accuracy)
- **MAIN-RAG**: Multi-Agent Filtering RAG (93.43% question-answering accuracy)

### Performance Benchmarks
- **CRAG**: 51% accuracy on CRAG benchmark, 320% improvement over standard RAG
- **Self-RAG**: 320% improvement on PopQA, 208% on ARC-Challenge
- **Deep RAG**: 8-15% improvement on reasoning-intensive tasks
- **Ensemble Methods**: 2-11% improvement over single RAG approaches

## 🏗️ Architecture Overview

### Agent Hierarchy

The system implements a three-tier hierarchical architecture:

#### 👑 Queen Agents (Central Coordination)
- **Hive Orchestrator**: Master coordinator managing query decomposition, agent assignment, and result synthesis
- **Context Manager**: Long context coordinator handling up to 2M tokens across distributed agents

#### 🐛 Worker Agents (Specialized RAG Tasks)
- **CRAG Agent**: Corrective RAG specialist with document evaluation and self-correction
- **Self-RAG Agent**: Adaptive RAG specialist using reflection tokens for dynamic retrieval decisions
- **Deep RAG Agent**: Reasoning RAG specialist implementing multi-step reasoning chains
- **Ensemble Agent**: Multi-method aggregator performing result fusion and consensus building

#### 🔍 Scout Agents (Exploration & Quality Assurance)
- **Query Analyzer**: Query understanding, complexity analysis, and routing intelligence
- **Knowledge Scout**: Knowledge discovery, pattern recognition, and landscape exploration
- **Quality Guard**: Quality assurance, fact-checking, and trust scoring

### Communication Protocol

Agents communicate through structured message passing:
- **Message Types**: task_assignment, agent_result, status_update, context_request
- **Priority System**: High/Medium/Low priority message routing
- **Async Processing**: Non-blocking message handling for scalability

## 🚀 Key Innovations

### 1. Hierarchical Agent Coordination
- **Queen-Worker-Scout Pattern**: Inspired by bee colony organization
- **Dynamic Task Assignment**: Agents assigned based on query complexity and requirements
- **Emergent Collaboration**: Collective intelligence beyond individual agent capabilities

### 2. Ensemble-Based Retrieval
- **Multi-Method Integration**: CRAG, Self-RAG, and Deep RAG working together
- **Confidence Weighting**: Results weighted by agent confidence scores
- **Consensus Building**: Agreement-based result validation and fusion

### 3. Long Context Management
- **Distributed Context**: 2M token capacity distributed across specialized agents
- **Strategic Segmentation**: Context allocated based on agent specialization
- **Attention Coordination**: Coordinated attention mechanisms across agents

### 4. Adaptive Query Routing
- **Complexity Analysis**: Automatic query complexity assessment
- **Agent Selection**: Dynamic agent assignment based on query characteristics
- **Resource Optimization**: Efficient resource utilization based on requirements

## 📊 Performance Improvements

Based on research benchmarks and expected improvements:

| Method | Accuracy Improvement | Best Use Case | Key Innovation |
|--------|---------------------|---------------|----------------|
| **CRAG** | 51% benchmark accuracy | Robust retrieval with uncertain data | Self-correction + web fallback |
| **Self-RAG** | 320% PopQA improvement | Efficient adaptive retrieval | Reflection tokens + dynamic decisions |
| **Deep RAG** | 8-15% reasoning improvement | Complex multi-step analysis | Strategic retrieval + reasoning chains |
| **Ensemble** | 2-11% over single methods | Production systems | Multi-method fusion + consensus |

### Expected System Performance
- **Accuracy**: 2-11% improvement over single RAG methods
- **Long Context**: Up to 2M tokens through agent collaboration  
- **Efficiency**: 10% improvement over full-context approaches
- **Scalability**: Linear scaling vs quadratic transformer costs
- **Reliability**: Fault tolerance through agent redundancy

## 💻 Technical Implementation

### Core Technologies
- **Python 3.9+**: Main implementation language
- **Ollama**: Local LLM serving and integration
- **LangChain**: Document processing and embeddings
- **ChromaDB**: Vector database for knowledge storage
- **NetworkX**: Agent communication graph management
- **AsyncIO**: Asynchronous agent coordination

### Agent Implementation
```python
# Hierarchical agent base class
class BaseAgent:
    def __init__(self, agent_id, name, agent_type, specialization):
        self.id = agent_id
        self.name = name
        self.type = agent_type  # QUEEN, WORKER, SCOUT
        self.specialization = specialization
        self.status = AgentStatus.IDLE
        self.message_queue = deque()
        
    async def send_message(self, receiver_id, message_type, content):
        # Inter-agent communication via hive
        
    async def process_task(self, task_data):
        # Agent-specific task processing
```

### Ensemble Coordination
```python
class EnsembleAgent:
    async def fuse_responses(self, query, responses, confidences):
        # Confidence-weighted response fusion
        # Consensus building and conflict resolution
        # Quality assessment and validation
```

### Long Context Management
```python
class ContextManager:
    async def allocate_context(self, agent_id, tokens_requested):
        # Dynamic context allocation
        # Memory optimization and compression
        # Attention coordination across agents
```

## 🔧 Installation and Setup

### Prerequisites
1. **Python 3.9+** installed
2. **Ollama** installed and running
3. **Git** for repository management

### Step 1: Install Ollama
```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from ollama.ai
```

### Step 2: Pull Required Models
```bash
# Primary LLM model
ollama pull llama3.1:8b

# Embedding model
ollama pull nomic-embed-text

# Verify models
ollama list
```

### Step 3: Setup Python Environment
```bash
# Create virtual environment
python -m venv hiverag_env
source hiverag_env/bin/activate  # Linux/macOS
# or: hiverag_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Initialize HiveRAG System
```python
from hiverag_system import HiveRAGSystem
import asyncio

async def main():
    # Initialize the hive
    hive = HiveRAGSystem()
    
    # Add knowledge documents
    documents = ["Your knowledge base content..."]
    hive.add_documents(documents)
    
    # Process query through multi-agent ensemble
    result = await hive.query("Your complex question here")
    
    print(f"Method: {result.method}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Answer: {result.answer}")
    print(f"Agents involved: {len(result.agent_contributions)}")

# Run the system
asyncio.run(main())
```

## 🎯 Usage Examples

### Basic Query Processing
```python
# Simple query (low complexity)
result = await hive.query("What is machine learning?")
# Expected: Self-RAG agent only

# Medium complexity query  
result = await hive.query("Compare different neural network architectures")
# Expected: CRAG + Self-RAG agents

# High complexity query
result = await hive.query("Analyze the implications of quantum computing on cryptographic security across multiple research domains")
# Expected: All agents (CRAG, Self-RAG, Deep RAG) + Ensemble coordination
```

### Performance Monitoring
```python
# Get system status
status = hive.get_system_status()
print(f"Active agents: {sum(1 for a in status['agents'].values() if a['status'] == 'active')}")
print(f"Total queries processed: {status['metrics']['total_queries']}")
print(f"Success rate: {status['metrics']['successful_queries'] / status['metrics']['total_queries']:.1%}")
```

### Agent-Specific Results
```python
# Examine individual agent contributions
for agent_id, contribution in result.agent_contributions.items():
    print(f"{agent_id}: {contribution['method']} - {contribution['confidence']:.2f} confidence")
```

## 📈 Performance Monitoring

### Real-time Metrics
- **Agent Status**: Active, processing, idle, error states
- **Query Performance**: Response time, accuracy, confidence scores
- **Ensemble Effectiveness**: Consensus scores, diversity metrics
- **Context Utilization**: Memory usage, allocation efficiency

### System Dashboard
The included web dashboard provides:
- **Hive Visualization**: Real-time agent communication and status
- **Performance Charts**: Historical accuracy and response time trends
- **Agent Activity**: Individual agent task monitoring
- **Query Analytics**: Complexity analysis and routing decisions

## 🔬 Research Applications

### Academic Use Cases
- **RAG Method Comparison**: Benchmark different approaches on same datasets
- **Multi-Agent Coordination**: Study emergent behaviors in agent collaboration
- **Long Context Processing**: Research distributed attention mechanisms
- **Ensemble Learning**: Investigate optimal fusion strategies

### Production Applications
- **Enterprise Knowledge Systems**: Large-scale document repositories
- **Customer Support**: Multi-source information integration
- **Research Assistance**: Complex multi-step analysis tasks
- **Legal & Compliance**: Comprehensive regulatory analysis

## 🛠️ Configuration Options

### System Configuration
```json
{
  "ollama": {
    "base_url": "http://localhost:11434",
    "primary_model": "llama3.1:8b",
    "embedding_model": "nomic-embed-text"
  },
  "hive": {
    "max_context_window": 2000000,
    "max_concurrent_queries": 10,
    "ensemble_methods": ["weighted_fusion", "consensus_voting"]
  },
  "vector_db": {
    "persist_directory": "./hiverag_db",
    "collection_name": "hive_knowledge",
    "chunk_size": 512,
    "chunk_overlap": 50
  }
}
```

### Agent Specialization Parameters
- **CRAG Agent**: Evaluation threshold (0.7), correction attempts (3)
- **Self-RAG Agent**: Reflection tokens, critique thresholds
- **Deep RAG Agent**: Max reasoning steps (5), complexity levels
- **Ensemble Agent**: Fusion methods, confidence weighting

## 🔍 Advanced Features

### Custom Agent Development
```python
class CustomRAGAgent(BaseAgent):
    def __init__(self, specialization):
        super().__init__(
            agent_id="custom-agent",
            name="Custom RAG Agent",
            agent_type=AgentType.WORKER,
            specialization=specialization
        )
    
    async def handle_message(self, message):
        # Custom task processing logic
```

### Multi-Modal Extensions
The architecture supports extension to multi-modal inputs:
- **Image Processing**: Visual question answering agents
- **Audio Analysis**: Speech and audio content agents  
- **Video Understanding**: Multi-frame analysis agents

### Distributed Deployment
```python
# Multiple Ollama instances
hive_config = {
    "ollama_endpoints": [
        "http://server1:11434",
        "http://server2:11434", 
        "http://server3:11434"
    ]
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Ollama Connection Errors**
   ```bash
   # Verify Ollama is running
   ollama serve
   ollama list
   ```

2. **Memory Issues with Large Context**
   ```python
   # Reduce context window
   config["hive"]["max_context_window"] = 1000000  # 1M tokens
   ```

3. **Agent Communication Timeouts**
   ```python
   # Increase timeout settings
   result = await hive.query(query, timeout=60.0)
   ```

4. **Vector Database Issues**
   ```bash
   # Reset vector database
   rm -rf ./hiverag_db
   ```

### Performance Optimization
- **Model Selection**: Use appropriate model sizes for hardware
- **Context Allocation**: Balance context distribution across agents
- **Batch Processing**: Group similar queries for efficiency
- **Caching**: Implement result caching for repeated queries

## 📚 Research Citations

When using HiveRAG in academic work, please cite the foundational research:

```bibtex
@article{nguyen2025marag,
  title={MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning},
  author={Nguyen, Thang and Chin, Peter and Tai, Yu-Wing},
  journal={arXiv preprint arXiv:2505.20096},
  year={2025}
}

@article{zhang2024chainofagents,
  title={Chain of Agents: Large Language Models Collaborating on Long-Context Tasks},
  author={Zhang, Yusen and Sun, Ruoxi and Chen, Yanfei},
  journal={arXiv preprint arXiv:2406.02818},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions to enhance HiveRAG:

### Areas for Enhancement
- **New RAG Methods**: Implement additional state-of-the-art RAG approaches
- **Agent Specializations**: Develop domain-specific agents
- **Evaluation Metrics**: Enhanced performance measurement tools
- **Multi-Modal Support**: Visual and audio processing capabilities
- **Distributed Computing**: Scale across multiple machines

### Development Process
1. Fork the repository
2. Create feature branch
3. Implement enhancements with tests
4. Submit pull request with detailed description

## 📄 License

This implementation is provided for research and educational purposes. Commercial use requires appropriate licensing of underlying technologies (Ollama, LangChain, etc.).

## 🆘 Support

For technical support and questions:
1. Check the troubleshooting section above
2. Review configuration parameters
3. Verify Ollama model availability
4. Test with the provided demo script

---

## 🌟 Future Roadmap

### Planned Enhancements
- **Graph RAG Integration**: Knowledge graph-based retrieval
- **Real-time Learning**: Continuous improvement from feedback
- **Multi-Language Support**: Non-English knowledge bases
- **Cloud Deployment**: Kubernetes orchestration
- **API Gateway**: RESTful API for integration

### Research Directions
- **Emergent Behaviors**: Study of collective intelligence patterns
- **Adaptive Specialization**: Dynamic agent role evolution
- **Context Compression**: Advanced long-context optimization
- **Meta-Learning**: System-level learning and adaptation

HiveRAG represents the cutting edge of multi-agent RAG systems, combining the latest research insights with practical implementation for real-world knowledge retrieval challenges.