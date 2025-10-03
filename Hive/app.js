// HiveRAG Multi-Agent RAG System
class HiveRAGSystem {
    constructor() {
        this.agentData = {
            "queenAgents": [
                {
                    "id": "hive-orchestrator",
                    "name": "Hive Orchestrator",
                    "role": "Master Coordinator",
                    "status": "active",
                    "currentTask": "Query decomposition and agent assignment",
                    "specialization": "Global optimization and strategic planning",
                    "performance": {"accuracy": 0.89, "efficiency": 0.92, "reliability": 0.95}
                },
                {
                    "id": "context-manager", 
                    "name": "Context Manager",
                    "role": "Long Context Coordinator",
                    "status": "active", 
                    "currentTask": "Managing 1.2M token context across 8 agents",
                    "specialization": "Long context optimization",
                    "performance": {"contextUtilization": 0.87, "memoryEfficiency": 0.91, "scalability": 0.94}
                }
            ],
            "workerAgents": [
                {
                    "id": "crag-agent",
                    "name": "CRAG Agent",
                    "role": "Corrective RAG Specialist", 
                    "status": "processing",
                    "currentTask": "Document quality evaluation and self-correction",
                    "specialization": "Robust retrieval with error correction",
                    "performance": {"accuracy": 0.91, "precision": 0.87, "recall": 0.89},
                    "benchmarkScore": "51% CRAG benchmark accuracy"
                },
                {
                    "id": "selfrag-agent",
                    "name": "Self-RAG Agent",
                    "role": "Adaptive RAG Specialist",
                    "status": "active",
                    "currentTask": "Reflection token analysis and retrieval decisions", 
                    "specialization": "Efficient adaptive retrieval",
                    "performance": {"adaptability": 0.93, "efficiency": 0.95, "tokenOptimization": 0.88},
                    "benchmarkScore": "320% PopQA improvement"
                },
                {
                    "id": "deeprag-agent", 
                    "name": "Deep RAG Agent",
                    "role": "Reasoning RAG Specialist",
                    "status": "processing",
                    "currentTask": "Multi-step reasoning chain construction",
                    "specialization": "Complex reasoning and analysis", 
                    "performance": {"reasoningDepth": 0.86, "strategicRetrieval": 0.84, "complexity": 0.92},
                    "benchmarkScore": "8-15% reasoning improvement"
                },
                {
                    "id": "ensemble-agent",
                    "name": "Ensemble Agent", 
                    "role": "Multi-method Aggregator",
                    "status": "active",
                    "currentTask": "Result fusion and consensus building",
                    "specialization": "Ensemble coordination",
                    "performance": {"consensus": 0.91, "aggregation": 0.89, "optimization": 0.87}
                }
            ],
            "scoutAgents": [
                {
                    "id": "query-analyzer",
                    "name": "Query Analyzer",
                    "role": "Query Intelligence",
                    "status": "idle",
                    "currentTask": "Monitoring for new queries",
                    "specialization": "Query understanding and routing",
                    "performance": {"intentDetection": 0.94, "complexityAnalysis": 0.88, "routing": 0.92}
                },
                {
                    "id": "knowledge-scout",
                    "name": "Knowledge Scout", 
                    "role": "Knowledge Discovery",
                    "status": "active",
                    "currentTask": "Knowledge graph updates and pattern recognition",
                    "specialization": "Knowledge landscape exploration",
                    "performance": {"discovery": 0.85, "mapping": 0.87, "patternRecognition": 0.91}
                },
                {
                    "id": "quality-guard",
                    "name": "Quality Guard",
                    "role": "Quality Assurance",
                    "status": "monitoring",
                    "currentTask": "Real-time quality monitoring",
                    "specialization": "Quality assurance and validation", 
                    "performance": {"validation": 0.93, "factChecking": 0.89, "trustScoring": 0.91}
                }
            ]
        };

        this.systemMetrics = {
            "totalAgents": 9,
            "activeAgents": 6,
            "processingAgents": 2,
            "idleAgents": 1,
            "averagePerformance": 0.897,
            "systemLoad": 0.73,
            "memoryUtilization": 0.68,
            "contextWindowUsage": "1.2M / 2M tokens",
            "ensembleAccuracy": 0.924,
            "responseTime": "2.3s average"
        };

        this.communicationFlows = [
            {"from": "hive-orchestrator", "to": "query-analyzer", "message": "New query received", "timestamp": "2025-09-30T00:15:23Z"},
            {"from": "query-analyzer", "to": "hive-orchestrator", "message": "Complex multi-hop query detected", "timestamp": "2025-09-30T00:15:24Z"},
            {"from": "hive-orchestrator", "to": "crag-agent", "message": "Assign retrieval task A", "timestamp": "2025-09-30T00:15:25Z"},
            {"from": "hive-orchestrator", "to": "selfrag-agent", "message": "Assign adaptive retrieval B", "timestamp": "2025-09-30T00:15:25Z"},
            {"from": "context-manager", "to": "deeprag-agent", "message": "Context segment 3 ready", "timestamp": "2025-09-30T00:15:26Z"}
        ];

        this.longContextMetrics = {
            "totalContextWindow": "2M tokens",
            "currentUtilization": "1.2M tokens", 
            "segments": [
                {"agent": "crag-agent", "allocation": "240K tokens", "efficiency": 0.89},
                {"agent": "selfrag-agent", "allocation": "180K tokens", "efficiency": 0.94},
                {"agent": "deeprag-agent", "allocation": "320K tokens", "efficiency": 0.86},
                {"agent": "ensemble-agent", "allocation": "160K tokens", "efficiency": 0.91},
                {"agent": "context-manager", "allocation": "300K tokens", "efficiency": 0.88}
            ]
        };

        this.currentMethod = 'ensemble';
        this.isProcessing = false;
        this.performanceChart = null;
        this.messageCount = 247;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.renderHiveVisualization();
        this.renderCommunicationFlow();
        this.renderContextAllocation();
        this.initializePerformanceChart();
        this.updateSystemMetrics();
        this.startRealTimeUpdates();
        this.setupQueryComplexityAnalysis();
    }

    setupEventListeners() {
        // Method tab selection
        document.querySelectorAll('.method-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const method = e.currentTarget.dataset.method;
                this.selectMethod(method);
            });
        });

        // Execute query button
        document.getElementById('executeQuery').addEventListener('click', () => {
            this.executeQuery();
        });

        // Query input for complexity analysis
        document.getElementById('queryInput').addEventListener('input', (e) => {
            this.analyzeQueryComplexity(e.target.value);
        });

        // Agent node interactions
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-backdrop')) {
                this.hideModal();
            }
        });
    }

    selectMethod(method) {
        this.currentMethod = method;
        
        // Update tab states
        document.querySelectorAll('.method-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-method="${method}"]`).classList.add('active');

        // Update agent highlights based on method
        this.highlightRelevantAgents(method);
    }

    highlightRelevantAgents(method) {
        const agentNodes = document.querySelectorAll('.agent-node');
        agentNodes.forEach(node => {
            node.classList.remove('highlighted');
        });

        // Highlight relevant agents based on method
        const relevantAgents = {
            'ensemble': ['hive-orchestrator', 'ensemble-agent', 'context-manager'],
            'crag': ['crag-agent', 'quality-guard'],
            'selfrag': ['selfrag-agent', 'query-analyzer'],
            'deeprag': ['deeprag-agent', 'knowledge-scout']
        };

        const agents = relevantAgents[method] || [];
        agents.forEach(agentId => {
            const node = document.querySelector(`[data-agent-id="${agentId}"]`);
            if (node) {
                node.classList.add('highlighted');
            }
        });
    }

    renderHiveVisualization() {
        const hiveContainer = document.getElementById('hiveVisualization');
        let html = '';

        // Render Queen Agents
        this.agentData.queenAgents.forEach(agent => {
            html += this.createAgentNode(agent, 'queen');
        });

        // Render Worker Agents
        this.agentData.workerAgents.forEach(agent => {
            html += this.createAgentNode(agent, 'worker');
        });

        // Render Scout Agents
        this.agentData.scoutAgents.forEach(agent => {
            html += this.createAgentNode(agent, 'scout');
        });

        hiveContainer.innerHTML = html;

        // Add click handlers for agent details
        this.setupAgentInteractions();
    }

    createAgentNode(agent, type) {
        const icons = {
            'queen': 'fas fa-crown',
            'worker': 'fas fa-hammer',
            'scout': 'fas fa-search'
        };

        const icon = icons[type] || 'fas fa-robot';
        
        return `
            <div class="agent-node ${type}" data-agent-id="${agent.id}" title="${agent.currentTask}">
                <div class="agent-status ${agent.status}"></div>
                <div class="agent-icon">
                    <i class="${icon}"></i>
                </div>
                <div class="agent-name">${agent.name}</div>
            </div>
        `;
    }

    setupAgentInteractions() {
        document.querySelectorAll('.agent-node').forEach(node => {
            node.addEventListener('click', (e) => {
                const agentId = e.currentTarget.dataset.agentId;
                this.showAgentDetails(agentId);
            });
        });
    }

    showAgentDetails(agentId) {
        // Find agent in data
        let agent = null;
        ['queenAgents', 'workerAgents', 'scoutAgents'].forEach(type => {
            const found = this.agentData[type].find(a => a.id === agentId);
            if (found) agent = found;
        });

        if (!agent) return;

        // Create a simple notification with agent details
        this.showNotification(`
            <strong>${agent.name}</strong><br>
            Role: ${agent.role}<br>
            Status: ${agent.status}<br>
            Task: ${agent.currentTask}
        `, 'info', 5000);
    }

    renderCommunicationFlow() {
        const flowContainer = document.getElementById('communicationFlow');
        let html = '';

        this.communicationFlows.forEach(flow => {
            const timeAgo = this.getTimeAgo(flow.timestamp);
            html += `
                <div class="comm-message">
                    <span class="comm-from">${this.getAgentDisplayName(flow.from)}</span>
                    <i class="fas fa-arrow-right"></i>
                    <span class="comm-to">${this.getAgentDisplayName(flow.to)}</span>
                    <span class="comm-message-text">${flow.message}</span>
                    <span class="comm-timestamp">${timeAgo}</span>
                </div>
            `;
        });

        flowContainer.innerHTML = html;
    }

    getAgentDisplayName(agentId) {
        const allAgents = [
            ...this.agentData.queenAgents,
            ...this.agentData.workerAgents,
            ...this.agentData.scoutAgents
        ];
        
        const agent = allAgents.find(a => a.id === agentId);
        return agent ? agent.name.split(' ')[0] : agentId;
    }

    getTimeAgo(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diff = Math.floor((now - time) / 1000);
        
        if (diff < 60) return `${diff}s ago`;
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        return `${Math.floor(diff / 3600)}h ago`;
    }

    renderContextAllocation() {
        const allocationContainer = document.getElementById('contextAllocation');
        let html = '';

        this.longContextMetrics.segments.forEach(segment => {
            const agentName = this.getAgentDisplayName(segment.agent);
            const efficiency = Math.round(segment.efficiency * 100);
            
            html += `
                <div class="context-segment">
                    <span class="segment-agent">${agentName}</span>
                    <span class="segment-allocation">${segment.allocation} (${efficiency}%)</span>
                </div>
            `;
        });

        allocationContainer.innerHTML = html;

        // Update context usage bar
        const contextUsed = document.getElementById('contextUsed');
        if (contextUsed) {
            contextUsed.style.width = '60%'; // 1.2M / 2M = 60%
        }
    }

    initializePerformanceChart() {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        
        this.performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['00:10', '00:11', '00:12', '00:13', '00:14', '00:15'],
                datasets: [
                    {
                        label: 'Ensemble',
                        data: [0.91, 0.92, 0.93, 0.92, 0.94, 0.95],
                        borderColor: '#ffd700',
                        backgroundColor: 'rgba(255, 215, 0, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'CRAG',
                        data: [0.87, 0.89, 0.91, 0.90, 0.92, 0.93],
                        borderColor: '#4caf50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        tension: 0.4,
                        fill: false
                    },
                    {
                        label: 'Self-RAG',
                        data: [0.93, 0.94, 0.95, 0.93, 0.96, 0.97],
                        borderColor: '#2196f3',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        tension: 0.4,
                        fill: false
                    },
                    {
                        label: 'Deep RAG',
                        data: [0.85, 0.86, 0.87, 0.88, 0.89, 0.90],
                        borderColor: '#ff8c00',
                        backgroundColor: 'rgba(255, 140, 0, 0.1)',
                        tension: 0.4,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0.8,
                        max: 1.0,
                        ticks: {
                            color: '#ffffff',
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    x: {
                        ticks: {
                            color: '#ffffff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff',
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 26, 46, 0.95)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#ffd700',
                        borderWidth: 1
                    }
                }
            }
        });
    }

    updateSystemMetrics() {
        document.getElementById('activeAgents').textContent = this.systemMetrics.activeAgents;
        document.getElementById('ensembleAccuracy').textContent = (this.systemMetrics.ensembleAccuracy * 100).toFixed(1) + '%';
        document.getElementById('contextUsage').textContent = this.longContextMetrics.currentUtilization.split(' ')[0];
        
        // Update hive stats
        document.getElementById('queenAgents').textContent = this.agentData.queenAgents.length;
        document.getElementById('workerAgents').textContent = this.agentData.workerAgents.length;
        document.getElementById('scoutAgents').textContent = this.agentData.scoutAgents.length;
    }

    analyzeQueryComplexity(query) {
        if (!query.trim()) {
            document.getElementById('complexityFill').style.width = '0%';
            document.getElementById('complexityValue').textContent = 'Low';
            return;
        }

        // Simple complexity analysis
        let complexity = 0;
        const words = query.split(' ').length;
        const questions = (query.match(/\?/g) || []).length;
        const technical = (query.toLowerCase().match(/\b(quantum|neural|algorithm|machine learning|artificial intelligence|blockchain|cryptocurrency)\b/g) || []).length;
        
        complexity += Math.min(words / 10, 0.3); // Length factor
        complexity += questions * 0.2; // Question complexity
        complexity += technical * 0.15; // Technical terms

        complexity = Math.min(complexity, 1);

        const percentage = Math.round(complexity * 100);
        document.getElementById('complexityFill').style.width = percentage + '%';
        
        if (complexity < 0.3) {
            document.getElementById('complexityValue').textContent = 'Low';
        } else if (complexity < 0.7) {
            document.getElementById('complexityValue').textContent = 'Medium';
        } else {
            document.getElementById('complexityValue').textContent = 'High';
        }
    }

    async executeQuery() {
        const queryInput = document.getElementById('queryInput');
        const query = queryInput.value.trim();

        if (!query) {
            this.showNotification('Please enter a query to engage the hive intelligence', 'warning');
            return;
        }

        if (this.isProcessing) {
            return;
        }

        this.isProcessing = true;
        this.showProcessingModal();
        this.startWorkflowAnimation();

        try {
            await this.simulateHiveProcessing(query);
            const results = await this.generateEnsembleResults(query);
            this.displayResults(results);
        } catch (error) {
            this.showNotification('Hive processing error occurred', 'error');
        } finally {
            this.isProcessing = false;
            this.hideModal();
            this.resetWorkflow();
        }
    }

    showProcessingModal() {
        const modal = document.getElementById('processingModal');
        modal.classList.remove('hidden');
        
        // Update processing agents display
        const processingAgents = document.getElementById('processingAgents');
        const activeAgents = this.getActiveAgentsForMethod(this.currentMethod);
        
        let html = '';
        activeAgents.forEach(agent => {
            html += `<div class="processing-agent">${agent}</div>`;
        });
        processingAgents.innerHTML = html;
    }

    hideModal() {
        const modal = document.getElementById('processingModal');
        modal.classList.add('hidden');
    }

    getActiveAgentsForMethod(method) {
        const methodAgents = {
            'ensemble': ['Hive Orchestrator', 'Ensemble Agent', 'CRAG Agent', 'Self-RAG Agent', 'Deep RAG Agent'],
            'crag': ['CRAG Agent', 'Quality Guard'],
            'selfrag': ['Self-RAG Agent', 'Query Analyzer'],
            'deeprag': ['Deep RAG Agent', 'Knowledge Scout']
        };
        
        return methodAgents[method] || methodAgents.ensemble;
    }

    startWorkflowAnimation() {
        const steps = document.querySelectorAll('.workflow-step');
        steps.forEach((step, index) => {
            setTimeout(() => {
                step.querySelector('.step-status').classList.remove('idle');
                step.querySelector('.step-status').classList.add('active');
            }, index * 800);
        });
    }

    resetWorkflow() {
        const steps = document.querySelectorAll('.workflow-step');
        steps.forEach((step, index) => {
            setTimeout(() => {
                step.querySelector('.step-status').classList.remove('active');
                step.querySelector('.step-status').classList.add('complete');
            }, index * 200);
        });

        setTimeout(() => {
            steps.forEach(step => {
                step.querySelector('.step-status').classList.remove('complete');
                step.querySelector('.step-status').classList.add('idle');
            });
        }, 3000);
    }

    async simulateHiveProcessing(query) {
        const steps = [
            'Analyzing query complexity and intent...',
            'Routing to specialized agent swarm...',
            'Coordinating parallel retrieval processes...',
            'Performing ensemble result aggregation...',
            'Applying consensus mechanisms...',
            'Finalizing hive intelligence synthesis...'
        ];

        const processingTitle = document.getElementById('processingTitle');
        const processingMessage = document.getElementById('processingMessage');

        for (let i = 0; i < steps.length; i++) {
            processingMessage.textContent = steps[i];
            await this.delay(1000 + Math.random() * 500);
        }
    }

    async generateEnsembleResults(query) {
        // Generate realistic ensemble results based on query
        const queryLower = query.toLowerCase();
        let response = '';
        let confidence = 0.85 + Math.random() * 0.1;
        let sources = [];

        // Generate contextual response
        if (queryLower.includes('quantum')) {
            response = 'Based on ensemble analysis across multiple RAG agents, quantum computing shows unprecedented advances in 2024-2025. The hive intelligence indicates breakthrough developments in quantum error correction, with IBM achieving 1000+ qubit stability, Google demonstrating quantum supremacy in optimization problems, and emerging quantum advantage in cryptographic applications.';
            sources = ['Quantum Computing Review 2025', 'IBM Quantum Network Papers', 'Google AI Research Publications'];
        } else if (queryLower.includes('neural') || queryLower.includes('machine learning')) {
            response = 'The agent ensemble reveals significant evolution in neural architectures. Transformer variants continue to dominate, while new paradigms like liquid neural networks and neuromorphic computing gain traction. Our multi-agent analysis shows convergence on hybrid architectures combining symbolic reasoning with connectionist approaches.';
            sources = ['Neural Architecture Survey 2025', 'MIT CSAIL Research', 'DeepMind Technical Reports'];
        } else if (queryLower.includes('rag') || queryLower.includes('retrieval')) {
            response = 'Ensemble analysis of retrieval-augmented generation reveals three dominant paradigms: Corrective RAG (51% benchmark improvement), Self-RAG (320% PopQA enhancement), and Deep RAG (8-15% reasoning boost). Our hive intelligence synthesizes these approaches for optimal performance across diverse query types.';
            sources = ['CRAG Research Paper', 'Self-RAG Technical Documentation', 'Deep RAG Methodology'];
        } else {
            response = `Hive ensemble processing of your query reveals multifaceted insights. Our specialized agent swarm has coordinated across ${this.getActiveAgentsForMethod(this.currentMethod).length} agents to provide comprehensive analysis. The consensus mechanism indicates high confidence in the synthesized response based on distributed knowledge retrieval and reasoning.`;
            sources = ['Knowledge Base Synthesis', 'Multi-Agent Consensus', 'Distributed Reasoning Network'];
        }

        // Add method-specific enhancements
        if (this.currentMethod === 'crag') {
            response += '\n\n[CRAG Enhancement: Document relevance verified, self-correction applied, web search integration confirmed]';
        } else if (this.currentMethod === 'selfrag') {
            response += '\n\n[Self-RAG Tokens: [Retrieve] → [Relevant] → [Generate] → [Critique: High Quality] → [Support]]';
        } else if (this.currentMethod === 'deeprag') {
            response += '\n\n[Deep RAG Chain: Query Decomposition → Strategic Retrieval → Multi-step Reasoning → Evidence Integration]';
        } else {
            response += '\n\n[Ensemble Synthesis: CRAG (92% confidence) + Self-RAG (96% relevance) + Deep RAG (88% reasoning depth)]';
        }

        return {
            method: this.currentMethod.toUpperCase(),
            confidence: confidence,
            response: response,
            sources: sources,
            processingTime: (1.5 + Math.random() * 1.0).toFixed(1),
            agentsUsed: this.getActiveAgentsForMethod(this.currentMethod)
        };
    }

    displayResults(results) {
        const resultsContainer = document.getElementById('resultsContainer');
        
        const html = `
            <div class="result-item">
                <div class="result-header">
                    <div class="result-method">${results.method} Ensemble Result</div>
                    <div class="result-confidence">${Math.round(results.confidence * 100)}% Confidence</div>
                </div>
                <div class="result-content">${results.response}</div>
                <div class="result-meta" style="margin-bottom: 16px;">
                    <strong>Active Agents:</strong> ${results.agentsUsed.join(', ')}
                </div>
                <div class="result-sources">
                    <strong>Knowledge Sources:</strong><br>
                    ${results.sources.map(source => `• ${source}`).join('<br>')}
                </div>
                <div style="margin-top: 12px; font-size: 12px; color: rgba(255,255,255,0.5);">
                    Processing time: ${results.processingTime}s | Tokens used: ${Math.round(Math.random() * 200 + 100)}K
                </div>
            </div>
        `;

        resultsContainer.innerHTML = html;
    }

    startRealTimeUpdates() {
        // Update communication flow every 5 seconds
        setInterval(() => {
            this.addNewCommunicationMessage();
        }, 5000);

        // Update performance metrics every 10 seconds
        setInterval(() => {
            this.updatePerformanceData();
        }, 10000);

        // Update agent status every 3 seconds
        setInterval(() => {
            this.updateAgentStatus();
        }, 3000);
    }

    addNewCommunicationMessage() {
        const messages = [
            { from: 'hive-orchestrator', to: 'ensemble-agent', message: 'Consensus threshold reached' },
            { from: 'context-manager', to: 'crag-agent', message: 'Context window optimized' },
            { from: 'quality-guard', to: 'selfrag-agent', message: 'Quality validation passed' },
            { from: 'knowledge-scout', to: 'deeprag-agent', message: 'New knowledge patterns detected' }
        ];

        const randomMessage = messages[Math.floor(Math.random() * messages.length)];
        randomMessage.timestamp = new Date().toISOString();
        
        this.communicationFlows.unshift(randomMessage);
        this.communicationFlows = this.communicationFlows.slice(0, 5); // Keep only recent messages
        
        this.renderCommunicationFlow();
        
        // Update message count
        this.messageCount += Math.floor(Math.random() * 3) + 1;
        document.getElementById('messageCount').textContent = this.messageCount;
    }

    updatePerformanceData() {
        if (this.performanceChart) {
            // Add new data point
            const newTime = new Date().toLocaleTimeString('en-US', { 
                hour12: false, 
                hour: '2-digit', 
                minute: '2-digit' 
            });
            
            this.performanceChart.data.labels.push(newTime);
            this.performanceChart.data.labels = this.performanceChart.data.labels.slice(-6);
            
            // Update each dataset with slight variations
            this.performanceChart.data.datasets.forEach(dataset => {
                const lastValue = dataset.data[dataset.data.length - 1];
                const variation = (Math.random() - 0.5) * 0.03; // ±1.5% variation
                const newValue = Math.max(0.8, Math.min(1.0, lastValue + variation));
                
                dataset.data.push(newValue);
                dataset.data = dataset.data.slice(-6);
            });
            
            this.performanceChart.update('none');
        }
    }

    updateAgentStatus() {
        // Randomly update some agent statuses
        const allAgents = [
            ...this.agentData.queenAgents,
            ...this.agentData.workerAgents,
            ...this.agentData.scoutAgents
        ];

        // Update a random agent
        const randomAgent = allAgents[Math.floor(Math.random() * allAgents.length)];
        const statuses = ['active', 'processing', 'idle'];
        const currentStatus = randomAgent.status;
        const newStatus = statuses.filter(s => s !== currentStatus)[Math.floor(Math.random() * 2)];
        
        randomAgent.status = newStatus;
        
        // Update the DOM
        const agentNode = document.querySelector(`[data-agent-id="${randomAgent.id}"]`);
        if (agentNode) {
            const statusEl = agentNode.querySelector('.agent-status');
            statusEl.className = `agent-status ${newStatus}`;
        }

        // Update system metrics
        const activeCount = allAgents.filter(a => a.status === 'active').length;
        const processingCount = allAgents.filter(a => a.status === 'processing').length;
        
        this.systemMetrics.activeAgents = activeCount + processingCount;
        document.getElementById('activeAgents').textContent = this.systemMetrics.activeAgents;
    }

    setupQueryComplexityAnalysis() {
        // Add sample queries for demonstration
        const sampleQueries = [
            "What are the latest developments in quantum computing?",
            "Analyze the impact of transformer architectures on NLP tasks",
            "How do ensemble methods improve RAG performance?",
            "Explain the differences between CRAG, Self-RAG, and Deep RAG"
        ];

        const queryInput = document.getElementById('queryInput');
        queryInput.placeholder = `Try asking: "${sampleQueries[Math.floor(Math.random() * sampleQueries.length)]}"`;
    }

    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification notification--${type}`;
        notification.innerHTML = message;
        
        Object.assign(notification.style, {
            position: 'fixed',
            top: '80px',
            right: '20px',
            padding: '16px 20px',
            borderRadius: '8px',
            color: 'white',
            fontWeight: '500',
            zIndex: '1001',
            minWidth: '300px',
            maxWidth: '400px',
            boxShadow: '0 8px 25px rgba(0,0,0,0.3)',
            animation: 'slideInRight 0.3s ease-out',
            backgroundColor: this.getNotificationColor(type),
            border: '1px solid rgba(255,255,255,0.2)'
        });

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, duration);
    }

    getNotificationColor(type) {
        const colors = {
            success: '#4caf50',
            error: '#f44336',
            warning: '#ff9800',
            info: '#2196f3'
        };
        return colors[type] || colors.info;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Add notification animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .agent-node.highlighted {
        border-color: #ffd700 !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.4) !important;
        transform: scale(1.05);
    }
`;
document.head.appendChild(style);

// Initialize the HiveRAG system
document.addEventListener('DOMContentLoaded', () => {
    new HiveRAGSystem();
});

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        switch(e.key) {
            case 'Enter':
                e.preventDefault();
                document.getElementById('executeQuery').click();
                break;
            case '1':
                e.preventDefault();
                document.querySelector('[data-method="ensemble"]').click();
                break;
            case '2':
                e.preventDefault();
                document.querySelector('[data-method="crag"]').click();
                break;
            case '3':
                e.preventDefault();
                document.querySelector('[data-method="selfrag"]').click();
                break;
            case '4':
                e.preventDefault();
                document.querySelector('[data-method="deeprag"]').click();
                break;
        }
    }
});