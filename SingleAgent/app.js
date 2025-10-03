// RAG Application JavaScript
class RAGApplication {
    constructor() {
        this.currentMethod = 'crag';
        this.activeTab = 'crag';
        this.isProcessing = false;
        
        this.ragMethods = {
            crag: {
                name: "Corrective RAG",
                accuracy: "51%",
                improvement: "Up to 320% over standard RAG",
                bestFor: "Robust retrieval with error correction",
                features: ["Document evaluation", "Web search fallback", "Self-correction", "Quality assessment"],
                color: "#4CAF50"
            },
            selfRag: {
                name: "Self-RAG", 
                accuracy: "320% improvement on PopQA",
                improvement: "208% on ARC-Challenge",
                bestFor: "Adaptive retrieval decisions",
                features: ["Reflection tokens", "Dynamic retrieval", "Critique mechanism", "Adaptive reasoning"],
                color: "#2196F3"
            },
            deepRag: {
                name: "Deep RAG",
                accuracy: "8-15% on reasoning tasks", 
                improvement: "End-to-end optimization",
                bestFor: "Complex multi-step reasoning",
                features: ["Decision process", "Strategic retrieval", "Reasoning chain", "Dynamic decisions"],
                color: "#FF9800"
            }
        };

        this.sampleQueries = [
            "What are the latest developments in quantum computing?",
            "Explain the differences between neural network architectures",
            "How do large language models handle context windows?",
            "What are the key challenges in retrieval-augmented generation?"
        ];

        this.performanceData = [
            {method: "CRAG", accuracy: 51, latency: 2.1, precision: 0.82},
            {method: "Self-RAG", accuracy: 48, latency: 1.8, precision: 0.79},
            {method: "DeepRAG", accuracy: 45, latency: 2.5, precision: 0.85}
        ];

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeChart();
        this.setupSettingsControls();
        this.addPlaceholderQuery();
    }

    setupEventListeners() {
        // Method button selection
        document.querySelectorAll('.method-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const method = e.currentTarget.dataset.method;
                this.selectMethod(method);
            });
        });

        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.currentTarget.dataset.tab;
                this.switchTab(tabName);
            });
        });

        // Query execution
        const queryBtn = document.getElementById('queryBtn');
        queryBtn.addEventListener('click', () => this.executeQuery());

        // Enter key on textarea
        const queryInput = document.getElementById('queryInput');
        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.executeQuery();
            }
        });

        // Modal close on backdrop click
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-backdrop')) {
                this.hideModal();
            }
        });
    }

    selectMethod(method) {
        this.currentMethod = method;
        
        // Update button states
        document.querySelectorAll('.method-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-method="${method}"]`).classList.add('active');

        // Switch to corresponding tab
        this.switchTab(method);

        // Update process indicators based on method
        this.updateProcessIndicators(method);
    }

    switchTab(tabName) {
        this.activeTab = tabName;
        
        // Update tab states
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-content`).classList.add('active');
    }

    updateProcessIndicators(method) {
        // Update confidence bars and status indicators with some animation
        const confidenceFills = document.querySelectorAll('.confidence-fill');
        const statusIndicators = document.querySelectorAll('.status-indicator');

        confidenceFills.forEach(fill => {
            const randomValue = 70 + Math.random() * 25; // 70-95%
            setTimeout(() => {
                fill.style.width = `${randomValue}%`;
            }, 100);
        });

        statusIndicators.forEach(indicator => {
            indicator.classList.remove('ready');
            setTimeout(() => {
                indicator.classList.add('ready');
            }, 200);
        });
    }

    async executeQuery() {
        const queryInput = document.getElementById('queryInput');
        const query = queryInput.value.trim();

        if (!query) {
            this.showNotification('Please enter a query', 'error');
            return;
        }

        if (this.isProcessing) {
            return;
        }

        this.isProcessing = true;
        this.showProcessingModal();

        try {
            // Simulate RAG processing steps
            await this.simulateRAGProcessing();
            
            // Generate mock results
            const results = await this.generateMockResults(query);
            
            // Display results
            this.displayResults(results);
            
        } catch (error) {
            this.showNotification('An error occurred during processing', 'error');
        } finally {
            this.isProcessing = false;
            this.hideModal();
        }
    }

    async simulateRAGProcessing() {
        const steps = this.getProcessingSteps(this.currentMethod);
        
        for (let i = 0; i < steps.length; i++) {
            document.getElementById('processingText').textContent = 'Processing Query...';
            document.getElementById('processingDetail').textContent = steps[i];
            await this.delay(800 + Math.random() * 400);
        }
    }

    getProcessingSteps(method) {
        const steps = {
            crag: [
                'Initializing CRAG pipeline',
                'Evaluating document relevance',
                'Performing quality assessment',
                'Triggering web search fallback',
                'Applying self-correction',
                'Generating final response'
            ],
            selfRag: [
                'Initializing Self-RAG pipeline',
                'Analyzing retrieval necessity',
                'Processing reflection tokens',
                'Evaluating document relevance',
                'Applying critique mechanism',
                'Generating adaptive response'
            ],
            deepRag: [
                'Initializing DeepRAG pipeline',
                'Performing initial analysis',
                'Making strategic retrieval decisions',
                'Processing reasoning chain',
                'Integrating multi-step reasoning',
                'Optimizing end-to-end generation'
            ]
        };
        
        return steps[method] || steps.crag;
    }

    async generateMockResults(query) {
        const methodInfo = this.ragMethods[this.currentMethod];
        
        // Generate realistic mock response based on query
        const responses = {
            'quantum computing': 'Quantum computing has seen significant advances in 2024-2025, particularly in error correction and quantum advantage demonstrations. Major breakthroughs include IBM\'s 1000+ qubit processors, Google\'s improved error rates, and Microsoft\'s topological qubit progress.',
            'neural network': 'Neural network architectures vary significantly in design and application. Transformers excel at sequence modeling, CNNs are optimal for spatial data, RNNs handle temporal sequences, and newer architectures like MLP-Mixers challenge traditional assumptions about attention mechanisms.',
            'language models': 'Large language models handle context windows through attention mechanisms and positional encoding. Recent advances include techniques like sliding window attention, sparse attention patterns, and architectural innovations that extend effective context length while maintaining computational efficiency.',
            'retrieval-augmented': 'Key challenges in RAG include retrieval quality, context integration, hallucination mitigation, and computational efficiency. Recent research focuses on corrective mechanisms, adaptive retrieval strategies, and end-to-end optimization of the retrieval-generation pipeline.'
        };

        // Find matching response
        let response = 'Based on the available knowledge base, I can provide insights on your query. ';
        for (const [key, value] of Object.entries(responses)) {
            if (query.toLowerCase().includes(key)) {
                response = value;
                break;
            }
        }

        // Add method-specific enhancements
        if (this.currentMethod === 'crag') {
            response += '\n\n[Document Evaluation: High Confidence] [Self-Correction Applied] [Web Search: Not Required]';
        } else if (this.currentMethod === 'selfRag') {
            response += '\n\n[Retrieve] → [Relevant] → [Generate] → [Critique: Satisfactory]';
        } else if (this.currentMethod === 'deepRag') {
            response += '\n\n→ Initial Analysis → Strategic Retrieval → Reasoning Integration → Final Generation';
        }

        return {
            method: methodInfo.name,
            confidence: 0.85 + Math.random() * 0.1,
            response: response,
            sources: [
                'Research Paper: "Advanced RAG Techniques" (2024)',
                'Knowledge Base: Technical Documentation',
                'External Source: Academic Database'
            ],
            processingTime: (1.8 + Math.random() * 0.8).toFixed(1)
        };
    }

    displayResults(results) {
        const container = document.getElementById('results-container');
        
        container.innerHTML = `
            <div class="result-item">
                <div class="result-header">
                    <span class="result-method">${results.method}</span>
                    <span class="confidence-score">${(results.confidence * 100).toFixed(0)}%</span>
                </div>
                <div class="result-text">${results.response}</div>
                <div class="result-sources">
                    <strong>Sources:</strong><br>
                    ${results.sources.map(source => `<a href="#" class="source-link">${source}</a>`).join('<br>')}
                </div>
                <div class="result-meta" style="margin-top: 12px; font-size: 12px; color: var(--color-text-secondary);">
                    Processing time: ${results.processingTime}s
                </div>
            </div>
        `;

        // Animate the result appearance
        const resultItem = container.querySelector('.result-item');
        resultItem.style.opacity = '0';
        resultItem.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            resultItem.style.transition = 'all 0.5s ease-out';
            resultItem.style.opacity = '1';
            resultItem.style.transform = 'translateY(0)';
        }, 100);
    }

    showProcessingModal() {
        const modal = document.getElementById('processingModal');
        modal.classList.remove('hidden');
    }

    hideModal() {
        const modal = document.getElementById('processingModal');
        modal.classList.add('hidden');
    }

    initializeChart() {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        
        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Speed', 'Precision', 'Robustness', 'Adaptability'],
                datasets: [
                    {
                        label: 'CRAG',
                        data: [51, 75, 82, 90, 70],
                        backgroundColor: 'rgba(76, 175, 80, 0.2)',
                        borderColor: '#4CAF50',
                        pointBackgroundColor: '#4CAF50',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#4CAF50'
                    },
                    {
                        label: 'Self-RAG',
                        data: [48, 85, 79, 75, 95],
                        backgroundColor: 'rgba(33, 150, 243, 0.2)',
                        borderColor: '#2196F3',
                        pointBackgroundColor: '#2196F3',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#2196F3'
                    },
                    {
                        label: 'DeepRAG',
                        data: [45, 65, 85, 80, 85],
                        backgroundColor: 'rgba(255, 152, 0, 0.2)',
                        borderColor: '#FF9800',
                        pointBackgroundColor: '#FF9800',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#FF9800'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            color: '#f5f5f5',
                            font: {
                                size: 12
                            }
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.5)',
                            backdropColor: 'transparent'
                        },
                        beginAtZero: true,
                        max: 100
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#f5f5f5',
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(22, 33, 62, 0.95)',
                        titleColor: '#f5f5f5',
                        bodyColor: '#f5f5f5',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1
                    }
                }
            }
        });
    }

    setupSettingsControls() {
        // Temperature range control
        const tempRange = document.querySelector('.range-input');
        const tempValue = document.querySelector('.range-value');
        
        if (tempRange && tempValue) {
            tempRange.addEventListener('input', (e) => {
                tempValue.textContent = e.target.value;
            });
        }

        // Model selector changes
        const modelSelect = document.querySelectorAll('select')[0];
        if (modelSelect) {
            modelSelect.addEventListener('change', () => {
                this.showNotification('Model configuration updated', 'success');
            });
        }
    }

    addPlaceholderQuery() {
        const queryInput = document.getElementById('queryInput');
        const randomQuery = this.sampleQueries[Math.floor(Math.random() * this.sampleQueries.length)];
        queryInput.placeholder = `Try asking: "${randomQuery}"`;
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification--${type}`;
        notification.innerHTML = `
            <i class="fas fa-${this.getNotificationIcon(type)}"></i>
            <span>${message}</span>
        `;
        
        // Style the notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '12px 16px',
            borderRadius: '8px',
            color: 'white',
            fontWeight: '500',
            zIndex: '1001',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            minWidth: '250px',
            animation: 'slideIn 0.3s ease-out',
            backgroundColor: this.getNotificationColor(type)
        });

        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || icons.info;
    }

    getNotificationColor(type) {
        const colors = {
            success: '#27ae60',
            error: '#e74c3c',
            warning: '#f39c12',
            info: '#2980b9'
        };
        return colors[type] || colors.info;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Add notification animations to CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new RAGApplication();
});

// Add some dynamic updates to simulate real-time data
setInterval(() => {
    // Update confidence bars with slight variations
    const confidenceFills = document.querySelectorAll('.confidence-fill');
    confidenceFills.forEach(fill => {
        const currentWidth = parseInt(fill.style.width) || 85;
        const variation = (Math.random() - 0.5) * 10; // ±5%
        const newWidth = Math.max(70, Math.min(95, currentWidth + variation));
        fill.style.width = `${newWidth}%`;
    });
}, 15000); // Update every 15 seconds

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.altKey) {
        switch(e.key) {
            case '1':
                e.preventDefault();
                document.querySelector('[data-method="crag"]').click();
                break;
            case '2':
                e.preventDefault();
                document.querySelector('[data-method="selfRag"]').click();
                break;
            case '3':
                e.preventDefault();
                document.querySelector('[data-method="deepRag"]').click();
                break;
        }
    }
});