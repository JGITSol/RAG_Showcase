# Create HiveRAG Multi-Agent System Architecture Flowchart
diagram_code = """
flowchart TD
    %% Queen Agents (Top Level)
    HO["`**Hive Orchestrator**
    Master Coordinator`"]
    CM["`**Context Manager**  
    Long Context Coord
    2M tokens capacity`"]
    
    %% Worker Agents (Middle Level)
    CRAG["`**CRAG Agent**
    Corrective RAG
    51% accuracy`"]
    SELF["`**Self-RAG Agent**
    Adaptive RAG  
    320% improvement`"]
    DEEP["`**Deep RAG Agent**
    Reasoning RAG
    8-15% improvement`"]
    ENS["`**Ensemble Agent**
    Result Fusion
    92% accuracy`"]
    
    %% Scout Agents (Bottom Level)
    QA["`**Query Analyzer**
    Query Intelligence`"]
    KS["`**Knowledge Scout**
    Knowledge Discovery`"]
    QG["`**Quality Guard**
    Quality Assurance`"]
    
    %% System Performance Box
    PERF["`**System Performance**
    Response Time: 2.3s avg
    Communication: <100ms
    Efficiency: +10%`"]
    
    %% Connections from Queen to Workers
    HO --> CRAG
    HO --> SELF  
    HO --> DEEP
    HO --> ENS
    
    CM --> CRAG
    CM --> SELF
    CM --> DEEP
    
    %% Connections from Workers to Scouts
    CRAG --> QA
    SELF --> KS
    DEEP --> QG
    ENS --> QA
    ENS --> KS
    ENS --> QG
    
    %% Performance connection
    ENS --> PERF
    
    %% Styling with hive-inspired colors
    classDef queen fill:#FFD700,stroke:#333,stroke-width:3px,color:#000
    classDef worker fill:#FFB000,stroke:#333,stroke-width:2px,color:#000
    classDef scout fill:#FF8C00,stroke:#333,stroke-width:2px,color:#000
    classDef perf fill:#4CAF50,stroke:#333,stroke-width:2px,color:#000
    
    class HO,CM queen
    class CRAG,SELF,DEEP,ENS worker  
    class QA,KS,QG scout
    class PERF perf
"""

# Create the mermaid diagram
png_path, svg_path = create_mermaid_diagram(
    diagram_code, 
    png_filepath='hiverag_architecture.png',
    svg_filepath='hiverag_architecture.svg',
    width=1400,
    height=1000
)

print(f"HiveRAG Multi-Agent System Architecture saved to:")
print(f"PNG: {png_path}")
print(f"SVG: {svg_path}")