from app.orchestrator.graph import create_orchestrator

if __name__ == "__main__":
    app_graph = create_orchestrator()
    print("\n=== MULTIMODAL ORCHESTRATOR WORKFLOW ===\n")
    
    # LangGraph provides a built-in method to print the graph structure in ASCII format directly to the terminal!
    app_graph.get_graph().print_ascii()
    
    print("\n========================================\n")
