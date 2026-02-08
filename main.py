from app.pipelines.graph_pipeline import generate_knowledge_graph

if __name__ == "__main__":
    html_path, graph_docs, _ = generate_knowledge_graph("data/papers/sample.pdf", is_path=True)
    print("HTML:", html_path)
    print("Nodes:", len(graph_docs[0].nodes))
    print("Rels:", len(graph_docs[0].relationships))
