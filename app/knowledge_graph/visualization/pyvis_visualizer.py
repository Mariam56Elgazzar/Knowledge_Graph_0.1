from __future__ import annotations
import os
from typing import List
from pyvis.network import Network
from langchain_experimental.graph_transformers.llm import GraphDocument

def visualize_graph(graph_documents: List[GraphDocument], output_file: str) -> str:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    net = Network(height="1000px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222")
    graph = graph_documents[0]

    for node in graph.nodes:
        net.add_node(node.id, label=node.id, title=node.type, color="#b9d9ea")

    for rel in graph.relationships:
        net.add_edge(rel.source.id, rel.target.id, label=rel.type, color="#97c2fc")

    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 110,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "enabled": true, "iterations": 900 }
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": true,
        "hover": true,
        "zoomView": true
      }
    }
    """)
    net.save_graph(output_file)
    return output_file
