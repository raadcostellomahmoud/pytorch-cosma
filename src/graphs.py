import random

import dash
import dash_cytoscape as cyto
import torch
from dash import html
from torch_geometric.utils import negative_sampling


class GraphVisualizer:
    def __init__(self, graph, predictions, ground_truth, subset_size=None):
        """
        Initializes the GraphVisualizer class.

        Args:
            graph (networkx.Graph): The graph to visualize.
            predictions (torch.Tensor): Model predictions for each node.
            ground_truth (torch.Tensor): Ground truth labels for each node.
            subset_size (int, optional): Number of nodes to visualize. If None, visualize all nodes.
        """
        self.graph = graph
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.subset_size = subset_size
        self.elements = self._prepare_elements()

    def _prepare_elements(self):
        """
        Prepares the Cytoscape elements for visualization.

        Returns:
            list: List of elements for Cytoscape visualization.
        """
        nodes = list(self.graph.nodes)
        if self.subset_size and self.subset_size < len(nodes):
            nodes = random.sample(nodes, self.subset_size)

        elements = []
        # Add nodes
        for node in nodes:
            correct = self.predictions[node].item() == self.ground_truth[node].item()
            elements.append(
                {
                    "data": {
                        "id": str(node),
                        "label": f"Node {node}",
                        "color": "green" if correct else "red",
                    }
                }
            )
        # Add edges
        for source, target in self.graph.edges:
            if source in nodes and target in nodes:
                elements.append(
                    {
                        "data": {
                            "source": str(source),
                            "target": str(target),
                        }
                    }
                )
        return elements

    def create_dash_app(self):
        """
        Creates a Dash app for graph visualization.

        Returns:
            dash.Dash: A Dash app instance.
        """
        app = dash.Dash(__name__)
        app.layout = html.Div(
            [
                html.H4(f"Graph Visualization (Subset of {self.subset_size or 'All'} Nodes)"),
                cyto.Cytoscape(
                    id="cytoscape",
                    elements=self.elements,
                    style={"width": "100%", "height": "800px"},
                    layout={"name": "cose"},  # 'cose' layout positions nodes dynamically
                    stylesheet=[
                        {
                            "selector": "node",
                            "style": {
                                "background-color": "data(color)",
                                "label": "data(label)",
                            },
                        },
                        {
                            "selector": "edge",
                            "style": {
                                "line-color": "#ccc",
                            },
                        },
                    ],
                ),
            ]
        )
        return app


def prepare_edge_labels(data):
    pos_edge_label = torch.ones(data.train_pos_edge_index.size(1))  # Positive edges
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,  # Positive edges
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),
    )
    neg_edge_label = torch.zeros(neg_edge_index.size(1))  # Negative edges

    # Combine positive and negative edges
    edge_index = torch.cat([data.train_pos_edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([pos_edge_label, neg_edge_label], dim=0)

    # Shuffle edges and labels
    perm = torch.randperm(edge_index.size(1))
    data.edge_index = edge_index[:, perm]
    data.edge_label = edge_label[perm]
    return data
