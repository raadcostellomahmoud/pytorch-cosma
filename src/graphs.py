import random

import dash
import dash_cytoscape as cyto
import torch
from dash import html
from torch_geometric.utils import negative_sampling


class GraphVisualizer:
    def __init__(self, graph, node_predictions, node_ground_truth, subset_size=None, edge_predictions=None,
                 edge_ground_truth=None):
        """
        Initializes the GraphVisualizer class.

        Args:
            graph (networkx.Graph): The graph to visualize.
            node_predictions (torch.Tensor): Node-level predictions.
            node_ground_truth (torch.Tensor): Ground truth labels for nodes.
            subset_size (int, optional): Number of nodes to visualize. If None, visualize all nodes.
            edge_predictions (torch.Tensor, optional): Edge-level predictions (logits or probabilities).
            edge_ground_truth (torch.Tensor, optional): Ground truth labels for edges.
        """
        self.graph = graph
        self.node_predictions = node_predictions
        self.node_ground_truth = node_ground_truth
        self.edge_predictions = edge_predictions
        self.edge_ground_truth = edge_ground_truth
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
            correct = self.node_predictions[node].item() == self.node_ground_truth[node].item()
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
        if self.edge_predictions is not None and self.edge_ground_truth is not None:
            # Case when edge predictions and ground truth are provided
            for i, (source, target) in enumerate(self.graph.edges):
                if source in nodes and target in nodes:
                    pred = self.edge_predictions[i] > 0  # Predicted as positive
                    true = self.edge_ground_truth[i].bool()  # Ground truth

                    # Skip true negatives
                    if not pred and not true:
                        continue

                    # Determine color for remaining edge types
                    if pred and true:
                        color = "green"  # Correct prediction
                    elif pred and not true:
                        color = "blue"  # False positive
                    elif not pred and true:
                        color = "red"  # False negative

                    elements.append(
                        {
                            "data": {
                                "source": str(source),
                                "target": str(target),
                            },
                            "style": {"line-color": color},
                        }
                    )
        else:
            # Case when only node predictions are provided, visualize all ground truth edges in gray
            for source, target in self.graph.edges:
                if source in nodes and target in nodes:
                    elements.append(
                        {
                            "data": {
                                "source": str(source),
                                "target": str(target),
                            },
                            "style": {"line-color": "gray"},
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
                                "line-color": "data(color)",
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
