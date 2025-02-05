import random

import dash
import dash_cytoscape as cyto
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import torch
from dash import Dash
from dash import dcc, html
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

    def save_to_html(self, file_path: str) -> None:
        """
        Saves the graph visualization as an HTML file.

        Args:
            file_path (str): The path to save the HTML file.
        """
        app = self.create_dash_app()
        app.run_server(debug=False, port=8050)
        with open(file_path, "w") as f:
            f.write(app.index_string)


class GraphVisualizer3D:
    def __init__(self, graph, node_predictions, node_ground_truth, subset_size=None, edge_predictions=None,
                 edge_ground_truth=None):
        """
        Initializes the GraphVisualizer3D class.

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

        # Subset graph nodes
        self.subset_nodes = self._subset_nodes()
        self.pos = nx.spring_layout(self.graph, dim=3, seed=42)  # Generate 3D positions for nodes

    def _subset_nodes(self):
        """
        Select a subset of nodes if subset_size is provided.

        Returns:
            set: Subset of nodes to visualize.
        """
        nodes = list(self.graph.nodes)
        if self.subset_size and self.subset_size < len(nodes):
            return set(np.random.choice(nodes, self.subset_size, replace=False))
        return set(nodes)

    def _get_node_traces(self):
        """
        Create 3D scatter trace for nodes.

        Returns:
            go.Scatter3d: Plotly scatter trace for nodes.
        """
        x, y, z, colors, labels = [], [], [], [], []

        for node in self.subset_nodes:
            x.append(self.pos[node][0])
            y.append(self.pos[node][1])
            z.append(self.pos[node][2])

            correct = self.node_predictions[node].item() == self.node_ground_truth[node].item()
            colors.append("green" if correct else "red")
            labels.append(
                f"Node {node}<br>Prediction: {self.node_predictions[node].item()}<br>Ground Truth: {self.node_ground_truth[node].item()}")

        return go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=8, color=colors, opacity=0.8),
            text=labels,
            hoverinfo="text"
        )

    def _get_edge_traces(self):
        """
        Create 3D line traces for edges.

        Returns:
            list: List of Plotly 3D scatter traces for edges.
        """
        edge_traces = []

        for i, (source, target) in enumerate(self.graph.edges):
            if source not in self.subset_nodes or target not in self.subset_nodes:
                continue

            x_coords = [self.pos[source][0], self.pos[target][0], None]
            y_coords = [self.pos[source][1], self.pos[target][1], None]
            z_coords = [self.pos[source][2], self.pos[target][2], None]

            if self.edge_predictions is not None and self.edge_ground_truth is not None:
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
            else:
                # If no edge prediction, use gray for all edges
                color = "gray"

            edge_traces.append(
                go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    mode="lines",
                    line=dict(color=color, width=2),
                    hoverinfo="none"
                )
            )

        return edge_traces

    def create_dash_app(self):
        """
        Creates a Dash app for 3D graph visualization.

        Returns:
            Dash: A Dash app instance.
        """
        node_trace = self._get_node_traces()
        edge_traces = self._get_edge_traces()

        fig = go.Figure(data=[node_trace] + edge_traces)
        fig.update_layout(
            scene=dict(
                bgcolor="black",  # Black background
                xaxis=dict(showbackground=False, gridcolor="white"),
                yaxis=dict(showbackground=False, gridcolor="white"),
                zaxis=dict(showbackground=False, gridcolor="white")
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H4("3D Graph Visualization"),
            dcc.Graph(figure=fig, style={"height": "800px"})
        ])

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
