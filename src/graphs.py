import random

import dash
import dash_cytoscape as cyto
from dash import html


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
