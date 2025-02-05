import base64
import io

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from dash import dcc, html
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from umap import UMAP


class LatentSpaceExplorer:
    """
    A utility class to extract the latent space of a model and reduce its dimensionality.

    This class extracts latent space representations from a trained model, reduces their dimensionality
    using UMAP, and prepares the data for visualization.

    Args:
        model (torch.nn.Module): The trained model to explore.
        train_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): The device on which the model and data reside.
    """

    def __init__(self, model: torch.nn.Module, train_loader: DataLoader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device

    def extract_latent_space(self, twin: bool = False) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """
        Extracts latent space representations from the model.

        Args:
            twin (bool, optional): Whether the model is a twin network. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, torch.Tensor]:
                - Latent points as a NumPy array.
                - Corresponding labels as a NumPy array.
                - Original input images as a stacked PyTorch tensor.
        """
        self.model.eval()
        latent_points = []
        labels_points = []
        all_inputs = []

        with torch.no_grad():
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.cpu().numpy()
                if twin:
                    latent_space = self.model.get_latent_features(inputs)
                else:
                    latent_space, _ = self.model(inputs, return_latent=True)
                latent_points.extend(latent_space.cpu().numpy())
                labels_points.extend(labels)
                all_inputs.extend(inputs.cpu())

        return np.array(latent_points), np.array(labels_points), torch.stack(all_inputs)

    def reduce_dimensionality(self, latent_points: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Reduces the dimensionality of latent space points using UMAP.

        Args:
            latent_points (np.ndarray): The latent space points to reduce.
            n_components (int, optional): The number of dimensions to reduce to. Defaults to 2.

        Returns:
            np.ndarray: The reduced dimensionality points.
        """
        reducer = UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(latent_points)


class Visualizer:
    def __init__(self, reduced_dimensionality, labels_points, selected_inputs):
        self.reduced_dimensionality = reduced_dimensionality
        self.labels_points = labels_points
        self.selected_inputs = selected_inputs

    def generate_hover_images(self):
        hover_images = []
        for img in self.selected_inputs:
            img_cpu = img.squeeze().numpy()
            fig = Figure(figsize=(0.5, 0.5))
            ax = fig.subplots()
            ax.imshow(img_cpu, cmap='gray')
            ax.axis('off')

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            hover_images.append(f"data:image/png;base64,{img_base64}")
        return hover_images

    def create_dash_app(self, hover_images):
        data = pd.DataFrame({
            "x": self.reduced_dimensionality[:, 0],
            "y": self.reduced_dimensionality[:, 1],
            "label": self.labels_points,
            "hover_image": hover_images,
        })

        fig = px.scatter(data, x="x", y="y", color="label", title="Latent Space with Hoverable MNIST Images")
        fig.update_traces(hoverinfo="none", hovertemplate=None)

        app = dash.Dash(__name__)

        app.layout = html.Div([
            dcc.Graph(
                id="scatter-plot",
                figure=fig,
                config={"displayModeBar": False},
                style={"height": "80vh"}
            ),
            html.Div(
                id="hover-image",
                style={
                    "position": "absolute",
                    "display": "none",
                    "pointerEvents": "none",
                    "zIndex": "1000",
                },
            ),
        ])

        @app.callback(
            [
                dash.Output("hover-image", "style"),
                dash.Output("hover-image", "children"),
            ],
            [dash.Input("scatter-plot", "hoverData")]
        )
        def display_hover_image(hoverData):
            if hoverData is None:
                return {"display": "none"}, None

            try:
                point_index = hoverData["points"][0].get("pointIndex", 0)
                img_src = data.loc[point_index, "hover_image"]
                bbox = hoverData["points"][0]["bbox"]
                x_cursor = bbox["x0"]
                y_cursor = bbox["y0"]
            except (KeyError, IndexError):
                return {"display": "none"}, None

            style = {
                "position": "absolute",
                "left": f"{x_cursor + 20}px",
                "top": f"{y_cursor + 20}px",
                "display": "block",
                "pointerEvents": "none",
                "zIndex": "1000",
            }

            return style, html.Img(src=img_src, style={"maxHeight": "100px", "maxWidth": "100px"})

        return app
