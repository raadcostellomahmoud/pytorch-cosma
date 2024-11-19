import base64
import io

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from dash import dcc, html
from matplotlib.figure import Figure
from umap import UMAP


class LatentSpaceExplorer:
    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device

    def extract_latent_space(self, siamese=False):
        self.model.eval()
        latent_points = []
        labels_points = []
        all_inputs = []

        with torch.no_grad():
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.cpu().numpy()
                if siamese:
                    latent_space = self.model.get_latent_features(inputs)
                else:
                    latent_space, _ = self.model(inputs, return_latent=True)
                latent_points.extend(latent_space.cpu().numpy())
                labels_points.extend(labels)
                all_inputs.extend(inputs.cpu())


        return np.array(latent_points), np.array(labels_points), torch.stack(all_inputs)

    def reduce_dimensionality(self, latent_points, n_components=2):
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
