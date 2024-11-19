import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config_validation import ConfigModel
from src.latent_space import LatentSpaceExplorer, Visualizer
from src.model_yaml_parser import YamlParser
from src.network_construction import BaseModel

classification_loss_weight = 0.1

# Define device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])  # , transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Load configuration from YAML
raw_config = YamlParser(
    "configs/example_conv_autoencoder_1.yaml").parse()  # raw_config is a dictionary from the YAML file

# Validate configuration
try:
    validated_config = ConfigModel(**raw_config).to_dict()
except ValueError as e:
    print("Configuration validation failed:", e)
    exit(1)

# Create model from configuration
model = BaseModel(validated_config)

# Train the model
model.train_model(epochs=1, train_loader=train_loader,
                  loss_function=nn.MSELoss(), use_reconstruction=True,
                  optimizer=torch.optim.Adam(model.parameters(), lr=1e-3), )

# Latent space exploration
explorer = LatentSpaceExplorer(model, train_loader, device)
latent_points, labels_points, all_inputs = explorer.extract_latent_space()
reduced_dimensionality = explorer.reduce_dimensionality(latent_points)

# Randomly sample points for visualization
sample_size = 100
indices = np.random.choice(len(reduced_dimensionality), size=sample_size, replace=False)
reduced_dimensionality = reduced_dimensionality[indices]
labels_points = labels_points[indices]
selected_inputs = all_inputs[indices]

# Visualize
visualizer = Visualizer(reduced_dimensionality, labels_points, selected_inputs)
hover_images = visualizer.generate_hover_images()
app = visualizer.create_dash_app(hover_images)
app.run_server(debug=True)

# # Initialize model, loss, and optimizer
# model = BaseModel(validated_config).to(device)
# loss_function = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# epochs = 1
# # Training loop
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     total_classification_loss = 0
#
#     for batch in train_loader:
#         inputs, labels = batch  # We don't need labels for autoencoder training
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#
#         optimizer.zero_grad()
#
#         # Forward pass
#         reconstructed, logits = model(inputs)
#
#         # Calculate loss
#         reconstruction_loss = loss_function(reconstructed, inputs)  # Comparing reconstructed image with input image
#         classification_loss = nn.functional.cross_entropy(logits, labels)
#         total_loss = reconstruction_loss + classification_loss * classification_loss_weight
#
#         total_loss.backward()  # Backpropagation
#         optimizer.step()  # Update weights
#
#         total_loss += total_loss.item()
#         total_classification_loss += classification_loss.item()
#
#     print(
#         f"Epoch [{epoch + 1}/{epochs}], Total Loss: {total_loss / len(train_loader)}, Total Classification Loss: {total_classification_loss / len(train_loader)}")
#
# print("Training complete.")
#
# # Plotting original vs reconstructed images
# model.eval()
# with torch.no_grad():
#     # Retrieve a single batch from the test set
#     for inputs, _ in test_loader:
#         inputs = inputs.to(device)
#
#         # Get the reconstructed images from the autoencoder
#         reconstructed, _ = model(inputs)
#
#         # Move tensors back to the CPU and detach them for visualization
#         inputs = inputs.cpu().detach()
#         reconstructed = reconstructed.cpu().detach()
#
#         # Plot a few original and reconstructed images side by side
#         num_images = 6
#         fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
#         for i in range(num_images):
#             # Original image
#             axes[0, i].imshow(inputs[i].squeeze(), cmap='gray')
#             axes[0, i].set_title("Original")
#             axes[0, i].axis("off")
#
#             # Reconstructed image
#             axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
#             axes[1, i].set_title("Reconstructed")
#             axes[1, i].axis("off")
#
#         plt.show()
#         break  # Only visualize the first batch
#
# model.eval()
# latent_points = []
# labels_points = []
# all_inputs = []  # New list to store all inputs
#
# with torch.no_grad():
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.cpu().numpy()
#
#         # Get latent space representations
#         latent_space, _ = model(inputs, return_latent=True)
#         latent_space = latent_space.cpu().numpy()
#
#         latent_points.extend(latent_space)
#         labels_points.extend(labels)
#         all_inputs.extend(inputs.cpu())  # Collect inputs (on CPU for later processing)
#
# # Convert collected data to arrays
# latent_points = np.array(latent_points)
# labels_points = np.array(labels_points)
# all_inputs = torch.stack(all_inputs)  # Convert list to a single tensor
#
# # Use UMAP for dimensionality reduction
# reducer = UMAP(n_components=2, random_state=42)
# reduced_dimensionality = reducer.fit_transform(latent_points)
#
# # Randomly sample 100 points
# sample_size = 100
# indices = np.random.choice(len(reduced_dimensionality), size=sample_size, replace=False)
#
# # Subset the data
# reduced_dimensionality = reduced_dimensionality[indices]
# labels_points = labels_points[indices]
# selected_inputs = all_inputs[indices]  # Use all_inputs to get the corresponding images
# hover_images = []
# for img in selected_inputs:
#     img_cpu = img.squeeze().numpy()
#
#     fig = Figure(figsize=(0.5, 0.5))
#     ax = fig.subplots()
#     ax.imshow(img_cpu, cmap='gray')
#     ax.axis('off')
#
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", bbox_inches='tight')
#     buf.seek(0)
#     img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#     hover_images.append(f"data:image/png;base64,{img_base64}")
#
# # Data preparation
# data = pd.DataFrame({
#     "x": reduced_dimensionality[:, 0],
#     "y": reduced_dimensionality[:, 1],
#     "label": labels_points,
#     "hover_image": hover_images,  # Add the base64 images
# })
#
# # Initialize Dash app
# app = dash.Dash(__name__)
#
# # Create the scatter plot
# fig = px.scatter(
#     data,
#     x="x",
#     y="y",
#     color="label",
#     title="Latent Space with Hoverable MNIST Images"
# )
#
# # Remove default hover behavior for customization
# fig.update_traces(hoverinfo="none", hovertemplate=None)
#
# # Layout with the scatter plot and a hidden image
# app.layout = html.Div([
#     dcc.Graph(
#         id="scatter-plot",
#         figure=fig,
#         config={"displayModeBar": False},  # Disable unnecessary UI elements
#         style={"height": "80vh"}
#     ),
#     html.Div(
#         id="hover-image",
#         style={
#             "position": "absolute",
#             "display": "none",  # Initially hidden
#             "pointerEvents": "none",  # Prevent interaction
#             "zIndex": "1000",
#         },
#     ),
# ])
#
#
# @app.callback(
#     [
#         dash.Output("hover-image", "style"),
#         dash.Output("hover-image", "children"),
#     ],
#     [dash.Input("scatter-plot", "hoverData")]
# )
# def display_hover_image(hoverData):
#     if hoverData is None:
#         return {"display": "none"}, None
#
#     # Extract point index and image source
#     try:
#         point_index = hoverData["points"][0].get("pointIndex", 0)
#         img_src = data.loc[point_index, "hover_image"]
#
#         # Extract bounding box for the hovered point
#         bbox = hoverData["points"][0]["bbox"]
#         x_cursor = bbox["x0"]
#         y_cursor = bbox["y0"]
#     except (KeyError, IndexError):
#         return {"display": "none"}, None
#
#     # Position the hover image near the cursor
#     style = {
#         "position": "absolute",
#         "left": f"{x_cursor + 20}px",  # Offset for visibility
#         "top": f"{y_cursor + 20}px",
#         "display": "block",
#         "pointerEvents": "none",
#         "zIndex": "1000",
#     }
#
#     return style, html.Img(src=img_src, style={"maxHeight": "100px", "maxWidth": "100px"})


# Run the app
# if __name__ == "__main__":
#     app.run_server(debug=True)
