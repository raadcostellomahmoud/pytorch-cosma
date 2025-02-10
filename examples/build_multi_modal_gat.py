import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch_geometric.data import Data, Batch

from pytorch_cosma.config_validation import ConfigModel
from pytorch_cosma.model_yaml_parser import YamlParser
from pytorch_cosma.network_construction import MultiModalGATModel

# Define device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, num_nodes=25, num_edges=50, img_size=(3, 128, 128), seq_len=1000, num_classes=51):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.img_size = img_size
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.data_list = self.generate_data()

    def generate_data(self):
        data_list = []
        for _ in range(self.num_nodes):
            x_images = torch.randn(1, *self.img_size)  # Adjusted shape
            x_one_hot = torch.randint(0, 2, (1, self.seq_len, 4)).float()  # Adjusted shape
            node_labels = torch.randint(0, 2, (self.num_classes,)).float().unsqueeze(0)  # Adjusted shape
            edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
            edge_attr = torch.randint(1, 10, (self.num_edges,)).float()
            data = Data(x_images=x_images, x_one_hot=x_one_hot, y=node_labels, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# Custom collate function for DataLoader
def collate_fn(batch):
    return Batch.from_data_list(batch)

# Load configuration and model
raw_config = YamlParser("configs/example_multimodal_gat.yaml").parse()
validated_config = ConfigModel(**raw_config).to_dict()
model_class = validated_config.pop("model_class")
model = globals()[model_class](validated_config, device=device)

# Create DataLoader
dataset = DummyDataset()
train_loader = DataLoader(dataset.data_list, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9,
                                                       patience=10, verbose=True)

criterion_node = nn.BCELoss()  # Regression for trait values

JUST_INFER_CONNECTED = True
if JUST_INFER_CONNECTED:
    criterion_edge = nn.HuberLoss(delta=200.0)  # Regression for edge overlap prediction
else:
    criterion_edge = nn.BCELoss()

# Train the model
model.train_model(
    train_loader=train_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion_node=criterion_node,
    criterion_edge=criterion_edge,
    device=device,
    num_epochs=10,
    edge_loss_weighting=1,
    epoch_report_interval=5,
    just_infer_connected=True
)