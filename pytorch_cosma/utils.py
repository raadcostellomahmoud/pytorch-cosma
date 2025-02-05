
import torch
from torch.utils.data import Dataset


import random


def prepare_edge_inputs(node_embeddings, edge_index):
    # Extract source and target node embeddings
    source_embeddings = node_embeddings[edge_index[0]]  # Source nodes
    target_embeddings = node_embeddings[edge_index[1]]  # Target nodes
    return source_embeddings, target_embeddings


class TwinDatasetFromDataset(Dataset):
    def __init__(self, dataset):
        """
        Args:
            dataset (torchvision.datasets): E.g. the MNIST dataset or any dataset with labels.
        """
        self.dataset = dataset

        # Precompute indices grouped by label
        self.label_to_indices = {label: [] for label in range(10)}  # Assuming 10 classes (0-9)
        for idx, (_, label) in enumerate(dataset):
            self.label_to_indices[label].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Generate a twin pair.

        Args:
            idx (int): Index of the primary image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (img1, img2, is_same)
        """
        img1, label1 = self.dataset[idx]

        # Randomly decide if we want a "same" or "different" pair
        is_same = random.choice([0, 1])
        if is_same:
            # Choose a random index with the same label
            idx2 = random.choice(self.label_to_indices[label1])
        else:
            # Choose a random label that's different
            different_label = random.choice([label for label in self.label_to_indices if label != label1])
            idx2 = random.choice(self.label_to_indices[different_label])

        img2, label2 = self.dataset[idx2]

        return (img1, img2, torch.tensor(is_same, dtype=torch.float32))
