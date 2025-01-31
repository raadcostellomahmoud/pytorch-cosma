
def prepare_edge_inputs(node_embeddings, edge_index):
    # Extract source and target node embeddings
    source_embeddings = node_embeddings[edge_index[0]]  # Source nodes
    target_embeddings = node_embeddings[edge_index[1]]  # Target nodes
    return source_embeddings, target_embeddings
