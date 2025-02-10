import yaml

class PruneConfig:
    """Configuration for model pruning"""
    def __init__(self, 
                 amount: float = 0.2,
                 method: str = 'l1_unstructured',
                 layers_to_prune: list[str] = None,
                 global_pruning: bool = True):
        self.amount = amount  # Fraction of connections to prune (0-1)
        self.method = method  # 'l1_unstructured', 'random_unstructured', 'ln_structured'
        self.layers_to_prune = layers_to_prune or ['Conv2d', 'Linear']
        self.global_pruning = global_pruning  # Whether to prune across layers

class YamlParser:
    def __init__(self, config_path):
        self.config_path = config_path

    def parse(self):
        with open(self.config_path) as file:
            config = yaml.safe_load(file)
            
        # Add pruning config if present
        if 'pruning' in config:
            config['pruning'] = PruneConfig(**config['pruning'])
            
        return config

    def validate(self, config):
        """
        Validates the parsed YAML configuration.

        Args:
            config (dict): The parsed YAML configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        required_keys = ["layers", "model_class"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")
