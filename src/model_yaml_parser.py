import yaml

class YamlParser:
    def __init__(self, config_path):
        self.config_path = config_path

    def parse(self):
        with open(self.config_path) as file:
            config = yaml.safe_load(file)
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
