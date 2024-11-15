import yaml

class YamlParser:
    def __init__(self, config_path):
        self.config_path = config_path

    def parse(self):
        with open(self.config_path) as file:
            config = yaml.safe_load(file)
        return config
