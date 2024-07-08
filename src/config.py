import yaml

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, path):
        keys = path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key, None)
            if value is None:
                break
        return value

