import yaml

with open("config.yaml", "r", encoding="utf-8") as arquivo:
    config = yaml.safe_load(arquivo)

print(config)