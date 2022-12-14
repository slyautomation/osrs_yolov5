import yaml

# make sure to install PyYAML
def get_class(object='cow'):
    with open("./data/osrs.yaml", "r", encoding="utf8") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        result = data['names']
        outcome = result.index(object.capitalize())
    print(result)
    print(outcome)
if __name__ == "__main__":
    get_class()