import json

def loads_json(filepath: str):
    """
    Load json file
    :param filepath: str
    :return:
    """
    with open(filepath, "r") as f:
        data = f.read()
        data = json.loads(data)
    return data

def load_json(filepath: str):
    """
    Load json file
    :param filepath: str
    :return:
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data