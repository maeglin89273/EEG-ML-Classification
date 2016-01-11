import json

__author__ = 'maeglin89273'

class JsonFileOracle:
    def __init__(self):
        pass

    def save(self, structure, path):
        with open(path, "w") as destination:
            json.dump(structure, destination, indent=4, separators=(",", ": "))

    def load(self, path):
        with open(path, "rt") as source:
            return json.load(source)
