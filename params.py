import json


class Params:
    def __init__(self, path):
        with open(path) as params_file:
            params = json.load(params_file)
            self.__dict__.update(params)
