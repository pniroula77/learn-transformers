class DataPreprocessor:
    def __init__(self, filename):
        self.filename = filename

    def read_file(self):
        with open(self.filename, "r") as f:
            data = f.readlines()
        return data
