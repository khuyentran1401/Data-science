class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_data(self):
        print(f"Loading data from {self.data_dir}")
