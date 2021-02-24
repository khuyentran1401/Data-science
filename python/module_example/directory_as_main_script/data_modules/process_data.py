class DataProcessor: 
    def __init__(self, data_name: str):
        self.data_name = data_name
    
    def process_data(self):
        print(f"Processing {self.data_name}")