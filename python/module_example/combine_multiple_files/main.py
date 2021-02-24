from data_modules import DataLoader, DataProcessor

data_loader = DataLoader('data/')
data_loader.load_data()

data_processor = DataProcessor('data1')
data_processor.process_data()