from load_data import DataLoader
from process_data import DataProcessor

data_loader = DataLoader('data/')
data_loader.load_data()

data_processor = DataProcessor('data1')
data_processor.process_data()