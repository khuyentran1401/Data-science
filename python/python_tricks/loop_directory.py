import os
import sys
import pandas as pd
import glob
def loop_directory(directory: str):
	'''Loop files in the directory'''

	for filename in os.listdir(directory):
		if filename.endswith(".csv"):
			file_directory = os.path.join(directory, filename)
			print(file_directory)
			pd.read_csv(file_directory)
		else:
			continue

def loop_directory_glob(directory: str):
	for file in glob.glob(os.path.join(directory, '*.csv')):
		print(file)

if __name__=='__main__':
	# loop_directory('data/')
	loop_directory_glob('data/')
