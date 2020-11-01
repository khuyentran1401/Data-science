import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

suicide = pd.read_csv('master.csv')

suicide[' gdp_for_year ($) ']= suicide[' gdp_for_year ($) '].apply(lambda val: val.replace(',', ''))
suicide[' gdp_for_year ($) '] = pd.to_numeric(suicide[' gdp_for_year ($) '])

train, test = train_test_split(suicide, test_size=0.2, random_state = 1)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = 1)
for train_index, test_index in split.split(suicide, suicide['generation']):
    strat_train = suicide.loc[train_index]
    strat_test = suicide.loc[test_index]

