import sweetviz as sv
import pandas as pd
from pywebio.input import file_upload
from pywebio.output import put_html, put_loading
from pywebio import start_server
import csv 
import re

def app():
    file = file_upload(label='Upload your CSV file', accept='.csv')
    content = file['content'].decode('utf-8').splitlines()

    df = content_to_pandas(content)
    create_profile(df)
    
def create_profile(df: pd.DataFrame):
    with put_loading(shape='grow'):
        report = sv.analyze(df)
        report.show_html()
    with open('SWEETVIZ_REPORT.html', 'r') as f:
        html = f.read()
        put_html(html)

def content_to_pandas(content: list):
    # Open the file for writing
    with open("tmp.csv", "w") as csv_file:
        # Create the writer object with tab delimiter
        writer = csv.writer(csv_file, delimiter = '\t')
        for line in content:
            # Writerow() needs a list of data to be written, so split at all empty spaces in the line 
            writer.writerow(re.split('\s+',line))
    return pd.read_csv("tmp.csv")
    

if __name__=='__main__':
    start_server(app, port=37791, debug=True)

from pywebio.input import *
from pywebio.output import *

input("What's your name?")
select("Select food", ['Orange', 'Apple'])
checkbox("Are your okay?", options=["I'm okay."])
radio("What do you like to do?", options=['Eat', 'Sleep', 'Study'])

