import requests 
from bs4 import BeautifulSoup

url = "https://khuyentran1401.github.io/Python-data-science-code-snippet/"
html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")