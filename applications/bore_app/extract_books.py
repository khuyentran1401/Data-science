import requests  
import spacy 
from rich import print

def extract_noun_phrases(text: str):
    """Extract noun phrases from a text"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]
        
def get_query(phrase: str):
    """Turn noun phrase into a query by replacing space with +"""
    return '+'.join(phrase.split(' '))
    
def get_query_for_noun_phrases(text: str):
    """Turn list of noun phrases into a list of queries"""
    noun_phrases = extract_noun_phrases(text)
    return [get_query(phrase) for phrase in noun_phrases]
      
def get_books(query: str):
    """Get the first 3 books based on the query"""
    api = f"https://openlibrary.org/search.json?title={query}"
    response = requests.get(api)
    content = response.json()['docs'][:3]
    return content 

def get_books_of_text(text: str):
    """Get books given a test"""
    queries = get_query_for_noun_phrases(text)
    books = []
    for query in queries:
        books.extend(get_books(query))
    return books
    
if __name__ == '__main__':
    text = 'Write a thank you letter to an influential person in your life'
    
    books = get_books_of_text(text)
    print(books[0])