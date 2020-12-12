import spacy
import streamlit as st 
from itertools import combinations 
import pandas as pd 


st.title('Similarity app')

col1, col2, col3 = st.beta_columns(3)
with col1:
    word_1 = st.text_input('word 1', 'shirt')
with col2:
    word_2 = st.text_input('word 2', 'jeans')
with col3:
    word_3 = st.text_input('word 3', 'apple')

nlp = spacy.load("en_core_web_md")
tokens = nlp(f"{word_1} {word_2} {word_3}")

# get combination of tokens
comb = combinations(tokens, 2)

most_similar = 0
match_tokens = None
compared_tokens = []
similarities = []
for token in list(comb):
    similarity = token[0].similarity(token[1])
    compared_tokens.append(token)
    similarities.append(similarity)
    if similarity > most_similar:

        most_similar = similarity
        match_tokens = token

st.write(f'{match_tokens[0]} and {match_tokens[1]} are the most similar with a similarity of {round(most_similar*100, 2)}%')
st.write('## Results')

df = pd.DataFrame({
  'Tokens': compared_tokens,
  'Similarity': similarities
}).sort_values(by='Similarity', ascending=False)

df