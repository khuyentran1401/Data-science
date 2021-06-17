from pywebio.input import * 
from pywebio.output import * 
from pywebio import start_server

def app():
    topics_to_articles = {
        'Data': ['How to Create Fake Data with Faker', 'Introduction to Schema: A Python Libary to Validate your Data'],
        'Machine Learning': ['human-learn: Create a Human Learning Model by Drawing', 'Introduction to Weight & Biases: Track and Visualize your Machine Learning Experiments in 3 Lines of Code'],
        'Visualization': ['How to Sketch your Data Science Ideas With Excalidraw', 'How to Create Mathematical Animations like 3Blue1Brown Using Python']
    }

    topics = list(topics_to_articles.keys())
    resource = input_group('Select a topic', [
        select('Topic', options=topics, name='topic',
                onchange=lambda t: input_update('article', options=topics_to_articles[t])
                ),
        select('Article', options=topics_to_articles[topics[0]], name='article')
    ])
    article = resource['article']
    put_markdown(f"You selected *{article}*")


if __name__ == '__main__':
    start_server(app, debug=True, port='44315')
