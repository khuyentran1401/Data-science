import requests
from pywebio import start_server
from pywebio.output import put_markdown, put_table, put_loading, use_scope
from pywebio.pin import pin_wait_change, put_select
from extract_books import get_books_of_text
from typing import List


def get_activity_content(inputs: dict):
    """Get a random activity using Bored API"""
    if inputs['value'] == 'random':
        api = "https://www.boredapi.com/api/activity"
    else:
        api = f"https://www.boredapi.com/api/activity?type={inputs['value']}"
    response = requests.get(api)
    content = response.json()
    return content

def display_activity_for_boredom():
    put_markdown("# Find things to do when you're bored")
    activity_types = ['random', "education", "recreational", "social",
                      "diy", "charity", "cooking", "relaxation", "music", "busywork"]
    put_select(name='type', label='Activity Type', options=activity_types)


def create_book_table(books: List[dict]):
    if books == []:
        put_markdown("No books with this topic is found")
        return
    book_table = [[book['title'], book.get('author_name', ['_'])[0]] for book in books]
    book_table.insert(0, ['Title', 'Author'])
    put_table(book_table)


def app():
    display_activity_for_boredom()

    while True:
        new_inputs = pin_wait_change(['type'])
        with use_scope('activity', clear=True):
            content = get_activity_content(new_inputs)
            put_markdown(f"""## Your activity for today: {content['activity']}""")
            put_markdown(f"""Number of participants: {content['participants']}
            Price: {content['price']}
            Accessibility: {content['accessibility']}""")

            put_markdown("## Books related to this activity you might be interested in:")
            with put_loading():
                books = get_books_of_text(content['activity'])
                create_book_table(books)



if __name__ == '__main__':
    start_server(app, port=36635, debug=True)
