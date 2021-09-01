import requests
from pywebio import start_server
from pywebio.output import put_markdown, use_scope
from pywebio.input import select, radio, input_group, input_update
from pywebio.pin import pin_wait_change, pin, put_select 


def get_bore_inputs():
    activity_types = ['random', "education", "recreational", "social", "diy", "charity", "cooking", "relaxation", "music", "busywork"]
    put_select(label='Activity Type', options=activity_types, name='type')


def get_activity_content(inputs: dict):
    if inputs['value'] == 'random':
        api = "https://www.boredapi.com/api/activity"
    else:
        api = f"https://www.boredapi.com/api/activity?type={inputs['value']}"
    response = requests.get(api)
    content = response.json()
    return content


def app():
    bored_inputs = get_bore_inputs()
    while True:
        new_inputs = pin_wait_change(['type'])
        with use_scope('activity', clear=True):
            content = get_activity_content(new_inputs)
            put_markdown(f"""Your activity for today: {content['activity']}
        Number of participants: {content['participants']}
        Price: {content['price']}
        Accessibility: {content['accessibility']}
        """)


if __name__ == '__main__':
    start_server(app, port=36635, debug=True)
