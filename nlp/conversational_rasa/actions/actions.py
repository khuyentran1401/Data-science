# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionShowMenu(Action):

    def name(self) -> Text:
        return "action_show_menu"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        
        menu = {'Chicken and rice': 10,
                'Pho': 11, 
                'Spicy chicken': 9}

        dispatcher.utter_message(f"The menu today is:\n")
        for food in menu:
            dispatcher.utter_message(food)

        return []
