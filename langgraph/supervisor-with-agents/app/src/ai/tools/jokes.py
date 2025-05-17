# src/ai/tools/joker.py
import requests

from src.ai.models.humor import Joke

def get_random_joke(category: str = "Any") -> Joke:
    """
    Fetches a random joke from the JokeAPI.

    Args:
        category: A string with the joke category. Valid values include: 'Programming', 'Pun', 'Misc', 'Dark', 'Spooky', 'Christmas'.

    Example:
        get_random_joke(category='Programming')
    """
    try:
        url = f"https://v2.jokeapi.dev/joke/{category}?safe-mode"
        response = requests.get(url)
        response.raise_for_status()
        joke_data = response.json()

        if joke_data.get("error"):
            return Joke(setup="Error", punchline=joke_data.get("message", "Unknown error"))

        if joke_data["type"] == "single":
            return Joke(setup=joke_data["joke"], punchline="")
        else:
            return Joke(setup=joke_data["setup"], punchline=joke_data["delivery"])

    except Exception as e:
        return Joke(setup="Failed to fetch joke", punchline=str(e))
