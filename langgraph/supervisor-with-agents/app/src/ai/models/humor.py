# src/ai/schemas/jokes.py
from pydantic import BaseModel

class Joke(BaseModel):
    setup: str
    punchline: str
