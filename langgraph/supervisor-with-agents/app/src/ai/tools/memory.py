from typing import Annotated

from langgraph.config import RunnableConfig
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore

def save_memory(
    memory: str, *, 
    config: RunnableConfig, 
    store: Annotated[BaseStore, InjectedStore()]
) -> str:
    '''Save the given memory for the current user.'''
    # This is a **tool** the model can use to save memories to storage
    user_id = config.get("configurable", {}).get("user_id")
    namespace = ("memories", user_id)
    store.put(namespace, f"memory_{len(store.search(namespace))}", {"data": memory})
    return f"Saved memory: {memory}"
