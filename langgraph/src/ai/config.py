# ai/agents/core.py
from os import getenv

# Load environment variables
provider_name=getenv("PROVIDER_NAME", "openai")
model_name=getenv("MODEL_NAME", "gpt-4o-mini")

# Função para estruturar entradas padrão para o agente
def prepare_query_inputs(data: dict) -> dict:
    user_input = data.get("message", "")
    user_id = data.get("user_id", "default_user")
    thread_id = data.get("thread_id", "thread-1")

    inputs = {
        "messages": [{"role": "user", "content": user_input}],
        "user_id": user_id,
        "thread_id": thread_id,
    }
    return inputs

# Função para gerar configurações específicas para o agente
def prepare_config(data: dict) -> dict:
    return {
        "stream_mode": "values",
        "stream": True,
        "stream_interval": 0.1,
        "max_tokens": 100,
        "temperature": 0.7,
        "configurable": {
            "user_id": data.get("user_id", "default_user"),
            "thread_id": data.get("thread_id", "thread-1")
        }
    }
