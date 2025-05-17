from langchain_core.messages import BaseMessage

def filter_response(raw_json):
    # Convert LangChain BaseMessage to dict if needed
    if isinstance(raw_json, BaseMessage):
        raw_json = raw_json.model_dump()

    # Filter the required information
    metadata=raw_json.get("response_metadata", {})
    token_usage=metadata.get("token_usage", {})

    return {
        "type": raw_json.get("type"),
        "name": raw_json.get("name"),
        "id": raw_json.get("id"),
        "content": raw_json.get("content"),
        "token_usage": {
            "model": metadata.get("model_name"),
            "input": token_usage.get("prompt_tokens"),
            "output": token_usage.get("completion_tokens"),
            "total": token_usage.get("total_tokens"),
        }
    }

def aggregate_tokens(filtered_responses):
    def get_input(response: dict, key: str):
        return response.get("token_usage", {}).get(key, 0) or 0 
    
    total_input = sum(
        get_input(response, "input") 
        for response in filtered_responses
    )
    total_output = sum(
        get_input(response, "output") 
        for response in filtered_responses
    )
    total_tokens = total_input + total_output
    return total_input, total_output, total_tokens

def process_query_result(result):
    # Coleta os estados do gráfico com os inputs e configurações
    query_response = result[1].get("messages", [{}])

    # Apply filtering to the response
    filtered_responses = list(map(filter_response, query_response))

    # Aggregate token usage
    total_input, total_output, total_tokens = aggregate_tokens(filtered_responses)

    # Return filtered responses and token usage
    return {
        "content": filtered_responses[-1].get("content"),
        "token_usage": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens
        },
        "steps": filtered_responses
    }