from pydantic import BaseModel

class QueryRequest(BaseModel):
    message: str
    user_id: str | None = None
    thread_id: str | None = None

class QueryResponse(BaseModel):
    output: str