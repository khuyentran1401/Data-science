from typing import Optional
from dotenv import load_dotenv
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.ai.graph import corp_workflow
from src.ai.utils import collect_graph_states
from src.ai.config import prepare_query_inputs, prepare_config
from src.utils import process_query_result

# Load environment variables
load_dotenv()

# Log configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."}
    )

# Specific exception handler for validation errors (e.g., incorrect input format)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

# Healthcheck
@app.get("/healthcheck")
async def health_check():
    return {"status": "ok"}

# Query endpoint
@app.post("/query")
async def query(request: Request):
    try:
        data = await request.json()
        inputs = prepare_query_inputs(data)
        config = prepare_config(data)

        result = collect_graph_states(corp_workflow, inputs, config=config)

        return process_query_result(result)
    except (ValueError, KeyError) as exc:
        logger.error(f"Error during processing: {exc}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error during processing: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")
