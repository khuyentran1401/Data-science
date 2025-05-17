from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime


class KPI(BaseModel):
    metric: str
    value: float
    unit: str
    target: float
    trend: Literal["up", "down", "stable"]
    last_updated: datetime
    critical_threshold: float = Field(..., description="Threshold that triggers alerts")
    status: Literal["on_track", "at_risk", "off_track"] = "on_track"
    owner: str = Field(..., description="Person or team responsible for this KPI")
