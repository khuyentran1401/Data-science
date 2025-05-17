from pydantic import BaseModel, EmailStr, Field
from typing import List, Literal, Optional
from datetime import date


class Purchase(BaseModel):
    date: date
    item: str
    amount: float


class Preferences(BaseModel):
    contact_method: Literal["email", "sms", "phone"]
    interests: List[str]
    marketing_opt_in: bool = False


class LoyaltyInfo(BaseModel):
    program_member: bool = True
    points: int = 0
    tier: Literal["Bronze", "Silver", "Gold", "Platinum"] = "Bronze"


class CustomerProfile(BaseModel):
    id: str
    name: str
    score: int = Field(..., ge=0, le=100)
    last_purchase: date
    email: EmailStr
    phone: str
    address: str
    purchase_history: List[Purchase]
    preferences: Optional[Preferences] = None
    loyalty: Optional[LoyaltyInfo] = None
