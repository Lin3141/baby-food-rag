from pydantic import BaseModel
from typing import List, Optional

class FoodItem(BaseModel):
    name: str
    category: str
    kcal_100g: float
    protein_g: float
    fiber_g: float
    iron_mg: float
    vit_a_ug: float
    vit_c_mg: float
    usda_url: str
    note: str

class Citation(BaseModel):
    food_name: str
    usda_url: str
    relevance_score: float

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: str  # "High", "Medium", "Low"
    retrieved_foods: List[FoodItem]
