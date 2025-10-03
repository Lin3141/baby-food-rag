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

class SafetyAlert(BaseModel):
    level: str  # "CRITICAL", "WARNING", "INFO"
    message: str
    source: str

class RAGAdvantage(BaseModel):
    evidence_based: bool
    source_cited: bool
    safety_checked: bool
    age_appropriate: bool
    medical_guidelines: str

class EnhancedAskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: str
    retrieved_foods: List[FoodItem]
    safety_alerts: List[SafetyAlert]
    rag_advantages: RAGAdvantage
    vs_chatgpt: str  # Explanation of why this is better than ChatGPT
