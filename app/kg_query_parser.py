import re
from typing import Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class ParsedQuery:
    food: Optional[str]
    age_months: Optional[int]
    query_type: str  # 'safety', 'nutrition', 'preparation', 'general'
    raw_question: str

class BabyFoodQueryParser:
    def __init__(self, food_names: List[str]):
        self.food_names = [name.lower() for name in food_names]
        
    def parse_query(self, question: str) -> ParsedQuery:
        """Parse user question to extract food, age, and intent"""
        question_lower = question.lower()
        
        # Extract age
        age = self._extract_age(question_lower)
        
        # Extract food
        food = self._extract_food(question_lower)
        
        # Determine query type
        query_type = self._determine_query_type(question_lower)
        
        return ParsedQuery(
            food=food,
            age_months=age,
            query_type=query_type,
            raw_question=question
        )
    
    def _extract_age(self, question: str) -> Optional[int]:
        """Extract age in months from question"""
        # Patterns for age extraction
        patterns = [
            r'(\d+)\s*month[s]?\s*old',
            r'(\d+)\s*mo\s*old',
            r'my\s*(\d+)\s*month',
            r'(\d+)\s*m\s*old',
            r'(\d+)\s*month[s]?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_food(self, question: str) -> Optional[str]:
        """Extract food name from question"""
        # Direct food name matching
        for food_name in self.food_names:
            if food_name in question:
                # Return the original case version
                for original_name in self.food_names:
                    if original_name.lower() == food_name:
                        return original_name.title()
        
        # Handle plurals and variations
        food_variations = {
            'apples': 'apple',
            'bananas': 'banana',
            'eggs': 'egg',
            'carrots': 'carrot',
            'peas': 'peas',  # already plural
        }
        
        for variation, canonical in food_variations.items():
            if variation in question and canonical in self.food_names:
                return canonical.title()
        
        return None
    
    def _determine_query_type(self, question: str) -> str:
        """Determine the type of question being asked"""
        safety_keywords = ['safe', 'can', 'okay', 'give', 'introduce', 'start']
        nutrition_keywords = ['protein', 'iron', 'vitamin', 'nutrition', 'healthy']
        prep_keywords = ['prepare', 'cook', 'make', 'serve', 'texture']
        
        if any(keyword in question for keyword in safety_keywords):
            return 'safety'
        elif any(keyword in question for keyword in nutrition_keywords):
            return 'nutrition'
        elif any(keyword in question for keyword in prep_keywords):
            return 'preparation'
        else:
            return 'general'
