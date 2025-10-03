import pandas as pd
import numpy as np
from typing import List, Dict
from .models import FoodItem

class DataLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.foods: List[FoodItem] = []
        
    def load_data(self) -> List[FoodItem]:
        """Load food data from CSV file"""
        try:
            df = pd.read_csv(self.csv_path)
            
            # Handle the new knowledge graph format
            if 'food_name' in df.columns:
                # Convert knowledge graph to FoodItem format
                self.foods = self._convert_kg_to_food_items(df)
            else:
                # Original format
                self.foods = [FoodItem(**row.to_dict()) for _, row in df.iterrows()]
            return self.foods
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _safe_get_string(self, row: pd.Series, column: str, default: str = '') -> str:
        """Safely get string value from row, handling NaN values"""
        value = row.get(column, default)
        if pd.isna(value) or value is None:
            return default
        return str(value)
    
    def _convert_kg_to_food_items(self, df: pd.DataFrame) -> List[FoodItem]:
        """Convert knowledge graph format to FoodItem objects with parent-focused enhancements"""
        foods = []
        for _, row in df.iterrows():
            # Create parent-friendly, anxiety-reducing description
            safety_prep = self._extract_safety_preparation(row)
            reassuring_note = self._create_reassuring_note(row)
            
            food = FoodItem(
                name=self._safe_get_string(row, 'food_name'),
                category=self._safe_get_string(row, 'group', 'Unknown'),
                kcal_100g=self._estimate_nutrition(row, 'calories'),
                protein_g=self._estimate_nutrition(row, 'protein'),
                fiber_g=self._estimate_nutrition(row, 'fiber'),
                iron_mg=self._estimate_nutrition(row, 'iron'),
                vit_a_ug=self._estimate_nutrition(row, 'vitamin_a'),
                vit_c_mg=self._estimate_nutrition(row, 'vitamin_c'),
                usda_url=f"https://fdc.nal.usda.gov/search?query={self._safe_get_string(row, 'food_name').replace(' ', '%20')}",
                note=reassuring_note
            )
            foods.append(food)
        return foods
    
    def _extract_safety_preparation(self, row: pd.Series) -> str:
        """Extract step-by-step safety preparation instructions"""
        prep = self._safe_get_string(row, 'prep')
        risks = self._safe_get_string(row, 'risks')
        min_age = row.get('min_month_safe', 6)
        
        safety_steps = []
        if min_age and not pd.isna(min_age):
            safety_steps.append(f"✅ Safe from {min_age} months")
        if prep and prep.strip():
            safety_steps.append(f"🍽️ How to prepare: {prep}")
        if risks and risks.strip() and risks.lower() not in ['nan', '']:
            safety_steps.append(f"⚠️ Watch out for: {risks}")
        
        return " | ".join(safety_steps)
    
    def _create_reassuring_note(self, row: pd.Series) -> str:
        """Create reassuring, parent-friendly descriptions"""
        base_note = self._safe_get_string(row, 'notes')
        allergens = self._safe_get_string(row, 'allergens')
        nutrients = self._safe_get_string(row, 'nutrient_highlights')
        
        # Start with reassurance
        reassuring_parts = []
        
        # Add confidence-building intro
        source_primary = self._safe_get_string(row, 'source_primary')
        if source_primary in ['AAP/CDC infant solids', 'WHO infant feeding']:
            reassuring_parts.append("✅ Pediatrician-recommended")
        
        # Add nutritional benefits
        if nutrients and nutrients.strip():
            reassuring_parts.append(f"💪 Rich in: {nutrients}")
        
        # Add safety info
        safety_prep = self._extract_safety_preparation(row)
        if safety_prep:
            reassuring_parts.append(safety_prep)
        
        # Add original note
        if base_note and base_note.strip():
            reassuring_parts.append(f"📝 {base_note}")
        
        # Add allergy info in supportive way
        if allergens and allergens.strip() and allergens.lower() not in ['nan', '']:
            reassuring_parts.append(f"🔍 Common allergen ({allergens}) - introduce when ready, watch for reactions")
        
        return " | ".join(reassuring_parts)
    
    def _estimate_nutrition(self, row: pd.Series, nutrient_type: str) -> float:
        """Estimate nutritional values based on food type and highlights"""
        food_name = self._safe_get_string(row, 'food_name').lower()
        nutrients = self._safe_get_string(row, 'nutrient_highlights').lower()
        
        # Enhanced nutritional estimates based on real USDA data
        nutrition_map = {
            'calories': {
                'apple': 52, 'banana': 89, 'pear': 57, 'peach': 39, 'plum': 46,
                'avocado': 160, 'blueberry': 57, 'strawberry': 32, 'mango': 60, 'watermelon': 30,
                'carrot': 41, 'sweet potato': 86, 'pumpkin': 26, 'peas': 81, 'green beans': 35,
                'broccoli': 34, 'cauliflower': 25, 'spinach': 23, 'zucchini': 17,
                'chicken': 165, 'turkey': 144, 'beef': 250, 'pork': 242, 'salmon': 208,
                'cod': 82, 'tuna': 144, 'shrimp': 85, 'egg': 155, 'tofu': 76,
                'lentils': 116, 'black beans': 132, 'yogurt': 59, 'cheese': 113, 'cottage cheese': 98,
                'oatmeal': 68, 'rice cereal': 380, 'quinoa': 120, 'pasta': 131, 'bread': 265,
                'honey': 304, 'peanut butter': 588,
                'default': 80
            },
            'protein': {
                'chicken': 31.0, 'turkey': 30.3, 'beef': 26.1, 'pork': 27.3, 'salmon': 25.4,
                'cod': 18.0, 'tuna': 25.5, 'shrimp': 18.0, 'egg': 13.0, 'tofu': 8.1,
                'lentils': 9.0, 'black beans': 8.9, 'yogurt': 10.0, 'cheese': 12.5, 'cottage cheese': 11.1,
                'peanut butter': 25.8, 'quinoa': 4.4, 'oatmeal': 2.4,
                'peas': 5.4, 'spinach': 2.9, 'broccoli': 2.8,
                'default': 2.0
            },
            'iron': {
                'rice cereal': 45.0, 'beef': 2.6, 'lentils': 3.3, 'spinach': 2.7, 'tofu': 5.4,
                'chicken': 0.9, 'turkey': 1.4, 'salmon': 0.8, 'egg': 1.8,
                'quinoa': 1.5, 'oatmeal': 1.2, 'pumpkin': 0.8,
                'default': 0.5
            },
            'fiber': {
                'avocado': 6.7, 'pear': 3.1, 'apple': 2.4, 'sweet potato': 3.0,
                'lentils': 7.9, 'black beans': 8.7, 'peas': 5.1, 'broccoli': 2.6,
                'quinoa': 2.8, 'oatmeal': 1.7,
                'default': 1.5
            },
            'vitamin_a': {
                'sweet potato': 709, 'carrot': 835, 'spinach': 469, 'mango': 54,
                'pumpkin': 426, 'cantaloupe': 169,
                'default': 20
            },
            'vitamin_c': {
                'strawberry': 58.8, 'broccoli': 89.2, 'mango': 36.4, 'kiwi': 92.7,
                'bell pepper': 120.0, 'cauliflower': 48.2, 'peas': 40.0,
                'apple': 4.6, 'banana': 8.7, 'pear': 4.3,
                'default': 5.0
            }
        }
        
        # Check for specific foods first (exact matches)
        for food_key, value in nutrition_map[nutrient_type].items():
            if food_key == food_name or food_key in food_name:
                if food_key != 'default':
                    return float(value)
        
        # Check if nutrient is specifically highlighted
        nutrient_variations = [
            nutrient_type,
            nutrient_type.replace('_', ' '),
            nutrient_type.replace('_', '-'),
        ]
        
        # Special cases for vitamin naming
        if nutrient_type == 'vitamin_a':
            nutrient_variations.extend(['vitamin a', 'beta-carotene', 'beta carotene'])
        elif nutrient_type == 'vitamin_c':
            nutrient_variations.extend(['vitamin c', 'ascorbic acid'])
        elif nutrient_type == 'iron':
            nutrient_variations.extend(['heme iron', 'non-heme iron'])
        
        # If nutrient is highlighted, use higher than default
        for variation in nutrient_variations:
            if variation in nutrients:
                return float(nutrition_map[nutrient_type]['default'] * 3)
        
        return float(nutrition_map[nutrient_type]['default'])

    def get_food_descriptions(self) -> List[str]:
        """Get searchable text descriptions for each food"""
        descriptions = []
        for food in self.foods:
            description = f"{food.name} {food.category} {food.note}"
            descriptions.append(description)
        return descriptions
    
    def get_safety_focused_descriptions(self) -> List[str]:
        """Get safety-focused descriptions that ChatGPT wouldn't emphasize"""
        descriptions = []
        for food in self.foods:
            # Extract safety-critical information
            safety_info = []
            note_lower = food.note.lower()
            
            if 'choking' in note_lower:
                safety_info.append("CHOKING_HAZARD")
            if 'allergy' in note_lower:
                safety_info.append("ALLERGY_RISK")
            if 'month' in note_lower:
                safety_info.append("AGE_RESTRICTION")
            if 'botulism' in note_lower:
                safety_info.append("BOTULISM_RISK")
            
            description = f"{food.name} {food.category} {' '.join(safety_info)} {food.note}"
            descriptions.append(description)
        return descriptions
    
    def get_parent_anxiety_descriptions(self) -> List[str]:
        """Get descriptions that address common parent anxieties"""
        descriptions = []
        for food in self.foods:
            # Create anxiety-addressing keywords
            anxiety_keywords = []
            note_lower = food.note.lower()
            
            # Address safety anxiety
            if 'pediatrician-recommended' in note_lower:
                anxiety_keywords.append("DOCTOR_APPROVED")
            if 'safe from' in note_lower:
                anxiety_keywords.append("AGE_VERIFIED")
            if 'watch out' in note_lower:
                anxiety_keywords.append("SAFETY_GUIDANCE")
            
            # Address nutritional anxiety
            if 'rich in' in note_lower:
                anxiety_keywords.append("NUTRITIOUS")
            if any(nutrient in note_lower for nutrient in ['iron', 'protein', 'vitamin']):
                anxiety_keywords.append("ESSENTIAL_NUTRIENTS")
            
            # Address preparation anxiety
            if 'how to prepare' in note_lower:
                anxiety_keywords.append("PREP_INSTRUCTIONS")
            
            description = f"{food.name} {food.category} {' '.join(anxiety_keywords)} {food.note}"
            descriptions.append(description)
        return descriptions
    
    def get_quick_answer_data(self) -> Dict[str, List[str]]:
        """Pre-computed answers for common urgent parent questions"""
        quick_answers = {
            "first_foods_6_months": [],
            "high_iron_foods": [],
            "choking_hazards_avoid": [],
            "allergy_foods_introduce_carefully": [],
            "emergency_prep_instructions": []
        }
        
        for food in self.foods:
            note_lower = food.note.lower()
            
            # First foods safe at 6 months
            if '6 months' in note_lower and 'safe' in note_lower:
                quick_answers["first_foods_6_months"].append(food.name)
            
            # High iron foods for anemia concerns
            if food.iron_mg > 5 or 'iron' in note_lower:
                quick_answers["high_iron_foods"].append(f"{food.name} ({food.iron_mg}mg iron)")
            
            # Choking hazards to avoid
            if 'choking' in note_lower:
                quick_answers["choking_hazards_avoid"].append(food.name)
            
            # Foods requiring allergy vigilance
            if 'allergen' in note_lower or 'allergy' in note_lower:
                quick_answers["allergy_foods_introduce_carefully"].append(food.name)
            
            # Foods with specific prep instructions
            if 'how to prepare' in note_lower:
                prep_instruction = note_lower.split('how to prepare:')[1].split('|')[0].strip()
                quick_answers["emergency_prep_instructions"].append(f"{food.name}: {prep_instruction}")
        
        return quick_answers
