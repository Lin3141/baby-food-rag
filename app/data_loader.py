import pandas as pd
from typing import List
from .models import FoodItem

class DataLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.foods: List[FoodItem] = []
        
    def load_data(self) -> List[FoodItem]:
        """Load food data from CSV file"""
        try:
            df = pd.read_csv(self.csv_path)
            self.foods = [FoodItem(**row.to_dict()) for _, row in df.iterrows()]
            return self.foods
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def get_food_descriptions(self) -> List[str]:
        """Get searchable text descriptions for each food"""
        descriptions = []
        for food in self.foods:
            description = f"{food.name} {food.category} {food.note}"
            descriptions.append(description)
        return descriptions
