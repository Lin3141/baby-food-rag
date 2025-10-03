import networkx as nx
from typing import Dict, List, Set, Tuple
from .models import FoodItem
import pandas as pd

class BabyFoodKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.food_nodes = {}
        self.entity_types = {
            'FOOD': 'food_item',
            'CATEGORY': 'food_category', 
            'AGE_GROUP': 'age_restriction',
            'ALLERGEN': 'allergen_type',
            'NUTRIENT': 'nutrient_type',
            'SAFETY_RISK': 'safety_concern',
            'PREP_METHOD': 'preparation'
        }
    
    def build_graph_from_data(self, foods: List[FoodItem], df: pd.DataFrame):
        """Build knowledge graph from baby food data"""
        
        # Add food nodes
        for food in foods:
            self._add_food_node(food)
        
        # Extract and add entities from CSV
        for _, row in df.iterrows():
            food_name = str(row.get('food_name', ''))
            self._extract_and_link_entities(food_name, row)
        
        # Create relationship connections
        self._create_safety_relationships()
        self._create_nutritional_relationships()
        self._create_age_relationships()
        
    def _add_food_node(self, food: FoodItem):
        """Add food as a node with all properties"""
        node_id = f"FOOD:{food.name}"
        self.food_nodes[food.name] = node_id
        
        self.graph.add_node(node_id, 
                           type='FOOD',
                           name=food.name,
                           category=food.category,
                           nutrition={
                               'calories': food.kcal_100g,
                               'protein': food.protein_g,
                               'iron': food.iron_mg,
                               'vitamin_a': food.vit_a_ug,
                               'vitamin_c': food.vit_c_mg
                           },
                           note=food.note)
    
    def _extract_and_link_entities(self, food_name: str, row: pd.Series):
        """Extract entities and create relationships"""
        food_node = f"FOOD:{food_name}"
        
        # Extract age restrictions
        min_age = row.get('min_month_safe')
        if pd.notna(min_age):
            age_node = f"AGE_GROUP:{min_age}_months"
            self.graph.add_node(age_node, type='AGE_GROUP', min_months=min_age)
            self.graph.add_edge(food_node, age_node, relation='SAFE_FROM_AGE')
        
        # Extract allergens
        allergens = str(row.get('allergens', ''))
        if allergens and allergens.lower() not in ['nan', '']:
            allergen_node = f"ALLERGEN:{allergens}"
            self.graph.add_node(allergen_node, type='ALLERGEN', name=allergens)
            self.graph.add_edge(food_node, allergen_node, relation='CONTAINS_ALLERGEN')
        
        # Extract safety risks
        risks = str(row.get('risks', ''))
        if risks and risks.lower() not in ['nan', '']:
            # Parse multiple risks
            risk_list = [r.strip() for r in risks.split(';')]
            for risk in risk_list:
                if 'choking' in risk.lower():
                    risk_node = f"SAFETY_RISK:choking"
                    self.graph.add_node(risk_node, type='SAFETY_RISK', risk_type='choking')
                    self.graph.add_edge(food_node, risk_node, relation='HAS_RISK')
        
        # Extract nutrients
        nutrients = str(row.get('nutrient_highlights', ''))
        if nutrients and nutrients.lower() not in ['nan', '']:
            nutrient_list = [n.strip() for n in nutrients.split(',')]
            for nutrient in nutrient_list:
                nutrient_node = f"NUTRIENT:{nutrient}"
                self.graph.add_node(nutrient_node, type='NUTRIENT', name=nutrient)
                self.graph.add_edge(food_node, nutrient_node, relation='RICH_IN')
    
    def _create_safety_relationships(self):
        """Create safety-based relationships between foods"""
        # Connect foods with similar safety profiles
        choking_foods = [n for n, d in self.graph.nodes(data=True) 
                        if d.get('type') == 'FOOD' and 'choking' in d.get('note', '').lower()]
        
        for i, food1 in enumerate(choking_foods):
            for food2 in choking_foods[i+1:]:
                self.graph.add_edge(food1, food2, relation='SIMILAR_SAFETY_PROFILE')
    
    def _create_nutritional_relationships(self):
        """Create nutrition-based relationships"""
        # Group foods by high nutrients
        high_iron_foods = [n for n, d in self.graph.nodes(data=True)
                          if d.get('type') == 'FOOD' and 
                          d.get('nutrition', {}).get('iron', 0) > 2.0]
        
        # Create "ALTERNATIVE_FOR" relationships
        for i, food1 in enumerate(high_iron_foods):
            for food2 in high_iron_foods[i+1:]:
                self.graph.add_edge(food1, food2, relation='NUTRITIONAL_ALTERNATIVE')
    
    def _create_age_relationships(self):
        """Create age-progression relationships"""
        # Connect foods suitable for same age groups
        age_groups = {}
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'FOOD':
                # Extract age from note
                note = data.get('note', '').lower()
                if '6 months' in note:
                    age_groups.setdefault('6_months', []).append(node)
                elif '12 months' in note:
                    age_groups.setdefault('12_months', []).append(node)
        
        # Connect foods in same age group
        for age, foods in age_groups.items():
            for i, food1 in enumerate(foods):
                for food2 in foods[i+1:]:
                    self.graph.add_edge(food1, food2, relation='SAME_AGE_GROUP')

class GraphRAGRetriever:
    def __init__(self, kg: BabyFoodKnowledgeGraph, foods: List[FoodItem]):
        self.kg = kg
        self.foods = foods
        self.food_lookup = {food.name: food for food in foods}
    
    def graph_retrieve(self, query: str, top_k: int = 3) -> Tuple[List[FoodItem], List[float], List[str]]:
        """Retrieve using graph relationships and reasoning"""
        
        # 1. Find directly relevant foods (traditional retrieval)
        direct_foods = self._find_direct_matches(query)
        
        # 2. Use graph to find related foods
        related_foods = self._find_graph_related_foods(direct_foods, query)
        
        # 3. Generate reasoning paths
        reasoning_paths = self._generate_reasoning_paths(direct_foods + related_foods, query)
        
        # 4. Score and rank
        all_foods = list(set(direct_foods + related_foods))
        scores = self._score_foods_with_graph(all_foods, query)
        
        # 5. Return top-k with reasoning
        top_foods_with_scores = sorted(zip(all_foods, scores), key=lambda x: x[1], reverse=True)[:top_k]
        top_foods = [food for food, _ in top_foods_with_scores]
        top_scores = [score for _, score in top_foods_with_scores]
        
        return top_foods, top_scores, reasoning_paths[:top_k]
    
    def _find_direct_matches(self, query: str) -> List[FoodItem]:
        """Find foods directly matching query terms"""
        query_lower = query.lower()
        matches = []
        
        for food in self.foods:
            if any(term in food.note.lower() for term in query_lower.split()):
                matches.append(food)
        
        return matches[:5]  # Limit initial matches
    
    def _find_graph_related_foods(self, seed_foods: List[FoodItem], query: str) -> List[FoodItem]:
        """Use graph relationships to find related foods"""
        related = []
        query_lower = query.lower()
        
        for food in seed_foods:
            food_node = f"FOOD:{food.name}"
            if food_node in self.kg.graph:
                
                # Find foods connected by relationships
                for neighbor in self.kg.graph.neighbors(food_node):
                    edge_data = self.kg.graph.get_edge_data(food_node, neighbor)
                    relation = edge_data.get('relation', '')
                    
                    # Follow relevant relationship paths
                    if self._is_relevant_relation(relation, query_lower):
                        # Get foods connected to this neighbor
                        for second_neighbor in self.kg.graph.neighbors(neighbor):
                            if second_neighbor.startswith('FOOD:'):
                                food_name = second_neighbor.replace('FOOD:', '')
                                if food_name in self.food_lookup:
                                    related.append(self.food_lookup[food_name])
        
        return list(set(related))  # Remove duplicates
    
    def _is_relevant_relation(self, relation: str, query: str) -> bool:
        """Determine if a graph relation is relevant to the query"""
        relation_relevance = {
            'iron': ['RICH_IN', 'NUTRITIONAL_ALTERNATIVE'],
            'allergy': ['CONTAINS_ALLERGEN', 'SIMILAR_SAFETY_PROFILE'],
            'choking': ['HAS_RISK', 'SIMILAR_SAFETY_PROFILE'],
            'age': ['SAFE_FROM_AGE', 'SAME_AGE_GROUP'],
            'month': ['SAFE_FROM_AGE', 'SAME_AGE_GROUP']
        }
        
        for keyword, relations in relation_relevance.items():
            if keyword in query and relation in relations:
                return True
        return False
    
    def _generate_reasoning_paths(self, foods: List[FoodItem], query: str) -> List[str]:
        """Generate human-readable reasoning paths"""
        paths = []
        
        for food in foods[:3]:  # Top 3 foods
            food_node = f"FOOD:{food.name}"
            if food_node in self.kg.graph:
                # Find the reasoning path
                path_parts = [f"📍 {food.name}"]
                
                # Check direct attributes
                if 'iron' in query.lower() and food.iron_mg > 2:
                    path_parts.append(f"→ High iron content ({food.iron_mg}mg)")
                
                # Check graph relationships
                for neighbor in self.kg.graph.neighbors(food_node):
                    edge_data = self.kg.graph.get_edge_data(food_node, neighbor)
                    relation = edge_data.get('relation', '')
                    
                    if relation == 'SAFE_FROM_AGE':
                        neighbor_data = self.kg.graph.nodes[neighbor]
                        path_parts.append(f"→ Safe from {neighbor_data.get('min_months')} months")
                    elif relation == 'CONTAINS_ALLERGEN':
                        neighbor_data = self.kg.graph.nodes[neighbor]
                        path_parts.append(f"→ Contains {neighbor_data.get('name')} allergen")
                
                paths.append(" ".join(path_parts))
        
        return paths
    
    def _score_foods_with_graph(self, foods: List[FoodItem], query: str) -> List[float]:
        """Score foods considering graph relationships"""
        scores = []
        
        for food in foods:
            base_score = 0.5  # Base relevance
            
            # Direct text matching
            if any(term in food.note.lower() for term in query.lower().split()):
                base_score += 0.3
            
            # Graph relationship bonus
            food_node = f"FOOD:{food.name}"
            if food_node in self.kg.graph:
                # Count relevant relationships
                relevant_relations = 0
                for neighbor in self.kg.graph.neighbors(food_node):
                    edge_data = self.kg.graph.get_edge_data(food_node, neighbor)
                    if self._is_relevant_relation(edge_data.get('relation', ''), query.lower()):
                        relevant_relations += 1
                
                base_score += min(relevant_relations * 0.1, 0.4)  # Cap bonus at 0.4
            
            scores.append(base_score)
        
        return scores
