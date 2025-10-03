import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .models import FoodItem
from .kg_query_parser import ParsedQuery

@dataclass
class KGFact:
    subject: str
    relation: str
    object: str
    source: str
    confidence: float = 1.0

@dataclass
class KGSubgraph:
    facts: List[KGFact]
    graph_path: List[str]
    safety_flags: List[str]

class KnowledgeGraphRetriever:
    def __init__(self, foods: List[FoodItem]):
        self.foods = foods
        self.food_lookup = {food.name.lower(): food for food in foods}
        self.kg = self._build_knowledge_graph()
        
    def _build_knowledge_graph(self) -> nx.DiGraph:
        """Build knowledge graph from food data with enhanced safety rules"""
        G = nx.DiGraph()
        
        for food in self.foods:
            food_node = f"FOOD:{food.name}"
            G.add_node(food_node, type="food", data=food)
            
            # Extract age safety from note with more specific rules
            note_lower = food.note.lower()
            food_lower = food.name.lower()
            
            # Hard-coded critical safety rules
            if 'honey' in food_lower:
                G.add_edge(food_node, "AGE:12", relation="SAFE_AT", source="AAP Critical Guidelines")
                G.add_edge(food_node, "RISK:botulism", relation="HAS_RISK", source="AAP Critical Guidelines")
            elif 'whole grapes' in food_lower or 'grape' in food_lower:
                G.add_edge(food_node, "AGE:48", relation="SAFE_AT", source="AAP Choking Guidelines")
                G.add_edge(food_node, "RISK:choking", relation="HAS_RISK", source="AAP Choking Guidelines")
            elif 'whole nuts' in food_lower or (any(nut in food_lower for nut in ['walnut', 'almond', 'peanut']) and 'whole' in note_lower):
                G.add_edge(food_node, "AGE:48", relation="SAFE_AT", source="AAP Choking Guidelines")
                G.add_edge(food_node, "RISK:choking", relation="HAS_RISK", source="AAP Choking Guidelines")
            elif "cow's milk" in food_lower and 'drink' in note_lower:
                G.add_edge(food_node, "AGE:12", relation="SAFE_AT", source="AAP Nutrition Guidelines")
                G.add_edge(food_node, "RISK:anemia", relation="HAS_RISK", source="AAP Nutrition Guidelines")
            
            # Extract from note content
            elif 'safe from 6 months' in note_lower:
                G.add_edge(food_node, "AGE:6", relation="SAFE_AT", source="AAP/CDC")
            elif 'safe from 12 months' in note_lower:
                G.add_edge(food_node, "AGE:12", relation="SAFE_AT", source="AAP/CDC")
            
            # Extract risks with enhanced detection
            if 'choking' in note_lower:
                G.add_edge(food_node, "RISK:choking", relation="HAS_RISK", source="Safety Database")
            if 'allergy' in note_lower or 'allergen' in note_lower:
                G.add_edge(food_node, "RISK:allergy", relation="HAS_RISK", source="Allergy Guidelines")
            if 'nitrate' in note_lower:
                G.add_edge(food_node, "RISK:nitrate", relation="HAS_RISK", source="CDC Guidelines")
            if 'botulism' in note_lower:
                G.add_edge(food_node, "RISK:botulism", relation="HAS_RISK", source="AAP Guidelines")
            
            # Extract nutrients
            if food.iron_mg > 2:
                G.add_edge(food_node, "NUTRIENT:iron", relation="CONTAINS", source="USDA Database")
            if food.vit_a_ug > 100:
                G.add_edge(food_node, "NUTRIENT:vitamin_a", relation="CONTAINS", source="USDA Database")
            if food.vit_c_mg > 20:
                G.add_edge(food_node, "NUTRIENT:vitamin_c", relation="CONTAINS", source="USDA Database")
            if food.protein_g > 10:
                G.add_edge(food_node, "NUTRIENT:protein", relation="CONTAINS", source="USDA Database")
        
        return G
    
    def retrieve_subgraph(self, parsed_query: ParsedQuery) -> KGSubgraph:
        """Retrieve relevant subgraph based on parsed query"""
        if not parsed_query.food:
            return KGSubgraph(facts=[], graph_path=[], safety_flags=[])
        
        food_node = f"FOOD:{parsed_query.food}"
        if food_node not in self.kg:
            return KGSubgraph(facts=[], graph_path=[], safety_flags=[])
        
        facts = []
        graph_path = [parsed_query.food]
        safety_flags = []
        
        # Get all edges from the food node
        for neighbor in self.kg.neighbors(food_node):
            edge_data = self.kg.get_edge_data(food_node, neighbor)
            relation = edge_data['relation']
            source = edge_data['source']
            
            fact = KGFact(
                subject=parsed_query.food,
                relation=relation,
                object=neighbor.split(':')[1] if ':' in neighbor else neighbor,
                source=source
            )
            facts.append(fact)
            graph_path.append(f"{relation} → {fact.object}")
            
            # Check for safety flags
            if relation == "HAS_RISK":
                safety_flags.append(fact.object)
            elif relation == "SAFE_AT" and parsed_query.age_months:
                min_age = int(fact.object)
                if parsed_query.age_months < min_age:
                    safety_flags.append(f"too_young_for_{parsed_query.food}")
        
        return KGSubgraph(facts=facts, graph_path=graph_path, safety_flags=safety_flags)
    
    def generate_llm_prompt(self, parsed_query: ParsedQuery, subgraph: KGSubgraph) -> str:
        """Generate LLM prompt with retrieved facts"""
        facts_text = "\n".join([
            f"- {fact.subject} {fact.relation} {fact.object} (Source: {fact.source})"
            for fact in subgraph.facts
        ])
        
        system_prompt = f"""You are a pediatric nutrition expert. Answer the user's question using ONLY the provided facts.

FACTS ABOUT {parsed_query.food.upper() if parsed_query.food else 'THE FOOD'}:
{facts_text}

SAFETY FLAGS: {', '.join(subgraph.safety_flags) if subgraph.safety_flags else 'None'}

RULES:
1. Use ONLY the facts provided above
2. If uncertain or missing information, clearly state "I don't have enough information"
3. Always include source citations
4. Format response as:
   ✅/⚠️/❌ [Safety verdict]
   🥄 Prep: [Preparation if available]
   ⚠️ Note: [Warnings if any]
   📚 Sources: [List sources]

USER QUESTION: {parsed_query.raw_question}

RESPONSE:"""
        
        return system_prompt
