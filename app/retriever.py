import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Tuple
from .models import FoodItem, Citation

class HybridRetriever:
    def __init__(self, foods: List[FoodItem], descriptions: List[str]):
        self.foods = foods
        self.descriptions = descriptions
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build BM25 index
        tokenized_descriptions = [desc.lower().split() for desc in descriptions]
        self.bm25 = BM25Okapi(tokenized_descriptions)
        
        # Build FAISS vector index
        self.embeddings = self.embedding_model.encode(descriptions)
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
    
    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[FoodItem], List[float]]:
        """Hybrid retrieval with nutrient-aware ranking"""
        
        # Check if this is a nutrient-specific query
        nutrient_query = self._detect_nutrient_query(query.lower())
        
        if nutrient_query:
            # For nutrient queries, prioritize foods high in that nutrient
            return self._nutrient_focused_retrieve(query, nutrient_query, top_k)
        else:
            # Use standard hybrid retrieval
            return self._standard_retrieve(query, top_k)
    
    def _detect_nutrient_query(self, query_lower: str) -> str:
        """Detect if query is asking for specific nutrients"""
        nutrient_keywords = {
            'protein': ['protein', 'growth', 'muscle'],
            'iron': ['iron', 'anemia', 'mineral'],
            'vitamin_c': ['vitamin c', 'immune', 'immunity'],
            'vitamin_a': ['vitamin a', 'vision', 'eye'],
            'fiber': ['fiber', 'digestion', 'digestive'],
            'calories': ['calories', 'energy', 'weight gain']
        }
        
        for nutrient, keywords in nutrient_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return nutrient
        return None
    
    def _nutrient_focused_retrieve(self, query: str, nutrient: str, top_k: int) -> Tuple[List[FoodItem], List[float]]:
        """Retrieve foods ranked by specific nutrient content"""
        
        # Sort foods by the requested nutrient
        nutrient_map = {
            'protein': lambda food: food.protein_g,
            'iron': lambda food: food.iron_mg,
            'vitamin_c': lambda food: food.vit_c_mg,
            'vitamin_a': lambda food: food.vit_a_ug,
            'fiber': lambda food: food.fiber_g,
            'calories': lambda food: food.kcal_100g
        }
        
        if nutrient in nutrient_map:
            # Sort by nutrient content (highest first)
            sorted_foods = sorted(self.foods, key=nutrient_map[nutrient], reverse=True)
            
            # Take top foods with significant amounts of the nutrient
            min_thresholds = {
                'protein': 5.0,  # At least 5g protein
                'iron': 1.0,     # At least 1mg iron
                'vitamin_c': 10.0, # At least 10mg vitamin C
                'vitamin_a': 50.0, # At least 50µg vitamin A
                'fiber': 2.0,    # At least 2g fiber
                'calories': 50.0  # At least 50 kcal
            }
            
            threshold = min_thresholds.get(nutrient, 0)
            high_nutrient_foods = [food for food in sorted_foods 
                                 if nutrient_map[nutrient](food) >= threshold]
            
            # If we don't have enough high-nutrient foods, include all sorted foods
            if len(high_nutrient_foods) < top_k:
                top_foods = sorted_foods[:top_k]
            else:
                top_foods = high_nutrient_foods[:top_k]
            
            # Generate scores based on nutrient content (normalized 0-1)
            max_value = max(nutrient_map[nutrient](food) for food in sorted_foods)
            min_value = min(nutrient_map[nutrient](food) for food in sorted_foods)
            
            if max_value > min_value:
                top_scores = [(nutrient_map[nutrient](food) - min_value) / (max_value - min_value) 
                             for food in top_foods]
            else:
                top_scores = [1.0] * len(top_foods)
            
            return top_foods, top_scores
        
        # Fallback to standard retrieval
        return self._standard_retrieve(query, top_k)
    
    def _standard_retrieve(self, query: str, top_k: int) -> Tuple[List[FoodItem], List[float]]:
        """Standard hybrid retrieval combining BM25 and vector search"""
        
        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Vector search scores
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        vector_scores, vector_indices = self.index.search(query_embedding.astype('float32'), len(self.foods))
        vector_scores = vector_scores[0]
        
        # Combine scores (weighted average)
        bm25_weight = 0.3
        vector_weight = 0.7
        
        # Normalize scores to [0, 1]
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-8)
        
        combined_scores = bm25_weight * bm25_scores + vector_weight * vector_scores
        
        # Get top-k results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        top_foods = [self.foods[i] for i in top_indices]
        top_scores = [combined_scores[i] for i in top_indices]
        
        return top_foods, top_scores
    
    def generate_answer(self, query: str, retrieved_foods: List[FoodItem], scores: List[float]) -> Tuple[str, str]:
        """Generate answer and confidence based on retrieved foods"""
        
        if not retrieved_foods or max(scores) < 0.3:
            internal_confidence = "Low"
            answer = f"I'm not sure about '{query}', but here's what we do know: "
            answer += f"{retrieved_foods[0].name} is a {retrieved_foods[0].category.lower()} with {retrieved_foods[0].note.lower()}"
        elif max(scores) > 0.7:
            internal_confidence = "High"
            answer = self._generate_detailed_answer(query, retrieved_foods)
        else:
            internal_confidence = "Medium"
            answer = self._generate_detailed_answer(query, retrieved_foods)
        
        # Convert to parent-friendly confidence
        parent_friendly_confidence = self._get_parent_friendly_confidence(internal_confidence, retrieved_foods)
        
        return answer, parent_friendly_confidence
    
    def _get_parent_friendly_confidence(self, internal_confidence: str, foods: List[FoodItem]) -> str:
        """Convert internal confidence to parent-friendly indicator"""
        
        # Check if we have medical authority in the notes
        has_medical_authority = any(
            any(authority in food.note for authority in ['Pediatrician-recommended', 'AAP', 'CDC', 'WHO'])
            for food in foods
        )
        
        if internal_confidence == "High":
            if has_medical_authority:
                return "Backed by medical guidelines"
            else:
                return "Well-established guidance"
        elif internal_confidence == "Medium":
            return "General guidance available"
        else:  # Low confidence
            return "Limited guidance - consult pediatrician"
    
    def _generate_detailed_answer(self, query: str, foods: List[FoodItem]) -> str:
        """Generate a detailed answer in the structured format"""
        if len(foods) == 0:
            return "❌ No relevant information found in our database.\n📚 Sources: Baby Food RAG Database"
        
        query_lower = query.lower()
        primary_food = foods[0]
        
        # Extract safety info from the comprehensive note
        safety_note = self._extract_safety_guidance(primary_food)
        prep_instructions = self._extract_prep_instructions(primary_food)
        age_info = self._extract_age_info(primary_food)
        sources = self._extract_sources(primary_food)
        
        # Generate structured response based on query type
        if any(word in query_lower for word in ['can i', 'safe', 'introduce', 'give']):
            # Safety-focused question
            answer_parts = []
            
            # Safety verdict
            if age_info:
                answer_parts.append(f"✅ Yes, {primary_food.name.lower()} can be introduced {age_info}.")
            else:
                answer_parts.append(f"✅ Yes, {primary_food.name.lower()} is generally safe for babies.")
            
            # Preparation instructions
            if prep_instructions:
                answer_parts.append(f"🥄 Prep: {prep_instructions}")
            
            # Safety warnings
            if safety_note:
                answer_parts.append(f"⚠️ Note: {safety_note}")
            
            # Sources
            answer_parts.append(f"📚 Sources: {sources}")
            
            return "\n".join(answer_parts)
            
        # ...same logic as simple_retriever for other query types...
        
        else:
            # General question
            answer_parts = []
            
            answer_parts.append(f"✅ {primary_food.name} is a {primary_food.category.lower()} suitable for babies.")
            
            prep = self._extract_prep_instructions(primary_food)
            if prep:
                answer_parts.append(f"🥄 Prep: {prep}")
                
            safety = self._extract_safety_guidance(primary_food)
            if safety:
                answer_parts.append(f"⚠️ Note: {safety}")
                
            answer_parts.append(f"📚 Sources: {self._extract_sources(primary_food)}")
            
            return "\n".join(answer_parts)
    
    def _extract_age_info(self, food: FoodItem) -> str:
        """Extract age information from food note"""
        note = food.note.lower()
        if 'safe from 6 months' in note:
            return "from 6 months"
        elif 'safe from 12 months' in note:
            return "from 12 months"
        elif '6 months' in note:
            return "from 6 months"
        elif '12 months' in note:
            return "from 12 months"
        return ""
    
    def _extract_prep_instructions(self, food: FoodItem) -> str:
        """Extract preparation instructions from food note"""
        note = food.note
        if 'how to prepare:' in note.lower():
            # Extract the preparation part
            prep_start = note.lower().find('how to prepare:') + len('how to prepare:')
            prep_end = note.find('|', prep_start)
            if prep_end == -1:
                prep_text = note[prep_start:].strip()
            else:
                prep_text = note[prep_start:prep_end].strip()
            return prep_text
        return ""
    
    def _extract_safety_guidance(self, food: FoodItem) -> str:
        """Extract safety warnings from food note"""
        note = food.note.lower()
        if 'watch out for:' in note:
            # Extract safety warning
            safety_start = note.find('watch out for:') + len('watch out for:')
            safety_end = note.find('|', safety_start)
            if safety_end == -1:
                safety_text = note[safety_start:].strip()
            else:
                safety_text = note[safety_start:safety_end].strip()
            return safety_text.capitalize()
        elif 'choking' in note:
            return "Potential choking hazard - prepare safely"
        elif 'allergy' in note:
            return "Watch for allergic reactions"
        return ""
    
    def _extract_sources(self, food: FoodItem) -> str:
        """Extract source information"""
        note = food.note
        sources = []
        
        if 'pediatrician-recommended' in note.lower():
            sources.append("AAP/CDC Guidelines")
        
        # Always include our curated database
        sources.append("Baby Food Safety Database")
        
        if not sources:
            sources.append("USDA Food Database")
        
        return ", ".join(sources)
    
    def get_citations(self, foods: List[FoodItem], scores: List[float]) -> List[Citation]:
        """Generate citations from retrieved foods"""
        citations = []
        for food, score in zip(foods[:3], scores[:3]):  # Top 3 citations
            citations.append(Citation(
                food_name=food.name,
                usda_url=food.usda_url,
                relevance_score=round(score, 3)
            ))
        return citations
