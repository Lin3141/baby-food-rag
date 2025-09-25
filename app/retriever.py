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
        """Hybrid retrieval combining BM25 and vector search"""
        
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
            confidence = "Low"
            answer = f"I'm not sure about '{query}', but here's what we do know: "
            answer += f"{retrieved_foods[0].name} is a {retrieved_foods[0].category.lower()} with {retrieved_foods[0].note.lower()}"
        elif max(scores) > 0.7:
            confidence = "High"
            answer = self._generate_detailed_answer(query, retrieved_foods)
        else:
            confidence = "Medium"
            answer = self._generate_detailed_answer(query, retrieved_foods)
        
        return answer, confidence
    
    def _generate_detailed_answer(self, query: str, foods: List[FoodItem]) -> str:
        """Generate a detailed answer based on retrieved foods"""
        if len(foods) == 0:
            return "No relevant information found."
        
        # Simple rule-based answer generation
        if any(word in query.lower() for word in ['iron', 'anemia', 'mineral']):
            iron_foods = sorted(foods, key=lambda x: x.iron_mg, reverse=True)[:2]
            answer = f"For iron content, {iron_foods[0].name} contains {iron_foods[0].iron_mg}mg iron per 100g"
            if len(iron_foods) > 1:
                answer += f", and {iron_foods[1].name} has {iron_foods[1].iron_mg}mg iron per 100g"
        elif any(word in query.lower() for word in ['vitamin a', 'vision', 'eye']):
            vita_foods = sorted(foods, key=lambda x: x.vit_a_ug, reverse=True)[:2]
            answer = f"For Vitamin A, {vita_foods[0].name} provides {vita_foods[0].vit_a_ug}µg per 100g"
        elif any(word in query.lower() for word in ['vitamin c', 'immune', 'immunity']):
            vitc_foods = sorted(foods, key=lambda x: x.vit_c_mg, reverse=True)[:2]
            answer = f"For Vitamin C, {vitc_foods[0].name} contains {vitc_foods[0].vit_c_mg}mg per 100g"
        elif any(word in query.lower() for word in ['protein', 'growth']):
            protein_foods = sorted(foods, key=lambda x: x.protein_g, reverse=True)[:2]
            answer = f"For protein, {protein_foods[0].name} provides {protein_foods[0].protein_g}g per 100g"
        elif any(word in query.lower() for word in ['fiber', 'digestion', 'digestive']):
            fiber_foods = sorted(foods, key=lambda x: x.fiber_g, reverse=True)[:2]
            answer = f"For fiber, {fiber_foods[0].name} contains {fiber_foods[0].fiber_g}g per 100g"
        else:
            # General answer
            answer = f"Based on your question, {foods[0].name} seems most relevant. {foods[0].note}"
        
        return answer
    
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
