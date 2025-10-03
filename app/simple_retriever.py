import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from typing import List, Tuple
from .models import FoodItem, Citation

class SimpleHybridRetriever:
    def __init__(self, foods: List[FoodItem], descriptions: List[str]):
        self.foods = foods
        self.descriptions = descriptions
        
        # Build BM25 index
        tokenized_descriptions = [desc.lower().split() for desc in descriptions]
        self.bm25 = BM25Okapi(tokenized_descriptions)
        
        # Build TF-IDF index (simpler alternative to sentence transformers)
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions)
    
    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[FoodItem], List[float]]:
        """Hybrid retrieval combining BM25 and TF-IDF with nutrient-aware ranking"""
        
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
        """Standard hybrid retrieval combining BM25 and TF-IDF"""
        
        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # TF-IDF scores
        query_vector = self.tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Combine scores (weighted average)
        bm25_weight = 0.4
        tfidf_weight = 0.6
        
        # Normalize scores to [0, 1]
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_scores = np.ones_like(bm25_scores) * 0.5
        
        if tfidf_scores.max() > tfidf_scores.min():
            tfidf_scores = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min())
        else:
            tfidf_scores = np.ones_like(tfidf_scores) * 0.5
        
        combined_scores = bm25_weight * bm25_scores + tfidf_weight * tfidf_scores
        
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
        """Generate response with clear visual hierarchy"""
        if len(foods) == 0:
            return "❌ **No information found**\n📚 **Source:** Database"
        
        primary_food = foods[0]
        food_name = primary_food.name.lower()
        query_lower = query.lower()
        
        response_parts = []
        
        # 1. SAFETY VERDICT (✅ + bold)
        age_info = self._extract_age_info(primary_food)
        if 'can i' in query_lower or 'safe' in query_lower:
            if age_info:
                response_parts.append(f"✅ **Yes, {food_name} is safe** {age_info}")
            else:
                response_parts.append(f"✅ **Yes, {food_name} is safe** for babies")
        else:
            # Nutrient-focused responses without icons
            if 'protein' in query_lower:
                response_parts.append(f"**{primary_food.name} provides** {primary_food.protein_g}g protein per 100g")
            elif 'iron' in query_lower:
                response_parts.append(f"**{primary_food.name} contains** {primary_food.iron_mg}mg iron per 100g")
            elif 'vitamin c' in query_lower:
                response_parts.append(f"**{primary_food.name} has** {primary_food.vit_c_mg}mg vitamin C per 100g")
            else:
                response_parts.append(f"✅ **{primary_food.name} is** a nutritious {primary_food.category.lower()}")
        
        # 2. WHY IT MATTERS (reasoning without icon)
        why_explanation = self._get_why_it_matters(primary_food, query_lower)
        if why_explanation:
            response_parts.append(f"\n**Why it matters:** {why_explanation}")
        
        # 3. PREPARATION (spacing without icon)
        prep = self._get_simple_prep_instruction(primary_food)
        if prep:
            response_parts.append(f"\n**Prep:** {prep}")
        
        # 4. KEY WARNING (bold without icon)
        warning = self._get_simple_warning(primary_food)
        if warning:
            response_parts.append(f"\n**Note:** {warning}")
        
        # 5. BENEFIT (benefit info without icon)
        benefit = self._get_nutritional_benefit(primary_food)
        if benefit:
            response_parts.append(f"\n**Benefit:** {benefit}")
        
        # 6. ACTIONABLE NEXT STEP (practical action without icon)
        action_step = self._get_actionable_next_step(primary_food, query_lower)
        if action_step:
            # Remove icon from action step
            clean_action = action_step.replace("👍 **Next step:**", "**Next step:**").replace("🚫 **Next step:**", "**Next step:**").replace("⚠️ **Next step:**", "**Next step:**")
            response_parts.append(f"\n{clean_action}")
        
        # 7. SOURCE (clean without icon)
        response_parts.append("\n**Sources:** AAP/CDC Guidelines")
        
        return "".join(response_parts)
    
    def _get_actionable_next_step(self, food: FoodItem, query: str) -> str:
        """Generate practical next step for parents"""
        food_name = food.name.lower()
        note_lower = food.note.lower()
        
        # Safety-first actions
        if 'honey' in food_name:
            return "🚫 **Next step:** Avoid before 12 months. Try maple syrup or mashed banana for sweetness instead."
        
        if 'choking' in note_lower:
            if 'grape' in food_name:
                return "⚠️ **Next step:** Always quarter grapes lengthwise. Never give whole grapes."
            else:
                return "⚠️ **Next step:** Prepare safely to avoid choking. Always supervise eating."
        
        # Query-specific actions
        if 'first food' in query or 'start' in query:
            first_food_actions = {
                'banana': "👍 **Next step:** Perfect starter! Try mashed banana mixed into baby cereal at breakfast.",
                'apple': "👍 **Next step:** Steam and mash smooth. Great first fruit to introduce.",
                'sweet potato': "👍 **Next step:** Steam until very soft, mash smooth. Excellent first vegetable.",
                'rice cereal': "👍 **Next step:** Mix thin with breast milk. Traditional first food choice.",
                'avocado': "👍 **Next step:** Mash ripe avocado smooth. Rich, creamy first food."
            }
            if food_name in first_food_actions:
                return first_food_actions[food_name]
        
        # Food-specific practical actions
        actions = {
            'banana': "👍 **Next step:** Try at breakfast mashed into oatmeal or as finger food strips.",
            'apple': "👍 **Next step:** Steam and mash, or try as soft cooked pieces for texture practice.",
            'chicken': "👍 **Next step:** Cook thoroughly, shred finely, and mix with favorite vegetables.",
            'salmon': "👍 **Next step:** Cook well, check carefully for bones, and flake into small pieces.",
            'egg': "👍 **Next step:** Scramble well-cooked and try at breakfast. Great protein source.",
            'yogurt': "👍 **Next step:** Offer plain whole-milk yogurt as snack or mixed with fruit.",
            'spinach': "👍 **Next step:** Steam and puree, then mix into pasta or rice dishes.",
            'rice cereal': "👍 **Next step:** Start thin, gradually thicken as baby adapts to textures."
        }
        
        if food_name in actions:
            return actions[food_name]
        
        # Generic action based on food category
        if food.category.lower() == 'fruit':
            return "👍 **Next step:** Try mashed first, then soft pieces as baby develops chewing skills."
        elif food.category.lower() == 'vegetable':
            return "👍 **Next step:** Steam until very soft, start with puree, progress to small pieces."
        elif food.category.lower() == 'protein':
            return "👍 **Next step:** Cook thoroughly and start with very small, soft pieces."
        else:
            return "👍 **Next step:** Start with small portions and watch for baby's reaction."

    def _get_why_it_matters(self, food: FoodItem, query: str) -> str:
        """Explain why this food recommendation matters"""
        food_name = food.name.lower()
        
        # Query-specific explanations
        if 'first food' in query or 'start' in query:
            if food_name in ['banana', 'apple', 'sweet potato', 'avocado']:
                return f'{food.name} is recommended as a first food because it\'s naturally soft, easy to digest, and gentle on developing stomachs'
            elif food_name in ['rice cereal', 'oatmeal']:
                return 'Iron-fortified cereals are often first foods because they help prevent iron deficiency as your baby\'s iron stores from birth begin to deplete'
        
        if 'iron' in query:
            return 'Iron is crucial after 6 months because babies\' iron stores from birth are depleting and breast milk alone may not provide enough'
        elif 'protein' in query:
            return 'Protein provides the building blocks for rapid growth and brain development during your baby\'s first year'
        elif 'vitamin c' in query:
            return 'Vitamin C supports immune system development and helps your baby absorb iron from other foods'
        
        # Food-specific explanations
        explanations = {
            'banana': 'Bananas are gentle first foods - naturally sweet, easy to mash, and rich in potassium for healthy muscle development',
            'avocado': 'Avocados provide healthy fats essential for brain development during this critical growth period',
            'apple': 'Apples introduce natural sweetness and fiber, helping develop taste preferences for healthy foods',
            'sweet potato': 'Sweet potatoes are naturally sweet and packed with beta-carotene for healthy vision development',
            'chicken': 'Chicken provides complete protein with all amino acids needed for your baby\'s rapid growth',
            'salmon': 'Salmon offers omega-3s crucial for brain development during the first year of life',
            'egg': 'Eggs are nutritional powerhouses and early introduction may help prevent allergies later',
            'spinach': 'Leafy greens provide iron and folate for healthy blood development, though portions should be small for young babies'
        }
        
        return explanations.get(food_name, f'{food.name} provides important nutrients during your baby\'s critical development phase')

    def _get_simple_prep_instruction(self, food: FoodItem) -> str:
        """Get simplified preparation instruction from food note"""
        note = food.note
        # Look for common prep indicators
        if 'steam' in note or 'boil' in note or 'cook' in note:
            return "Cook until soft, then mash or puree"
        elif 'raw' in note or 'fresh' in note:
            return "Serve raw, ensure it's clean and safe"
        elif 'bake' in note:
            return "Bake until soft, then mash or puree"
        elif 'sauté' in note or 'stir-fry' in note:
            return "Cook quickly in a small amount of oil"
        return "Prepare as desired, ensure it's age-appropriate"
    
    def _get_simple_warning(self, food: FoodItem) -> str:
        """Get simplified warning from food note"""
        note = food.note.lower()
        if 'choking' in note:
            return "Potential choking hazard - ensure proper preparation"
        elif 'allergy' in note:
            return "Watch for potential allergic reactions"
        elif 'iron' in note and 'spinach' in note:
            return "High in oxalates; serve in moderation and ensure a balanced diet"
        return ""
    
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
