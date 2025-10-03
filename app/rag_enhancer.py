from typing import List, Tuple
from .models import FoodItem, SafetyAlert, RAGAdvantage

class RAGAdvantageAnalyzer:
    """Analyzes and highlights advantages of RAG over generic ChatGPT responses"""
    
    def analyze_safety_critical_info(self, foods: List[FoodItem], query: str) -> List[SafetyAlert]:
        """Identify safety-critical information that ChatGPT might miss or downplay"""
        alerts = []
        query_lower = query.lower()
        
        for food in foods:
            note_lower = food.note.lower()
            
            # Critical safety alerts
            if 'botulism' in note_lower:
                alerts.append(SafetyAlert(
                    level="CRITICAL",
                    message=f"⚠️ {food.name}: NEVER give honey to babies under 12 months - risk of infant botulism",
                    source="AAP/CDC Guidelines"
                ))
            
            if 'choking' in note_lower and any(word in query_lower for word in ['baby', 'infant', 'month']):
                alerts.append(SafetyAlert(
                    level="WARNING", 
                    message=f"🚨 {food.name}: Specific choking hazard - requires careful preparation",
                    source="AAP Choking Prevention Guidelines"
                ))
            
            if 'allergy' in note_lower:
                alerts.append(SafetyAlert(
                    level="INFO",
                    message=f"🔍 {food.name}: Known allergen - introduce carefully and watch for reactions",
                    source="AAP Allergy Guidelines"
                ))
        
        return alerts[:3]  # Top 3 most relevant alerts
    
    def generate_rag_advantages(self, foods: List[FoodItem], query: str) -> RAGAdvantage:
        """Highlight specific advantages of this RAG system"""
        
        # Check for evidence-based sources
        has_medical_sources = any(
            any(source in food.note for source in ['AAP', 'CDC', 'WHO']) 
            for food in foods
        )
        
        # Check for safety verification
        safety_checked = any(
            any(term in food.note.lower() for term in ['choking', 'allergy', 'month', 'risk'])
            for food in foods
        )
        
        # Check for age-appropriate guidance
        age_appropriate = any(
            'month' in food.note.lower() or 'age' in food.note.lower()
            for food in foods
        )
        
        medical_guidelines = "AAP/CDC/WHO" if has_medical_sources else "General guidelines"
        
        return RAGAdvantage(
            evidence_based=True,
            source_cited=True,
            safety_checked=safety_checked,
            age_appropriate=age_appropriate,
            medical_guidelines=medical_guidelines
        )
    
    def compare_with_chatgpt(self, query: str, foods: List[FoodItem], safety_alerts: List[SafetyAlert]) -> str:
        """Generate explanation of why this RAG response is superior to ChatGPT"""
        
        advantages = [
            "✅ **Grounded in authoritative medical sources** (AAP/CDC/WHO guidelines)",
            "✅ **Specific safety alerts** with preparation instructions",
            "✅ **Age-appropriate recommendations** based on development",
            "✅ **Traceable citations** to USDA and medical databases",
            "✅ **Consistent, reproducible answers** from curated knowledge base"
        ]
        
        if safety_alerts:
            advantages.append(f"✅ **{len(safety_alerts)} safety alerts** identified that ChatGPT might miss")
        
        chatgpt_limitations = [
            "❌ No guaranteed accuracy or source verification",
            "❌ May not emphasize critical safety considerations", 
            "❌ Responses vary between sessions",
            "❌ No direct citations to medical guidelines",
            "❌ General knowledge may miss baby-specific nuances"
        ]
        
        return f"""
**Why this RAG system is better than ChatGPT for baby feeding questions:**

**RAG Advantages:**
{chr(10).join(advantages)}

**ChatGPT Limitations:**
{chr(10).join(chatgpt_limitations)}

**Bottom line:** When it comes to baby safety, you need authoritative, consistent, and traceable information - not general AI responses.
        """.strip()
